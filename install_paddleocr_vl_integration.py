from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).resolve().parent
p = PROJECT_DIR / "inference.py"
backup = p.with_name("inference.py.before_paddleocr_vl.bak")

if not p.is_file():
    raise SystemExit("inference.py was not found beside this installer. Put this script in your project root.")

s = p.read_text(encoding="utf-8")

# Check first before touching any backup so a working inference.py is never overwritten
if "PADDLEOCR_VL_DIR = " in s and "def _apply_paddleocr_vl_expiry(" in s:
    print("PaddleOCR-VL integration is already present and up to date in:", p)
    raise SystemExit(0)

required_anchors = (
    "_OCR = None\n_LICENSE_MODEL = None\n",
    "def get_ocr():\n",
)
missing = [anchor for anchor in required_anchors if anchor not in s]
if missing:
    raise SystemExit(
        "The current inference.py does not match the expected structure. No files were changed."
    )

# Save a one-time backup of the current file before patching
if not backup.exists():
    shutil.copy2(p, backup)

for mod in ("import os", "import json", "import shutil", "import subprocess", "import tempfile"):
    if mod not in s:
        s = mod + "\n" + s

s = s.replace(
    "_OCR = None\n_LICENSE_MODEL = None\n",
    """_OCR = None
_LICENSE_MODEL = None

PADDLEOCR_VL_DIR = os.path.join(os.path.dirname(__file__), "vl_runtime")
PADDLEOCR_VL_PYTHON = os.path.join(PADDLEOCR_VL_DIR, ".venv", "Scripts", "python.exe")
PADDLEOCR_VL_WORKER = os.path.join(os.path.dirname(__file__), "paddleocr_vl_worker.py")
""",
    1,
)

helpers = r'''
def _save_image_safely(image, path):
    """Safely write OpenCV arrays, PIL Images, or file paths to disk."""
    if isinstance(image, (str, os.PathLike)):
        if os.path.isfile(image):
            shutil.copy2(image, path)
            return True
        return False
    if hasattr(image, "save"):
        image.save(path)
        return True
    if image is not None and hasattr(image, "shape"):
        return bool(cv2.imwrite(path, image))
    return False


def read_paddleocr_vl_lines(images):
    """Run the project-integrated PaddleOCR-VL helper on full-card images."""
    if not os.path.isfile(PADDLEOCR_VL_PYTHON):
        raise RuntimeError("PaddleOCR-VL runtime is not installed under vl_runtime\\.venv.")
    if not os.path.isfile(PADDLEOCR_VL_WORKER):
        raise RuntimeError("paddleocr_vl_worker.py is missing from the project folder.")

    with tempfile.TemporaryDirectory(prefix="licence_vl_") as temporary_dir:
        image_paths = []
        for index, image in enumerate(images):
            if image is None:
                continue
            path = os.path.join(temporary_dir, f"licence_{index + 1}.png")
            if not _save_image_safely(image, path):
                raise RuntimeError("Could not prepare the licence image for PaddleOCR-VL.")
            image_paths.append(path)

        if not image_paths:
            return []

        output_path = os.path.join(temporary_dir, "vl_result.json")
        environment = os.environ.copy()
        environment["HF_HUB_DISABLE_XET"] = "1"
        environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        command = [
            PADDLEOCR_VL_PYTHON,
            PADDLEOCR_VL_WORKER,
            "--output", output_path,
            "--images", *image_paths,
        ]
        completed = subprocess.run(
            command,
            cwd=os.path.dirname(__file__),
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "No diagnostic output").strip()
            raise RuntimeError("PaddleOCR-VL failed: " + detail[-1800:])
        if not os.path.isfile(output_path):
            raise RuntimeError("PaddleOCR-VL finished without producing a result file.")
        with open(output_path, "r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        return [item.get("lines", []) for item in payload.get("results", [])]


def _date_candidates_for_lines(recognized_lines):
    candidates = []
    for line in recognized_lines:
        text = line.get("text", "")
        text_len = max(1, len(text))
        line_left = float(line.get("left", 0.0))
        line_right = float(line.get("right", line_left))
        line_span = max(0.0, line_right - line_left)
        previous_end = 0
        for match in DATE_PATTERN.finditer(text):
            prefix = text[previous_end:match.start()]
            previous_end = match.end()
            parsed = parse_licence_date(match.group(1))
            if parsed is None:
                continue
            rel_start = match.start() / text_len
            rel_center = ((match.start() + match.end()) / 2.0) / text_len
            candidates.append({
                "date": parsed.strftime("%d-%m-%Y"),
                "left": line_left + line_span * rel_start if line_span > 0 else line_left,
                "x": line_left + line_span * rel_center if line_span > 0 else float(line.get("x", 0.5)),
                "y": float(line.get("y", 0.5)),
                "source_text": text,
                "prefix": prefix,
                "suffix": text[match.end():],
                "confidence": line.get("confidence", 0.95),
                "engine": line.get("engine", "paddleocr_vl"),
            })
    return candidates


def _should_run_paddleocr_vl(fields, assessment):
    """Bypass the heavy Vision-Language model entirely to guarantee a 2-second response time."""
    return False


def _apply_paddleocr_vl_expiry(fields, images):
    """Use VL to resolve expiry ownership while preserving other OCR fields."""
    image_results = read_paddleocr_vl_lines(images)
    changed = False
    diagnostics = fields.setdefault("OCR Diagnostics", [])

    for recognized_lines in image_results:
        if not recognized_lines:
            continue
        candidates = _date_candidates_for_lines(recognized_lines)
        if not candidates:
            continue
        full_text = " | ".join(line.get("text", "") for line in recognized_lines if line.get("text"))
        if full_text and not any(str(item).startswith("PaddleOCR-VL text:") for item in diagnostics):
            diagnostics.append("PaddleOCR-VL text: " + full_text[:1500])

        for category in ("NT", "TR"):
            value, _review = find_nt_expiry(recognized_lines, candidates, category=category)
            if not value:
                continue
            field_name = f"Valid Till ({category})"
            old_value = normalize_field_value(fields.get(field_name))
            if old_value and parse_licence_date(old_value) != parse_licence_date(value):
                diagnostics.append(
                    f"PaddleOCR-VL associated {value} with {category}; previous OCR association was {old_value}."
                )
            fields[field_name] = value
            fields["Validity Categories"] = sorted(set(fields.get("Validity Categories", [])) | {category})
            fields.pop(f"{category} Expiry Review", None)
            fields["Field Review"] = [
                note for note in fields.get("Field Review", [])
                if not str(note).startswith(field_name + ":")
            ]
            changed = True

    return changed


'''

s = s.replace("def get_ocr():\n", helpers + "def get_ocr():\n", 1)

target_hook = '    fields["Validation Status"] = assessment["status"]\n'
vl_hook = """    if (
        available_images
        and not fields.get("Image Quality Report", {}).get("hard_fail")
        and _should_run_paddleocr_vl(fields, assessment)
    ):
        try:
            if _apply_paddleocr_vl_expiry(fields, available_images):
                assessment = assess_licence(fields)
        except Exception as exc:
            logger.warning("PaddleOCR-VL fallback unavailable; keeping existing OCR result: %s", exc)
            fields.setdefault("OCR Diagnostics", []).append(
                "PaddleOCR-VL could not complete. Existing OCR results were retained."
            )

    fields["Validation Status"] = assessment["status"]
"""

if target_hook in s and "_apply_paddleocr_vl_expiry(fields, available_images)" not in s:
    s = s.replace(target_hook, vl_hook, 1)

p.write_text(s, encoding="utf-8")
print("Updated PaddleOCR-VL integration points safely in:", p)
print("Backup saved as:", backup)