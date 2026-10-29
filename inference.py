import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date, datetime

import cv2
import numpy as np
import torch  # noqa: F401  # load torch DLLs before PaddleOCR on Windows
from huggingface_hub import hf_hub_download
from paddleocr import PaddleOCR
from ultralytics import YOLO

from validation_rules import (
    PARIVAHAN_PORTAL_URL,
    check_image_quality,
    evaluate_licence_validation,
    validate_selected_category_expiry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

HF_REPO = "logasanjeev/indian-id-validator"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DL_NUMBER_PATTERN = re.compile(r"\b([A-Z]{2}[-\s]?[0-9A-Z]{2,3}[-\s]?\d{4}[-\s]?\d{5,7})\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4}\s*-\s*\d{2}\s*-\s*\d{2}|\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})(?!\d)"
)
SUPPORTED_STATE_CODES = frozenset(
    "AN AP AR AS BR CH CG DL GA GJ HR HP JK JH KA KL LA LD MP MH MN ML MZ NL OD OR PY PB RJ SK TN TG TS TR DD UK UA UP WB".split()
)
DATE_FORMATS = ("%d-%m-%Y", "%d/%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y")
VALID_BLOOD_GROUP_PATTERN = re.compile(r"\b((?:A1B|A2B|AB|A1|A2|B1|A|B|O)\s*[+-])(?![A-Z0-9])", re.I)


def parse_licence_date(value):
    value = re.sub(r"\s*([./-])\s*", r"\1", normalize_field_value(value))
    for format_string in DATE_FORMATS:
        try:
            return datetime.strptime(value, format_string).date()
        except ValueError:
            continue
    return None


DISPLAY_FIELDS = [
    "DL No",
    "Name",
    "DOB",
    "Date of Issue",
    "Date of First Issue",
    "Valid Till (NT)",
    "Valid Till (TR)",
    "Valid Till",
    "Blood Group",
    "Organ Donor",
    "Relation With",
    "Address",
    "RTO",
    "State",
    "Vehicle Type",
]


def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        config_path = hf_hub_download(repo_id=HF_REPO, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()
_OCR = None
_LICENSE_MODEL = None

PADDLEOCR_VL_DIR = os.path.join(os.path.dirname(__file__), "vl_runtime")
PADDLEOCR_VL_PYTHON = os.path.join(PADDLEOCR_VL_DIR, ".venv", "Scripts", "python.exe")
PADDLEOCR_VL_WORKER = os.path.join(os.path.dirname(__file__), "paddleocr_vl_worker.py")


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


def get_ocr():
    global _OCR
    if _OCR is None:
        try:
            _OCR = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                enable_mkldnn=False,
            )
        except TypeError:
            _OCR = PaddleOCR(lang="en")
    return _OCR


def load_license_model():
    global _LICENSE_MODEL
    if _LICENSE_MODEL is not None:
        return _LICENSE_MODEL

    model_path = CONFIG["models"]["Driving_License"]["path"]
    local_path = os.path.join(os.path.dirname(__file__), model_path)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        model_path = local_path
    else:
        model_path = hf_hub_download(repo_id=HF_REPO, filename=model_path)
    _LICENSE_MODEL = YOLO(model_path)
    return _LICENSE_MODEL


def upscale_image(image, scale=2):
    return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_CUBIC)


def unblur_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(image):
    image = upscale_image(image, scale=2)
    image = unblur_image(image)
    image = denoise_image(image)
    image = enhance_contrast(image)
    return image


def read_image(image_input):
    """Read image into standard BGR format without destructive blind rotation."""
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Failed to load image: {image_input}")
        return image

    if not isinstance(image_input, np.ndarray) or image_input.size == 0:
        raise ValueError("A valid driving licence image is required.")

    image = image_input
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def extract_text_from_ocr(ocr_result):
    extracted_text = []
    if not ocr_result:
        return "No text detected"

    for line in ocr_result:
        if line is None:
            continue
        for word_info in line:
            if word_info is None or len(word_info) < 2 or not word_info[1]:
                continue
            extracted_text.append(str(word_info[1][0]).strip())
    return " ".join(extracted_text) if extracted_text else "No text detected"


def find_nt_expiry(recognized_lines, date_candidates, category="NT", excluded_dates=None):
    excluded_dates = set(excluded_dates or ())
    validity = re.compile(r"\b(valid(?:ity|\s*(?:till|upto|up\s+to|until|through))|expiry|expires)\b", re.I)
    nt = re.compile(r"(?:\b|\()(?:NT|NON[ -]*TRANSPORT)(?:\b|\))" if category == "NT"
                    else r"(?:\b|\()(?:T|TR|TRANSPORT)(?:\b|\))", re.I)
    other_field = re.compile(r"\b(issue|issued|doi|aed|birth|dob|valid\s+from)\b", re.I)
    categories = re.compile(r"(?:\b|\()(?:NT|NON[ -]*TRANSPORT|TR|T|TRANSPORT)(?:\b|\))", re.I)
    standalone = re.compile(r"[\s():;,.\[\]-]*(?:" + categories.pattern + r")[\s():;,.\[\]-]*", re.I)

    def suffix_category(candidate):
        match = re.match(r"^[\s:;,.\-(\[]*" + categories.pattern, candidate.get("suffix", ""), re.I)
        if not match:
            return ""
        return "NT" if re.search(r"(?:NT|NON[ -]*TRANSPORT)", match.group(), re.I) else "TR"

    def eligible(candidate):
        if candidate["date"] in excluded_dates:
            return False
        if other_field.search(candidate.get("prefix", candidate["source_text"])):
            return False

        nearby_other_fields = [
            line for line in recognized_lines
            if other_field.search(line["text"])
            and not validity.search(line["text"])
            and not DATE_PATTERN.search(line["text"])
            and nearby_value(line, candidate)
        ]
        if not nearby_other_fields:
            return True

        if validity.search(candidate["source_text"]):
            return True

        nearby_validity_labels = [
            line for line in recognized_lines
            if validity.search(line["text"]) and nearby_value(line, candidate)
        ]
        if nearby_validity_labels:
            closest_validity = min(distance(line, candidate) for line in nearby_validity_labels)
            closest_other = min(distance(line, candidate) for line in nearby_other_fields)
            return closest_validity <= closest_other
        return False

    def confident(*items):
        return all(item.get("confidence") is not None and item["confidence"] >= 0.45 for item in items)

    def distance(first, second):
        return (first["x"] - second["x"]) ** 2 + (first["y"] - second["y"]) ** 2

    validity_labels = [line for line in recognized_lines if validity.search(line["text"])]
    anchors = []
    for line in recognized_lines:
        if not nt.search(line["text"]):
            continue
        if category == "TR" and re.search(r"\bNON[ -]*TRANSPORT\b", line["text"], re.I):
            continue
        if standalone.fullmatch(line["text"]) or validity.search(line["text"]) or any(
            abs(line["x"] - label["x"]) <= 0.25 and abs(line["y"] - label["y"]) <= 0.16
            for label in validity_labels
        ):
            anchors.append(line)

    owners = [line for line in recognized_lines if validity.search(line["text"])
              or other_field.search(line["text"])
              or categories.search(line["text"])]
    category_owners = [owner for owner in owners if not (
        validity.search(owner["text"])
        and not re.search(r"(?:\b|\()(?:NT|T|TR|TRANSPORT|NON[ -]*TRANSPORT)(?:\b|\))", owner["text"], re.I)
    )]
    matches = [(candidate["date"], confident(candidate)) for candidate in date_candidates
               if eligible(candidate) and suffix_category(candidate) == category]
    for anchor in anchors:
        if any(candidate["source_text"] == anchor["text"] and suffix_category(candidate)
               for candidate in date_candidates):
            continue
        has_heading = validity.search(anchor["text"]) or any(
            abs(anchor["x"] - label["x"]) <= .25 and abs(anchor["y"] - label["y"]) <= .16
            for label in validity_labels
        )
        preceding = []
        for candidate in date_candidates:
            if not eligible(candidate) or suffix_category(candidate):
                continue
            dx, dy = candidate["x"] - anchor["x"], candidate["y"] - anchor["y"]
            same_box = candidate["source_text"] == anchor["text"]
            before_suffix = standalone.fullmatch(anchor["text"]) and (
                (-.32 <= dx < 0 and abs(dy) <= .035)
                or (-.08 <= dy < 0 and abs(dx) <= .14)
            )
            after_label = has_heading and (same_box or (0 <= dx <= .32 and abs(dy) <= .045)
                                          or (0 < dy <= .16 and abs(dx) <= .14))
            if not (before_suffix or after_label):
                continue
            if not same_box and any(
                owner is not anchor and distance(owner, candidate) < distance(anchor, candidate)
                for owner in category_owners
            ):
                continue
            if before_suffix:
                preceding.append(candidate)
            else:
                matches.append((candidate["date"], confident(anchor, candidate)))
        if preceding:
            nearest = min(distance(anchor, candidate) for candidate in preceding)
            matches.extend((candidate["date"], confident(anchor, candidate)) for candidate in preceding
                           if abs(distance(anchor, candidate) - nearest) <= 1e-9)

    values = {value for value, _ in matches}
    if len(values) > 1:
        # When an issue date and future expiry date both sit under a combined header row, pick the future expiry date
        today = date.today()
        future_vals = [v for v in values if (parse_licence_date(v) or today) > today]
        if len(future_vals) == 1:
            values = {future_vals[0]}
        else:
            return "", f"Conflicting {category} expiry dates were read. Compare them with the licence."
    if not values:
        return "", ""
    value = next(iter(values))
    if not any(conf for candidate, conf in matches if candidate == value):
        return value, f"The {category} expiry date has low or unavailable OCR confidence. Confirm it on the licence."
    return value, ""


def read_ocr_lines(image):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A valid licence image is required.")

    image_height, image_width = image.shape[:2]
    ocr_result = get_ocr().ocr(image, cls=True)
    if not ocr_result:
        return []

    recognized_lines = []
    for page in ocr_result:
        if not page:
            continue
        for entry in page:
            if not entry or len(entry) < 2 or not entry[1]:
                continue
            text = str(entry[1][0]).strip()
            if not text:
                continue
            try:
                coordinates = entry[0]
                x_values = [float(point[0]) for point in coordinates]
                y_values = [float(point[1]) for point in coordinates]
            except (IndexError, TypeError, ValueError):
                continue
            if not x_values or not y_values:
                continue
            try:
                score = float(entry[1][1])
                score = score if np.isfinite(score) and 0 <= score <= 1 else None
            except (IndexError, TypeError, ValueError):
                score = None
            recognized_lines.append({
                "text": text,
                "confidence": score,
                "left": min(x_values) / max(image_width, 1),
                "right": max(x_values) / max(image_width, 1),
                "top": min(y_values) / max(image_height, 1),
                "bottom": max(y_values) / max(image_height, 1),
                "x": sum(x_values) / len(x_values) / max(image_width, 1),
                "y": sum(y_values) / len(y_values) / max(image_height, 1),
            })

    if not recognized_lines:
        return []

    return sorted(recognized_lines, key=lambda line: (line["y"], line["x"]))


INDIAN_STATES_UTS = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
)
ISSUE_LABEL = re.compile(r"\b(?:date\s+of\s+issue|issue\s+date|date\s+issued|doi)\b", re.I)
FIRST_ISSUE_LABEL = re.compile(r"\bdate\s+of\s+first\s+issue\b", re.I)

FIELD_LABEL = re.compile(
    r"\b(?:address|blood\s*(?:group|grp)|bg|[o0]rgan\s*don[oa]r|name|dob|date\s+of\s+birth|"
    r"birth|issue\s+date|date\s+(?:of\s+)?(?:first\s+)?issue|doi|aed|valid(?:ity|\s+till|\s+from)|"
    r"expiry|rto|state|vehicle|holder.?s?\s+signature|signature|son|daughter|wife)\b|\bS\s*[/\\]\s*[WD]",
    re.I,
)

ADDRESS_BOUNDARY = re.compile(
    r"\b(?:blood\s*(?:group|grp)|bg|[o0]rgan\s*don[oa]r|dob|date\s+of\s+birth|doi|"
    r"issue\s+date|date\s+of\s+(?:first\s+)?issue|valid(?:ity|\s+till)|"
    r"rto|signature|vehicle\s+type|DL\s*(?:No|Number))\b|\b(?:name|state)\s*:", re.I,
)


def extract_state(text):
    text = str(text).replace("&", " and ")
    found = []
    for state in sorted(INDIAN_STATES_UTS, key=len, reverse=True):
        pattern = r"\b" + r"\s*".join(re.escape(word) for word in state.split()) + r"\b"
        match = re.search(pattern, text, re.I)
        if match:
            found.append(state)
            text = text[:match.start()] + " " + text[match.end():]
    return found[0] if len(set(found)) == 1 else ""


def nearby_value(label, candidate):
    if label is candidate:
        return True
    dx, dy = candidate["x"] - label["x"], candidate["y"] - label["y"]
    return ((candidate["left"] >= label["right"] - .02 and 0 <= dx <= .44 and abs(dy) <= .048)
            or (0 < dy <= .09 and abs(candidate["left"] - label["left"]) <= .15))


def choose_field(output, field, candidates):
    if not candidates:
        return
    values = {value for value, _ in candidates}
    if len(values) == 1 and any(confident for _, confident in candidates):
        output[field] = next(iter(values))
    else:
        output[field] = ""
        reason = "conflicting OCR reads" if len(values) > 1 else "low or unavailable OCR confidence"
        output.setdefault("Field Review", []).append(f"{field}: {reason}. Confirm it on the licence.")


def extract_labelled_fields(lines):
    output = {}
    labels = [
        line for line in lines
        if FIELD_LABEL.search(line["text"]) and not re.search(r"\bsignature\b", line["text"], re.I)
    ]
    specs = {
        "DL No": (re.compile(r"^(?:DL\s*(?:No\.?|Number)|licen[cs]e\s*(?:No\.?|Number))\s*[:.-]?", re.I), r"[A-Z]{2}[A-Z0-9 /-]{5,25}"),
        "Name": (re.compile(r"^(?:name(?:\s+of\s+(?:the\s+)?holder)?|holder[’']?s?\s+name)\s*[:.-]?", re.I), r"[A-Z][A-Z .’'-]{1,79}"),
        "Valid Till": (re.compile(r"\b(?:valid\s+(?:till|upto|up\s+to|until|through)|validity|expiry\s+date|date\s+of\s+expiry|expires)\b\s*[:.-]?", re.I), DATE_PATTERN.pattern),
        "Blood Group": (re.compile(r"\b(?:blood\s*(?:group|grp)|bg)\b\s*[:.-]?", re.I), r"(?:A1B|A2B|AB|A1|A2|B1|A|B|O)\s*[+-]"),
        "Organ Donor": (re.compile(r"\b[o0]rgan\s*don[oa]r\b\s*[:.-]?", re.I), r"(?:YES|NO|Y|N)"),
        "DOB": (re.compile(r"\b(?:DOB|date\s+of\s+birth|birth\s+date)\b\s*[:.-]?", re.I), DATE_PATTERN.pattern),
        "Date of Issue": (ISSUE_LABEL, DATE_PATTERN.pattern),
        "Date of First Issue": (FIRST_ISSUE_LABEL, DATE_PATTERN.pattern),
    }
    for field, (pattern, value_pattern) in specs.items():
        candidates = []
        scores = []
        min_conf = 0.45 if field in ("DL No", "Name", "DOB") else 0.30
        for label in lines:
            match = pattern.search(label["text"])
            if not match:
                continue
            if field == "Valid Till" and re.search(r"(?:\b|\()(?:NT|TR|T|TRANSPORT)(?:\b|\))", label["text"], re.I):
                continue
            for line in lines:
                if not nearby_value(label, line):
                    continue
                if line is not label:
                    if FIELD_LABEL.search(line["text"]) or re.search(r"(?:\b|\()(?:NT|TR|T|TRANSPORT)(?:\b|\))", line["text"], re.I):
                        continue
                    if any(owner is not label and nearby_value(owner, line)
                           and abs(owner["x"] - line["x"]) + abs(owner["y"] - line["y"])
                           < abs(label["x"] - line["x"]) + abs(label["y"] - line["y"])
                           for owner in labels):
                        continue
                text = label["text"][match.end():].lstrip(" :.-") if line is label else line["text"]
                next_label = FIELD_LABEL.search(text)
                if next_label:
                    text = text[:next_label.start()]
                value_match = re.fullmatch(r"\s*(" + value_pattern + r")\s*[;:]*\s*", text, re.I)
                if not value_match and field == "Date of Issue":
                    # On tabular cards where 2 dates sit on the row below Issue Date & Validity(NT), take the first date
                    first_date_m = DATE_PATTERN.search(text)
                    if first_date_m and abs(line["left"] - label["left"]) <= 0.12:
                        value_match = first_date_m
                if not value_match:
                    continue
                value = value_match.group(1)
                if field.startswith("Date") or field in ("Valid Till", "DOB"):
                    parsed_date = parse_licence_date(value)
                    if parsed_date is None:
                        output.setdefault("Field Review", []).append(f"{field}: invalid calendar date. Upload a clearer image.")
                        continue
                    value = parsed_date.strftime("%d-%m-%Y")
                elif field not in ("Name", "DL No"):
                    value = re.sub(r"\s+", "", value.upper())
                confident = all(item.get("confidence") is not None and item["confidence"] >= min_conf
                                for item in (label, line))
                candidates.append((value, confident))
                actual_scores = [item.get("confidence") for item in (label, line)]
                scores.append((value, min(actual_scores) if None not in actual_scores else None))
        choose_field(output, field, candidates)
        if field in ("Name", "DL No") and output.get(field):
            supporting = [score for value, score in scores if value == output[field] and score is not None]
            output.setdefault("Field Confidence", {})[field] = max(supporting) if supporting else None

    if "DL No" not in output:
        numbers = []
        number_scores = []
        for label in lines:
            match = specs["DL No"][0].search(label["text"])
            if not match:
                if DL_NUMBER_PATTERN.fullmatch(label["text"]):
                    score = label.get("confidence")
                    numbers.append((label["text"], score is not None and score >= .45))
                    number_scores.append(score)
                continue
            fragments = [label["text"][match.end():].strip(" :.-")]
            used = [label]
            for line in lines:
                if line is label or not nearby_value(label, line):
                    continue
                if re.fullmatch(r"[A-Z0-9 /-]{1,24}", line["text"], re.I) and not FIELD_LABEL.search(line["text"]):
                    fragments.append(line["text"])
                    used.append(line)
            value = " ".join(part for part in fragments if part)
            if len(re.sub(r"[ -]", "", value)) >= 8:
                scores = [line.get("confidence") for line in used]
                score = min(scores) if None not in scores else None
                numbers.append((value, score is not None and score >= .45))
                number_scores.append(score)
        choose_field(output, "DL No", numbers)
        if output.get("DL No"):
            output.setdefault("Field Confidence", {})["DL No"] = max((score for score in number_scores if score is not None), default=None)

    addresses = []
    for label in lines:
        match = re.search(r"\baddress\b\s*[:.-]?", label["text"], re.I)
        if not match:
            continue
        tail = label["text"][match.end():].strip()
        boundary = ADDRESS_BOUNDARY.search(tail)
        if boundary:
            tail = tail[:boundary.start()].strip()
        parts = [tail] if tail else []
        used = [label]
        previous_y = label["y"]
        address_left = label["left"]
        previous_right = label["right"]
        for line in lines:
            if boundary:
                break
            if line is label or line["y"] < label["y"] - .015:
                continue
            if line["y"] == label["y"] and line["left"] < label["right"]:
                continue
            same_row = abs(line["y"] - previous_y) <= .015
            continues_row = same_row and -.02 <= line["left"] - previous_right <= .22
            aligned = min(abs(line["left"] - label["left"]), abs(line["left"] - address_left)) <= .14
            if not (continues_row or aligned):
                continue
            if line["bottom"] - line["top"] > .12:
                continue
            if line["y"] - previous_y > .075 or ADDRESS_BOUNDARY.search(line["text"]):
                break
            if not parts:
                address_left = line["left"]
            parts.append(line["text"])
            used.append(line)
            previous_y = line["y"]
            previous_right = line["right"]
        if parts:
            addresses.append((" ".join(parts), all(line.get("confidence") is not None
                                                  and line["confidence"] >= .40 for line in used)))
    choose_field(output, "Address", addresses)

    states = []
    for line in lines:
        if not re.search(r"\b(?:government|govt|state|union\s+territory)\b", line["text"], re.I):
            continue
        context = line["text"]
        used = [line]
        if not extract_state(context):
            for following in lines:
                if following is not line and nearby_value(line, following) and extract_state(following["text"]):
                    context += " " + following["text"]
                    used.append(following)
        state = extract_state(context)
        if state:
            states.append((state, all(item.get("confidence") is not None and item["confidence"] >= .40 for item in used)))
    choose_field(output, "State", states)
    return output


def read_vertical_issue_fields(image):
    height, width = image.shape[:2]
    edge_width = max(1, round(width * .18))
    bands = [cv2.rotate(strip, direction)
             for strip in (image[:, :edge_width], image[:, -edge_width:])
             for direction in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    gap = 24
    canvas = np.full((4 * (edge_width + gap), height, 3), 255, dtype=np.uint8)
    for index, band in enumerate(bands):
        top = index * (edge_width + gap)
        canvas[top:top + edge_width] = band
    lines = read_ocr_lines(canvas)
    candidates = {"Date of Issue": [], "Date of First Issue": []}
    reviews = []
    for index in range(4):
        top, bottom = index * (edge_width + gap), index * (edge_width + gap) + edge_width
        band_lines = [line for line in lines if top <= line["y"] * canvas.shape[0] < bottom]
        fields = extract_labelled_fields(band_lines)
        reviews.extend(note for note in fields.get("Field Review", []) if note.startswith("Date"))
        for field in candidates:
            if fields.get(field):
                candidates[field].append((fields[field], True))
    output = {}
    for field, values in candidates.items():
        choose_field(output, field, values)
    if reviews:
        output.setdefault("Field Review", []).extend(reviews)
    return output


def extract_license_ocr_fields(image):
    recognized_lines = read_ocr_lines(image)
    output_fields = extract_labelled_fields(recognized_lines)
    output_fields["Full Image OCR"] = " | ".join(line["text"] for line in recognized_lines)
    output_fields["OCR Detections"] = recognized_lines
    output_fields["Validity Categories"] = sorted({
        category for category, pattern in (("NT", r"(?:\b|\()(?:NT|NON[ -]*TRANSPORT)(?:\b|\))"), ("TR", r"(?:\b|\()(?:TR|T|TRANSPORT)(?:\b|\))"))
        for line in recognized_lines
        if re.search(pattern, line["text"], re.I)
        and (re.search(r"\b(?:valid|validity|expiry|expires)\b", line["text"], re.I)
             or re.fullmatch(r"[\s():-]*(?:NT|T|TR|TRANSPORT|NON[ -]*TRANSPORT)[\s():-]*", line["text"], re.I))
        and not (category == "TR" and re.search(r"\bNON[ -]*TRANSPORT\b", line["text"], re.I))
    })
    if not output_fields.get("Date of Issue") or not output_fields.get("Date of First Issue"):
        try:
            vertical = read_vertical_issue_fields(image)
        except Exception as exc:
            logger.warning("Vertical issue-date OCR failed: %s", exc)
            vertical = {}
        output_fields.setdefault("Field Review", []).extend(vertical.pop("Field Review", []))
        for field, value in vertical.items():
            if field not in output_fields:
                output_fields[field] = value
            elif not value or (output_fields[field] and output_fields[field] != value):
                choose_field(output_fields, field, [(output_fields[field], True), (value, True)])

    date_candidates = _date_candidates_for_lines(recognized_lines)
    if not date_candidates:
        return output_fields

    output_fields["Detected Dates"] = [candidate["date"] for candidate in date_candidates]
    excluded_dates = {
        output_fields.get("DOB", ""),
        output_fields.get("Date of Issue", ""),
        output_fields.get("Date of First Issue", ""),
    } - {""}

    for category in ("NT", "TR"):
        expiry, review = find_nt_expiry(recognized_lines, date_candidates, category=category, excluded_dates=excluded_dates)
        if expiry or review:
            output_fields["Validity Categories"] = sorted(set(output_fields["Validity Categories"]) | {category})
        if expiry:
            output_fields[f"Valid Till ({category})"] = expiry
        if review:
            if category == "NT":
                output_fields["NT Expiry Review"] = review
            else:
                output_fields["TR Expiry Review"] = review
                output_fields.setdefault("Field Review", []).append(review)
    validity_dates = [output_fields[key] for key in ("Valid Till (NT)", "Valid Till (TR)") if output_fields.get(key)]
    if validity_dates:
        output_fields["Valid Till Dates"] = list(dict.fromkeys(validity_dates))
    return output_fields


def normalize_field_value(value):
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip(" |")
    if cleaned.lower() in {"no text detected", ""}:
        return ""
    return cleaned


def assess_nt_expiry(fields, today=None):
    today = today if today is not None else date.today()
    value = normalize_field_value(fields.get("Valid Till (NT)", ""))
    review = fields.get("NT Expiry Review")
    if review:
        return {"expiry_status": "review", "expiry_message": str(review), "expiry_date": value}
    expiry = None
    for format_string in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            expiry = datetime.strptime(value, format_string).date()
            break
        except ValueError:
            continue
    if expiry is None:
        return {"expiry_status": "review", "expiry_date": value,
                "expiry_message": "Valid Till (NT) is missing or unreadable. Upload a clearer photo showing the NT expiry date."}
    expired = expiry < today
    return {"expiry_status": "expired" if expired else "valid", "expiry_date": expiry.isoformat(),
            "expiry_message": (f"NT expiry date: {expiry:%d-%m-%Y}. "
                               + ("The non-transport validity has expired." if expired
                                  else "The non-transport validity has not expired.")
                               + f" Checked on {today:%d-%m-%Y}; based on NT expiry only.")}


def assess_applicable_expiry(fields, today=None):
    """Delegate category-specific validity evaluation to validation_rules.py."""
    return validate_selected_category_expiry(fields, today=today)


def assess_licence(fields, today=None):
    """Delegate full validation and user-friendly messaging to validation_rules.py."""
    return evaluate_licence_validation(fields, today=today, DL_NUMBER_PATTERN=DL_NUMBER_PATTERN)


def _process_licence_image(image_input):
    image = read_image(image_input)
    model = load_license_model()
    class_names = CONFIG["models"]["Driving_License"]["classes"]
    results = model(image)
    output_image = image.copy()
    h, w, _ = output_image.shape
    filtered_boxes = {}

    for result in results:
        if not result.boxes:
            continue
        for box in result.boxes:
            cls = int(box.cls[0].item())
            if cls >= len(class_names):
                continue
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()
            class_name = class_names[cls]
            if cls not in filtered_boxes or conf > filtered_boxes[cls]["conf"]:
                filtered_boxes[cls] = {"conf": conf, "xyxy": xyxy, "class_name": class_name}

    detected_text = {}
    for data in filtered_boxes.values():
        x_min, y_min, x_max, y_max = map(int, data["xyxy"])
        class_name = data["class_name"]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        region_img = image[y_min:y_max, x_min:x_max]
        if region_img.size == 0:
            continue

        # Read clean upscaled crop first to preserve word spacing; fallback to heavy filter only if empty
        clean_crop = cv2.copyMakeBorder(
            upscale_image(region_img, scale=2), 12, 12, 12, 12,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        crop_lines = read_ocr_lines(clean_crop)
        if not crop_lines:
            filtered_crop = cv2.copyMakeBorder(
                preprocess_image(region_img), 12, 12, 12, 12,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
            crop_lines = read_ocr_lines(filtered_crop)

        raw_text = " ".join(line["text"] for line in crop_lines).strip()

        if class_name == "Blood Group":
            bg_match = VALID_BLOOD_GROUP_PATTERN.search(raw_text)
            if not bg_match:
                continue
            raw_text = re.sub(r"\s+", "", bg_match.group(1).upper())

        elif class_name in ("DOB", "Date of Issue", "Date of First Issue", "Valid Till", "Valid Till (NT)", "Valid Till (TR)"):
            date_m = DATE_PATTERN.search(raw_text)
            if not date_m:
                continue
            parsed_d = parse_licence_date(date_m.group(1))
            if not parsed_d:
                continue
            raw_text = parsed_d.strftime("%d-%m-%Y")

        elif class_name == "Relation With":
            raw_text = re.sub(
                r"^(?:Son\s*/\s*Daughter\s*/\s*Wife\s+of|S\s*/\s*D\s*/\s*W\s+of|S\s*/\s*W\s*/\s*D\s*(?:of)?|S/O|D/O|W/O|Father[’']?s?\s+Name)\s*[:.-]?\s*",
                "",
                raw_text,
                flags=re.I,
            ).strip()

        elif class_name == "Address":
            raw_text = re.sub(r"^Address\s*[:.-]?\s*", "", raw_text, flags=re.I).strip()

        elif class_name in ("DL No", "Name"):
            raw_text = re.sub(
                r"^(?:DL\s*(?:No\.?|Number)|licen[cs]e\s*(?:No\.?|Number)|Name(?:\s+of\s+(?:the\s+)?holder)?)\s*[:.-]?\s*",
                "",
                raw_text,
                flags=re.I,
            ).strip()
            if class_name == "DL No":
                dl_m = DL_NUMBER_PATTERN.search(raw_text)
                if dl_m:
                    raw_text = dl_m.group(1)
            scores = [line.get("confidence") for line in crop_lines]
            detected_text.setdefault("Field Confidence", {})[class_name] = min(scores) if scores and None not in scores else None

        if raw_text:
            detected_text[class_name] = raw_text
            detected_text.setdefault("Crop OCR", {})[class_name] = raw_text

        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (46, 184, 107), 2)
        cv2.putText(
            output_image,
            class_name,
            (x_min, max(18, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (36, 99, 235),
            2,
        )

    try:
        full_image_fields = extract_license_ocr_fields(image)
        crop_confidence = detected_text.get("Field Confidence", {})
        for k, v in full_image_fields.pop("Field Confidence", {}).items():
            if v is not None:
                crop_confidence[k] = max(v, crop_confidence.get(k) or 0.0)
        # Never overwrite a valid YOLO crop extraction with an empty string ""
        for k, v in full_image_fields.items():
            if v or k not in detected_text or not detected_text.get(k):
                detected_text[k] = v
        detected_text["Field Confidence"] = crop_confidence
    except Exception:
        logger.exception("Full-image licence OCR failed")

    state_extracted = extract_state(detected_text.get("State", "")) or extract_state(detected_text.get("Full Image OCR", ""))
    detected_text["State"] = state_extracted
    annotated_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return detected_text, annotated_rgb


def weak_field_read(fields, field):
    score = fields.get("Field Confidence", {}).get(field, 1)
    notes = [str(note) for note in fields.get("Field Review", []) if str(note).startswith(field + ":")]
    if field.startswith("Valid Till ("):
        notes.append(str(fields.get("NT Expiry Review" if "(NT)" in field else "TR Expiry Review", "")))
    return score is None or score < .45 or any("low" in note.lower() or "unavailable" in note.lower() for note in notes)


def merge_licence_fields(current, incoming, is_retry=False):
    core = {"DL No", "Name", "DOB", "Date of Issue", "Date of First Issue",
            "Valid Till (NT)", "Valid Till (TR)", "Valid Till (T)", "Valid Till"}
    protected = {
        field for field in core
        if current.get(field) and not weak_field_read(current, field)
        and (is_retry or weak_field_read(incoming, field))
    }
    protected_notes = {"NT Expiry Review" for field in protected if field == "Valid Till (NT)"}
    protected_notes.update("TR Expiry Review" for field in protected if field in ("Valid Till (TR)", "Valid Till (T)"))
    attempts = current.setdefault("OCR Attempts", [])
    if incoming.get("Full Image OCR"):
        attempts.append(incoming["Full Image OCR"])
    for key, value in incoming.items():
        if key in protected or key in protected_notes:
            current.setdefault("OCR Diagnostics", []).append(f"Ignored secondary read of {key}: {value}")
            continue
        if key == "Field Confidence":
            continue
        if key == "Field Review":
            for note in value:
                target = "OCR Diagnostics" if (is_retry or any(str(note).startswith(field + ":") for field in protected)) else key
                if note not in current.get(target, []):
                    current.setdefault(target, []).append(note)
            continue
        if key == "OCR Detections":
            current.setdefault(key, []).extend(value)
            continue
        if key in core and current.get(key) and value:
            old = normalize_field_value(current[key]).upper()
            new = normalize_field_value(value).upper()
            if key == "DL No":
                old, new = re.sub(r"[ -]", "", old), re.sub(r"[ -]", "", new)
            elif key.startswith("Date") or key.startswith("Valid Till") or key == "DOB":
                old, new = parse_licence_date(old), parse_licence_date(new)
            if old != new and not weak_field_read(current, key):
                if is_retry:
                    continue
                current[key] = ""
                current.setdefault("Field Review", []).append(f"{key}: conflicting OCR reads. Upload a clearer image.")
                if key.startswith("Valid Till ("):
                    current["NT Expiry Review" if "(NT)" in key else "TR Expiry Review"] = "Conflicting expiry dates. Upload a clearer image."
                continue
        if key in core and any(str(note).startswith(key + ": conflicting") for note in current.get("Field Review", [])):
            continue
        if value or key not in current:
            current[key] = value
            if key in core and value:
                notes = current.get("Field Review", [])
                current["Field Review"] = [note for note in notes if not str(note).startswith(key + ": low")]
                if key.startswith("Valid Till ("):
                    note_key = "NT Expiry Review" if "(NT)" in key else "TR Expiry Review"
                    if note_key not in incoming and "low" in str(current.get(note_key, "")).lower():
                        current.pop(note_key, None)
                if key in incoming.get("Field Confidence", {}):
                    current.setdefault("Field Confidence", {})[key] = incoming["Field Confidence"][key]
    return current


def process_driving_license(image_input, save_json=False, output_json="detected_license.json"):
    """Accept single image or front/back/category dict with guaranteed UI preview output."""
    payload = image_input if isinstance(image_input, dict) else {"front": image_input}
    fields = {"Screening Category": payload.get("category", "Automatic")}
    annotated = None
    available_images = []

    if payload.get("front") is None and payload.get("back") is None:
        fields["Image Quality Report"] = check_image_quality(None)

    for side in ("front", "back"):
        source = payload.get(side)
        if source is None:
            continue
        try:
            image = read_image(source)
            if annotated is None:
                annotated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if side == "front" or "Image Quality Report" not in fields:
                fields["Image Quality Report"] = check_image_quality(image)
                if fields["Image Quality Report"].get("hard_fail"):
                    break
            side_fields, side_annotated = _process_licence_image(source)
            available_images.append(image)
            merge_licence_fields(fields, side_fields, is_retry=False)
            if side == "front" or annotated is None:
                annotated = side_annotated
        except Exception:
            logger.exception("Licence image processing failed for %s", side)
            fields["Processing Error"] = True

    assessment = assess_licence(fields)
    retry_reasons = {
        "NUMBER_UNREADABLE",
        "NUMBER_UNSUPPORTED",
        "NUMBER_INCOMPLETE",
        "NUMBER_SUSPICIOUS",
        "NUMBER_LOW_CONFIDENCE",
        "NAME_UNREADABLE",
        "FIELD_UNCERTAIN",
        "FIELD_CONFLICT",
        "DOB_MISSING",
        "INVALID_DATE",
        "MISSING_NT_DATE",
        "MISSING_TR_DATE",
    }
    if (
        available_images
        and not fields.get("Image Quality Report", {}).get("hard_fail")
        and assessment["reason_codes"]
        and assessment["reason_codes"][0] in retry_reasons
    ):
        for image in available_images:
            try:
                merge_licence_fields(fields, extract_license_ocr_fields(preprocess_image(image)), is_retry=True)
            except Exception:
                logger.exception("Licence OCR retry failed")
        assessment = assess_licence(fields)

    if (
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
    fields["Validation Message"] = assessment["message"]
    fields["Official Portal"] = PARIVAHAN_PORTAL_URL
    if save_json:
        with open(output_json, "w", encoding="utf-8") as output:
            json.dump(fields, output, indent=4)
    return fields, annotated, assessment


def driving_license(image_path, save_json=True, output_json="detected_license.json"):
    fields, _, _ = process_driving_license(image_path, save_json=save_json, output_json=output_json)
    return fields


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract fields from an Indian driving licence image.")
    parser.add_argument("image_path", help="Path to the driving licence image")
    parser.add_argument("--output-json", default="detected_license.json")
    parser.add_argument("--no-save-json", action="store_false", dest="save_json")
    parser.add_argument("--category", choices=("Automatic", "NT", "TR"), default="Automatic")
    parser.add_argument("--reverse-image", help="Optional image of the reverse side")
    args = parser.parse_args()

    image_input = {"front": args.image_path, "back": args.reverse_image, "category": args.category}
    result = driving_license(image_input, save_json=args.save_json, output_json=args.output_json)
    print(json.dumps(result, indent=4))