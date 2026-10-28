"""PaddleOCR-VL subprocess worker used by the existing licence validator."""

import argparse
import json
import os
import re
from pathlib import Path
import cv2

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL


def _prepare_fast_vl_image(image_path, max_side=1280):
    """
    Downscale oversized camera images to max_side (1280px) in place before VL inference
    and return (image_path_str, actual_width, actual_height).
    """
    actual_width, actual_height = 1000, 640
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return str(image_path), actual_width, actual_height
        h, w = img.shape[:2]
        actual_width, actual_height = int(w), int(h)
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            actual_width, actual_height = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (actual_width, actual_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(image_path), resized)
    except Exception:
        pass
    return str(image_path), max(1, actual_width), max(1, actual_height)


def _clean_line(value):
    text = str(value or "").replace("\r", "\n").strip()
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    text = text.strip().strip("|")
    if "|" in text:
        text = " ".join(part.strip() for part in text.split("|") if part.strip())
    text = text.replace("**", "").replace("__", "").strip()
    if not text or re.fullmatch(r"[\s\-+|:=.]+", text):
        return ""
    return re.sub(r"\s+", " ", text)


def _unwrap_result(result):
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload


def _bbox_limits(raw_bbox, width, height):
    """Handle PaddleX rectangle, quadrilateral, and polygon bbox formats."""
    try:
        if hasattr(raw_bbox, "tolist"):
            raw_bbox = raw_bbox.tolist()
        if not raw_bbox:
            return 0.0, 0.0, float(width), float(height)
        if isinstance(raw_bbox[0], (list, tuple)):
            values = [float(value) for row in raw_bbox for value in row]
        else:
            values = [float(value) for value in raw_bbox]
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0, float(width), float(height)

    if len(values) == 4:
        x1, y1, x2, y2 = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return 0.0, 0.0, float(width), float(height)

    x1 = max(0.0, min(float(width), x1))
    x2 = max(x1, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(y1, min(float(height), y2))
    return x1, y1, x2, y2


def _result_lines(result, image_width=1000, image_height=640):
    payload = _unwrap_result(result)
    width = max(1, int(payload.get("width") or image_width or 1000))
    height = max(1, int(payload.get("height") or image_height or 640))
    blocks = payload.get("parsing_res_list") or []
    lines = []

    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        content = str(block.get("block_content") or "")
        raw_lines = [_clean_line(line) for line in content.splitlines()]
        raw_lines = [line for line in raw_lines if line]
        if not raw_lines:
            continue

        x1, y1, x2, y2 = _bbox_limits(block.get("block_bbox", []), width, height)
        line_height = max(1.0, (y2 - y1) / len(raw_lines))
        for line_index, text in enumerate(raw_lines):
            top = y1 + line_index * line_height
            bottom = min(y2, top + line_height)
            lines.append({
                "text": text,
                "confidence": 0.95,
                "engine": "paddleocr_vl",
                "block_id": block.get("block_id", block_index),
                "left": x1 / width,
                "right": x2 / width,
                "top": top / height,
                "bottom": bottom / height,
                "x": ((x1 + x2) / 2.0) / width,
                "y": ((top + bottom) / 2.0) / height,
            })

    lines.sort(key=lambda line: (line["y"], line["x"]))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()

    prepared_records = [_prepare_fast_vl_image(path) for path in args.images]
    prepared_images = [record[0] for record in prepared_records]
    optimal_threads = max(4, (os.cpu_count() or 6))

    pipeline = PaddleOCRVL(
        device="cpu",
        cpu_threads=optimal_threads,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    results = list(pipeline.predict(prepared_images))
    output = []
    for index, (path, img_w, img_h) in enumerate(prepared_records):
        result = results[index] if index < len(results) else None
        output.append({
            "image": str(Path(path).name),
            "lines": _result_lines(result, image_width=img_w, image_height=img_h) if result is not None else [],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": output}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()