import numpy as np
import pytest

import inference


def test_horizontal_date_of_issue_skips_vertical_ocr(monkeypatch):
    """Happy Path: If horizontal OCR already found Date of Issue, vertical strip OCR must not run or wipe it."""
    vertical_called = {"count": 0}

    def fake_read_ocr_lines(_image):
        return [
            {"text": "DL No: TN01 20150012345", "confidence": 0.96, "left": 0.1, "right": 0.6, "top": 0.1, "bottom": 0.15, "x": 0.35, "y": 0.12},
            {"text": "Date of Issue: 03-11-2015", "confidence": 0.95, "left": 0.1, "right": 0.5, "top": 0.3, "bottom": 0.35, "x": 0.30, "y": 0.32},
            {"text": "Valid Till (NT): 02-11-2035", "confidence": 0.95, "left": 0.1, "right": 0.5, "top": 0.4, "bottom": 0.45, "x": 0.30, "y": 0.42},
        ]

    def fake_vertical(_image):
        vertical_called["count"] += 1
        return {"Date of Issue": "01-01-2099"}

    monkeypatch.setattr(inference, "read_ocr_lines", fake_read_ocr_lines)
    monkeypatch.setattr(inference, "read_vertical_issue_fields", fake_vertical)

    dummy_img = np.full((400, 600, 3), 255, dtype=np.uint8)
    fields = inference.extract_license_ocr_fields(dummy_img)

    assert fields.get("Date of Issue") == "03-11-2015"
    assert vertical_called["count"] == 0


def test_merge_licence_fields_unions_validity_categories():
    """Boundary Case: Front and Back sides with different validity categories must merge instead of overwriting."""
    front = {"DL No": "TN0120150012345", "Validity Categories": ["NT"], "Field Confidence": {"DL No": 0.95}}
    back = {"Validity Categories": ["TR"], "Valid Till (TR)": "10-10-2030"}

    merged = inference.merge_licence_fields(front, back, is_retry=False)
    assert merged["Validity Categories"] == ["NT", "TR"]
    assert merged["Valid Till (TR)"] == "10-10-2030"


def test_read_image_rejects_empty_input():
    """Negative Path: Empty or None image arrays must raise ValueError cleanly."""
    with pytest.raises(ValueError, match="valid driving licence image"):
        inference.read_image(None)
    with pytest.raises(ValueError, match="valid driving licence image"):
        inference.read_image(np.array([], dtype=np.uint8))


def test_preprocess_image_fast_bilateral_execution():
    """Performance Boundary: preprocess_image on a large 1800x1200 array must complete in < 1 second."""
    import time

    large_img = np.random.randint(0, 255, (1200, 1800, 3), dtype=np.uint8)
    start = time.perf_counter()
    out = inference.preprocess_image(large_img)
    elapsed = time.perf_counter() - start

    assert out is not None and out.ndim == 3
    assert max(out.shape[:2]) <= 1200
    assert elapsed < 1.5