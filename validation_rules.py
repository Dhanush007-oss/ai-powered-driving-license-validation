"""
Production Validation Engine for Indian Driving Licence Screening.

Deterministic business rules:
  1. Pre-flight Quality Screening (Resolution, Blur, Lighting, Glare, File Integrity)
  2. Document Verification (Filter out Aadhaar, PAN, Voter ID, Passport)
  3. Universal Licence Number Validation (Prefix Reconstruction & State Codes)
  4. Mandatory Field Integrity (Name, Date of Birth, Issue Date Chronology)
  5. Mathematical Category Expiry Binding (NT / TR Isolation via Date Elimination)
"""

from datetime import date, datetime
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("dl_validator.rules")

PARIVAHAN_PORTAL_URL: str = "https://sarathi.parivahan.gov.in/"
PARIVAHAN_MAIN_URL: str = "https://parivahan.gov.in/"

AUTHENTICITY_SCOPE_NOTE: str = (
    "Scope & Official Notice: This automated screening inspects physical card image quality, "
    "printed field legibility, and validity chronology. Because live government cryptographic API "
    "keys are restricted, physical image processing cannot verify authenticity, duplicates, or "
    "active suspensions. For official verification, visit the Parivahan Sarathi Portal: "
    f"{PARIVAHAN_PORTAL_URL}"
)

# Standardized ISO / Indian State & Union Territory 2-Letter Codes
SUPPORTED_STATE_CODES: frozenset = frozenset(
    "AN AP AR AS BR CH CG DL GA GJ HR HP JK JH KA KL LA LD MP MH MN ML MZ NL OD OR PY PB RJ SK TN TG TS TR DD UK UA UP WB".split()
)

STATE_NAME_TO_CODE: Dict[str, str] = {
    "andhra pradesh": "AP",
    "arunachal pradesh": "AR",
    "assam": "AS",
    "bihar": "BR",
    "chhattisgarh": "CG",
    "delhi": "DL",
    "goa": "GA",
    "gujarat": "GJ",
    "haryana": "HR",
    "himachal pradesh": "HP",
    "jammu and kashmir": "JK",
    "jharkhand": "JH",
    "karnataka": "KA",
    "kerala": "KL",
    "madhya pradesh": "MP",
    "maharashtra": "MH",
    "manipur": "MN",
    "meghalaya": "ML",
    "mizoram": "MZ",
    "nagaland": "NL",
    "odisha": "OD",
    "punjab": "PB",
    "rajasthan": "RJ",
    "sikkim": "SK",
    "tamil nadu": "TN",
    "telangana": "TS",
    "tripura": "TR",
    "uttar pradesh": "UP",
    "uttarakhand": "UK",
    "west bengal": "WB",
    "andaman and nicobar islands": "AN",
    "chandigarh": "CH",
    "dadra and nagar haveli and daman and diu": "DD",
    "ladakh": "LA",
    "lakshadweep": "LD",
    "puducherry": "PY",
}

STATE_CODE_TO_NAME: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PY": "Puducherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TS": "Telangana",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Regex Blueprints: Handles standardized Sarathi 15-to-16 char schemes and legacy RTO formats
CONFIGURED_DL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^([A-Z]{2})([0-9]{2})([0-9]{4})([0-9]{7})$"),
    re.compile(r"^([A-Z]{2})([0-9]{3})([0-9]{4})([0-9]{6,7})$"),
    re.compile(r"^([A-Z]{2})([0-9A-Z]{2,3})([0-9]{4})([0-9]{5,7})$"),
)

DATE_FORMATS: Tuple[str, ...] = (
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d %b %Y",
    "%d %B %Y",
)

DATE_TOKEN_REGEX: str = (
    r"(\d{4}\s*-\s*\d{2}\s*-\s*\d{2}|\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)

OTHER_ID_KEYWORDS: re.Pattern = re.compile(
    r"\b(?:UNIQUE\s+IDENTIFICATION\s+AUTHORITY|AADHAAR|MERA\s+AADHAAR|INCOME\s+TAX\s+DEPARTMENT|"
    r"PERMANENT\s+ACCOUNT\s+NUMBER|ELECTION\s+COMMISSION\s+OF\s+INDIA|ELECTOR\s+PHOTO\s+IDENTITY|PASSPORT\s+OF\s+INDIA)\b",
    re.I,
)

DL_DOCUMENT_KEYWORDS: re.Pattern = re.compile(
    r"\b(?:DRIVING\s+LICEN[CS]E|MOTOR\s+DRIVING|LICENSING\s+AUTHORITY|AUTHORISATION\s+TO\s+DRIVE|"
    r"MCWG|MCWOG|LMV|TRANS|NON[ -]*TRANSPORT|VALID\s+TILL|VALIDITY|DL\s*NO|LICEN[CS]E\s*NO|FORM\s*7)\b",
    re.I,
)

VEHICLE_CLASS_REGEX: re.Pattern = re.compile(
    r"\b(MCWG|MCWOG|MCW/O|LMV-NT|LMV-TR|LMV|FVG|HGMV|HPMV|HTV|HMV|TRANS|3W-NT|3W-TR|2W|E-RICKSHAW|ADAPTED\s+VEHICLE|LDRXCV)\b",
    re.I,
)


def build_error_message(error_title: str, what_was_the_error: str, how_to_fix: str, include_rto_link: bool = False) -> str:
    """Format structured diagnostic errors for UI consumption."""
    msg = f"NOT VALID | Error ({error_title}): {what_was_the_error} | How to fix: {how_to_fix}"
    if include_rto_link:
        msg += f" | Official RTO / Parivahan Portal: {PARIVAHAN_PORTAL_URL}"
    return msg


def normalize_text(value: Any) -> str:
    """Strip OCR artifacts, repeated spaces, and null tokens."""
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip(" |")
    if cleaned.lower() in {"no text detected", "none", "null", "—", "-"}:
        return ""
    return cleaned


def parse_date_safely(value: Any) -> Optional[date]:
    """Parse dates robustly across Indian regional and standard delimiters."""
    cleaned = re.sub(r"\s*([./-])\s*", r"\1", normalize_text(value))
    if not cleaned:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def recover_missing_core_fields(fields: Dict[str, Any], today: Optional[date] = None) -> None:
    """Contextual regex harvesting from unsegmented OCR raw text."""
    today = today if today is not None else date.today()
    full_ocr = str(fields.get("Full Image OCR", ""))

    # Sanitize already mapped date keys
    for date_field in ("DOB", "Date of Issue", "Date of First Issue", "Valid Till", "Valid Till (NT)", "Valid Till (TR)"):
        raw_val = normalize_text(fields.get(date_field))
        if raw_val and parse_date_safely(raw_val) is None:
            m = re.search(DATE_TOKEN_REGEX, raw_val)
            if m and parse_date_safely(m.group(1)):
                fields[date_field] = parse_date_safely(m.group(1)).strftime("%d-%m-%Y")
            else:
                fields[date_field] = ""

    if full_ocr:
        # Recover DOB
        if not normalize_text(fields.get("DOB")):
            dob_match = re.search(
                r"\b(?:DOB|Date\s+of\s+Birth|Birth\s+Date)\b\s*[:.-]?\s*" + DATE_TOKEN_REGEX,
                full_ocr,
                re.I,
            )
            if dob_match:
                parsed = parse_date_safely(dob_match.group(1))
                if parsed:
                    fields["DOB"] = parsed.strftime("%d-%m-%Y")

        # Recover Date of Issue
        if not normalize_text(fields.get("Date of Issue")):
            doi_match = re.search(
                r"\b(?:Date\s+of\s+Issue|Issue\s+Date|Date\s+Issued|DOI)\b[^0-9]{0,25}" + DATE_TOKEN_REGEX,
                full_ocr,
                re.I,
            )
            if doi_match:
                parsed = parse_date_safely(doi_match.group(1))
                if parsed and parsed <= today:
                    fields["Date of Issue"] = parsed.strftime("%d-%m-%Y")

        # Recover tabular Issue and Validity pairs
        tabular_match = re.search(
            r"\b(?:Issue\s+Date|Date\s+of\s+Issue|DOI)\b.*?\b(?:Valid(?:ity)?\s*\(?\s*NT\s*\)?|NT)\b[^0-9]{0,45}"
            + DATE_TOKEN_REGEX
            + r"[\s|,;:/-]{1,15}"
            + DATE_TOKEN_REGEX,
            full_ocr,
            re.I,
        )
        if tabular_match:
            d1 = parse_date_safely(tabular_match.group(1))
            d2 = parse_date_safely(tabular_match.group(2))
            if d1 and d2 and d1 <= today and d2 > d1:
                if not normalize_text(fields.get("Date of Issue")):
                    fields["Date of Issue"] = d1.strftime("%d-%m-%Y")
                if not normalize_text(fields.get("Valid Till (NT)")):
                    fields["Valid Till (NT)"] = d2.strftime("%d-%m-%Y")
                    fields.pop("NT Expiry Review", None)

        # Recover general validity
        if not normalize_text(fields.get("Valid Till")):
            vt_match = re.search(
                r"\b(?:Valid\s*(?:Till|Upto|Up\s+to|Until|Through)|Validity|Expiry(?:\s+Date)?)\b[^0-9]{0,35}"
                + DATE_TOKEN_REGEX,
                full_ocr,
                re.I,
            )
            if vt_match:
                parsed = parse_date_safely(vt_match.group(1))
                if parsed:
                    fields["Valid Till"] = parsed.strftime("%d-%m-%Y")

        # Recover Blood Group
        if not normalize_text(fields.get("Blood Group")):
            bg_search = re.search(
                r"\b(?:blood\s*(?:group|grp)|b\.?\s*g\.?|bg)\b[\s|:.;,-]{0,20}((?:A1B|A2B|AB|A1|A2|B1|A|B|O)\s*[+-])",
                full_ocr,
                re.I,
            )
            if bg_search:
                fields["Blood Group"] = re.sub(r"\s+", "", bg_search.group(1).upper())

        # Recover Organ Donor
        if not normalize_text(fields.get("Organ Donor")):
            od_search = re.search(
                r"\b(?:organ|0rgan)\s*(?:donor|donar|doner)\b[\s|:.;,-]*(?:holder[’']?s?\s*signature[\s|:.;,-]*)?\b(YES|NO|Y|N)\b",
                full_ocr,
                re.I,
            )
            if od_search:
                fields["Organ Donor"] = od_search.group(1).upper()

        # Recover Relation Name
        if not normalize_text(fields.get("Relation With")):
            rel_search = re.search(
                r"(?:\b(?:Son\s*/\s*Daughter\s*/\s*Wife\s+of|S\s*/\s*D\s*/\s*W\s+of|S\s*/\s*W\s*/\s*D\s*(?:of)?|S/O|D/O|W/O|Father[’']?s?\s+Name)\b\s*[:.-]?\s*)([A-Z][A-Z .'-]{2,45}?)(?=\s*\||\b(?:Address|DOB|Blood|BG|Issue|Valid|Organ|DL)\b|$)",
                full_ocr,
                re.I,
            )
            if rel_search:
                fields["Relation With"] = normalize_text(rel_search.group(1))

        # Recover Vehicle Classes
        if not normalize_text(fields.get("Vehicle Type")):
            v_matches: List[str] = []
            for m in VEHICLE_CLASS_REGEX.finditer(full_ocr):
                token = m.group(1).upper()
                if token not in v_matches:
                    v_matches.append(token)
            if v_matches:
                fields["Vehicle Type"] = ", ".join(v_matches)

    # State Code Inference from DL Number
    dl_val = re.sub(r"[\s/.-]+", "", normalize_text(fields.get("DL No"))).upper()
    if len(dl_val) >= 4 and dl_val[:2] in STATE_CODE_TO_NAME:
        if not normalize_text(fields.get("State")):
            fields["State"] = STATE_CODE_TO_NAME[dl_val[:2]]
        if not normalize_text(fields.get("RTO")) and re.match(r"^[A-Z]{2}[0-9A-Z]{2}", dl_val[:4]):
            fields["RTO"] = f"{dl_val[:2]}-{dl_val[2:4]}"

    # Clean stale field review warnings if successfully recovered
    review_notes = fields.get("Field Review")
    if isinstance(review_notes, list) and review_notes:
        cleaned_notes = []
        for note in review_notes:
            note_str = str(note)
            resolved = False
            for key in (
                "DL No",
                "Name",
                "DOB",
                "Date of Issue",
                "Date of First Issue",
                "Valid Till",
                "Valid Till (NT)",
                "Valid Till (TR)",
                "Blood Group",
                "Organ Donor",
            ):
                if note_str.startswith(f"{key}:") and normalize_text(fields.get(key)):
                    resolved = True
                    break
            if not resolved:
                cleaned_notes.append(note)
        fields["Field Review"] = cleaned_notes


def recover_category_date_from_text(
    full_ocr: str,
    category: str,
    excluded_dates: Optional[Set[str]] = None,
    today: Optional[date] = None,
) -> str:
    """Extract explicit category validity tokens (NT or TR) from OCR text."""
    if not full_ocr:
        return ""
    today = today if today is not None else date.today()
    excluded = set(excluded_dates or ())
    cat_token = (
        r"(?:\(?NT\)?|N\.T\.?|NON[ -]*TRANSPORT)"
        if category == "NT"
        else r"(?:\(?TR\)?|T\.R\.?|TRANSPORT|\([Tt]\))"
    )
    validity_token = r"(?:वैधता|valid(?:\s*till|\s*upto|\s*up\s+to|\s*until|\s*through|ity)?|expiry|expires|aed)"

    pattern_inline = re.compile(
        validity_token + r"[^\d]{0,25}" + cat_token + r"[^\d]{0,25}" + DATE_TOKEN_REGEX,
        re.I,
    )
    pattern_suffix = re.compile(
        validity_token + r"?[^\d]{0,25}" + DATE_TOKEN_REGEX + r"[\s|():;,-]{0,12}(?:" + validity_token + r"[\s|():;,-]{0,8})?" + cat_token,
        re.I,
    )
    pattern_prefix = re.compile(
        cat_token + r"[\s|():;,-]{0,20}" + validity_token + r"?[^\d]{0,25}" + DATE_TOKEN_REGEX,
        re.I,
    )

    for regex in (pattern_inline, pattern_suffix, pattern_prefix):
        for match in regex.finditer(full_ocr):
            window = full_ocr[max(0, match.start() - 25) : min(len(full_ocr), match.end() + 25)]
            if category == "TR" and re.search(r"\bNON[ -]*TRANSPORT\b", window, re.I):
                continue
            parsed = parse_date_safely(match.group(1))
            if parsed:
                formatted = parsed.strftime("%d-%m-%Y")
                if formatted not in excluded:
                    return formatted

    return ""


# ==============================================================================
# 1. IMAGE QUALITY GATE
# ==============================================================================

def check_image_quality(image_input: Any) -> Dict[str, Any]:
    """
    Production Pre-flight Quality Gate.
    Evaluates physical properties: focus, brightness, extreme glare, and resolution.
    Decoupled from framing aspect ratio (never rejects portrait mobile uploads).
    """
    report: Dict[str, Any] = {
        "hard_fail": False,
        "reason_code": "",
        "message": "",
        "soft_flags": {
            "low_resolution": False,
            "blurry": False,
            "too_dark": False,
            "overexposed": False,
        },
        "telemetry": {},
    }

    if image_input is None:
        report.update(
            hard_fail=True,
            reason_code="NO_IMAGE",
            message=build_error_message(
                "Image Not Uploaded",
                "No driving licence photo was detected in the payload.",
                "Upload a clear photograph of your Indian driving licence.",
            ),
        )
        return report

    image: Optional[np.ndarray] = None
    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.isfile(image_input):
            report.update(
                hard_fail=True,
                reason_code="INVALID_FILE_FORMAT",
                message=build_error_message(
                    "Unreadable Image File",
                    "The specified image file could not be found or read.",
                    "Upload a valid JPG, PNG, or supported image file.",
                ),
            )
            return report
        image = cv2.imread(str(image_input))
    elif isinstance(image_input, np.ndarray) and image_input.size > 0:
        image = image_input.copy()

    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        report.update(
            hard_fail=True,
            reason_code="INVALID_FILE_FORMAT",
            message=build_error_message(
                "Unsupported File Format",
                "The image array is corrupted, empty, or unreadable.",
                "Ensure your camera produces standard uncorrupted RGB image files.",
            ),
        )
        return report

    height, width = int(image.shape[0]), int(image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    glare_ratio = float(np.mean(gray >= 248))
    dark_text_ratio = float(np.mean(gray < 100))

    report["telemetry"] = {
        "width": width,
        "height": height,
        "blur_score": round(laplacian_var, 2),
        "brightness_score": round(mean_brightness, 2),
        "glare_percentage": round(glare_ratio * 100.0, 2),
    }

    # Resolution Gate
    if width < 100 or height < 100:
        report.update(
            hard_fail=True,
            reason_code="LOW_RESOLUTION",
            message=build_error_message(
                "Image Resolution Too Low",
                f"The image resolution ({width}x{height} px) is below minimum optical legibility limits.",
                "Capture the card closer to the camera lens (minimum 300x300 pixels).",
            ),
        )
        return report

    if width < 300 or height < 300:
        report["soft_flags"]["low_resolution"] = True

    # Focus / Blur Gate
    if laplacian_var < 10.0:
        report.update(
            hard_fail=True,
            reason_code="BLURRY_IMAGE",
            message=build_error_message(
                "Blurry Image",
                "The uploaded image is severely out of focus or suffers from heavy motion blur.",
                "Hold the phone steady and allow the camera to focus on printed card text.",
            ),
        )
        return report
    if laplacian_var < 35.0:
        report["soft_flags"]["blurry"] = True

    # Underexposure Gate
    if mean_brightness < 20.0:
        report.update(
            hard_fail=True,
            reason_code="TOO_DARK",
            message=build_error_message(
                "Image Too Dark",
                "The lighting is insufficient to distinguish text characters from the background.",
                "Take the photo in a well-lit room or disable shadows over the card surface.",
            ),
        )
        return report
    if mean_brightness < 45.0:
        report["soft_flags"]["too_dark"] = True

    # Glare Gate (Distinguishes legitimate white card scans from destructive flash glare)
    is_pristine_doc_scan = (glare_ratio > 0.30) and (dark_text_ratio >= 0.015) and (laplacian_var >= 35.0)

    if (mean_brightness > 246.0 or glare_ratio > 0.72) and not is_pristine_doc_scan:
        report.update(
            hard_fail=True,
            reason_code="OVEREXPOSED",
            message=build_error_message(
                "Severe Glare / Overexposure",
                "Direct flash reflection or overexposure has washed out critical card fields.",
                "Turn off your camera flash and take the photo under diffused indirect lighting.",
            ),
        )
        return report
    if (mean_brightness > 225.0 or glare_ratio > 0.30) and not is_pristine_doc_scan:
        report["soft_flags"]["overexposed"] = True

    return report


def is_licence_cropped(fields: Dict[str, Any]) -> bool:
    """Flag images where text lines clip against outer image bounds."""
    detections = fields.get("OCR Detections") or []
    if not detections:
        return False
    border_hits = sum(
        1
        for line in detections
        if line.get("left", 0.5) <= 0.005
        or line.get("right", 0.5) >= 0.995
        or line.get("top", 0.5) <= 0.005
        or line.get("bottom", 0.5) >= 0.995
    )
    return bool(border_hits >= max(2, len(detections) // 2))


# ==============================================================================
# 2. UNIVERSAL LICENCE NUMBER & DATE EVALUATION
# ==============================================================================

def validate_licence_number(
    raw_number: str,
    today: Optional[date] = None,
    detected_state: Optional[str] = None,
) -> Tuple[bool, str, str, str]:
    """
    Universal Licence Number Validator.
    Reconstructs cropped state prefixes dynamically via card context.
    """
    today = today if today is not None else date.today()
    cleaned_raw = normalize_text(raw_number)

    if not cleaned_raw:
        return (
            False,
            "",
            "NUMBER_UNREADABLE",
            build_error_message(
                "Licence Number Not Detected",
                "The driving licence number could not be located on the card face.",
                "Ensure the complete licence number is visible, clear of thumb covers, and uncropped.",
            ),
        )

    # Universal Dynamic Prefix Fallback: Recover truncated state code
    if detected_state and not re.match(r"^[A-Z]{2}", cleaned_raw):
        normalized_state_name = normalize_text(detected_state).lower()
        for full_name, code in STATE_NAME_TO_CODE.items():
            if full_name in normalized_state_name:
                cleaned_raw = code + re.sub(r"^[^0-9A-Z]+", "", cleaned_raw)
                break

    if re.search(r"[^A-Z0-9\s/.-]", cleaned_raw, re.I):
        return (
            False,
            cleaned_raw,
            "NUMBER_SUSPICIOUS",
            build_error_message(
                "Unclear Licence Number Characters",
                f"The extracted licence number ('{cleaned_raw}') contains invalid characters or noise.",
                "Check for surface scratches or dirt over the licence number and take a clearer photo.",
            ),
        )

    normalized = re.sub(r"[\s/.-]+", "", cleaned_raw).upper()

    if len(normalized) < 13 or len(normalized) > 17:
        return (
            False,
            normalized,
            "NUMBER_INCOMPLETE",
            build_error_message(
                "Incomplete Licence Number",
                f"Extracted number ('{cleaned_raw}') has {len(normalized)} chars (expected 15-16).",
                "Ensure the full number from state code to final sequence digit is framed.",
            ),
        )

    for pattern in CONFIGURED_DL_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            state_code, _rto, issue_year_str, _serial = match.groups()
            try:
                issue_year = int(issue_year_str)
            except (ValueError, TypeError):
                continue
            if state_code in SUPPORTED_STATE_CODES and 1950 <= issue_year <= today.year:
                return True, normalized, "", ""

    return (
        False,
        normalized,
        "NUMBER_UNSUPPORTED",
        build_error_message(
            "Licence Number Format Not Recognized",
            f"The format of extracted number ('{cleaned_raw}') is unrecognized.",
            "Verify the card is a standard Indian Driving Licence. Format: SS-RR-YYYYNNNNNNN.",
            include_rto_link=True,
        ),
    )


def validate_selected_category_expiry(fields: Dict[str, Any], today: Optional[date] = None) -> Dict[str, Any]:
    """
    Mathematical Date Elimination Resolver.
    Eliminates DOB and Issue dates to automatically deduce validity dates.
    """
    today = today if today is not None else date.today()
    recover_missing_core_fields(fields, today=today)
    full_ocr = str(fields.get("Full Image OCR", ""))

    non_expiry_dates: Set[str] = {
        normalize_text(fields.get("DOB")),
        normalize_text(fields.get("Date of Issue")),
        normalize_text(fields.get("Date of First Issue")),
    } - {""}

    # 1. Check explicit NT field
    nt_raw = normalize_text(fields.get("Valid Till (NT)"))
    if nt_raw in non_expiry_dates:
        nt_raw = ""
    if not nt_raw:
        nt_raw = recover_category_date_from_text(full_ocr, "NT", excluded_dates=non_expiry_dates, today=today)
        if nt_raw:
            fields["Valid Till (NT)"] = nt_raw
            fields.pop("NT Expiry Review", None)

    # 2. Check explicit TR field
    tr_raw = normalize_text(fields.get("Valid Till (TR)") or fields.get("Valid Till (T)"))
    if tr_raw in non_expiry_dates:
        tr_raw = ""
    if not tr_raw:
        tr_raw = recover_category_date_from_text(
            full_ocr,
            "TR",
            excluded_dates=non_expiry_dates | ({nt_raw} if nt_raw else set()),
            today=today,
        )
        if tr_raw:
            fields["Valid Till (TR)"] = tr_raw
            fields.pop("TR Expiry Review", None)

    # 3. Check generic "Valid Till" label
    generic_raw = normalize_text(fields.get("Valid Till"))
    if generic_raw in non_expiry_dates:
        generic_raw = ""
    printed_cats = fields.get("Validity Categories") or []

    if not nt_raw and generic_raw and not tr_raw:
        if "TR" not in printed_cats or "NT" in printed_cats:
            nt_raw = generic_raw
            fields["Valid Till (NT)"] = generic_raw
            fields.pop("NT Expiry Review", None)
        elif printed_cats == ["TR"]:
            tr_raw = generic_raw
            fields["Valid Till (TR)"] = generic_raw
            fields.pop("TR Expiry Review", None)

    # 4. Mathematical Pool Fallback: Unassigned Future Dates
    if not nt_raw and not tr_raw:
        candidate_dates: List[str] = list(fields.get("Detected Dates") or [])
        if not candidate_dates:
            for match in re.finditer(DATE_TOKEN_REGEX, full_ocr):
                parsed_candidate = parse_date_safely(match.group(1))
                if parsed_candidate:
                    candidate_dates.append(parsed_candidate.strftime("%d-%m-%Y"))

        future_pool: List[Tuple[date, str]] = []
        for date_str in candidate_dates:
            if date_str in non_expiry_dates:
                continue
            parsed_d = parse_date_safely(date_str)
            if parsed_d and parsed_d > today:
                future_pool.append((parsed_d, date_str))

        if len(future_pool) == 1:
            nt_raw = future_pool[0][1]
            fields["Valid Till (NT)"] = nt_raw
            fields["Valid Till"] = nt_raw
            fields.pop("NT Expiry Review", None)
        elif len(future_pool) > 1:
            # If multiple future dates exist, pick the latest expiry for non-transport
            future_pool.sort(key=lambda item: item[0])
            nt_raw = future_pool[-1][1]
            fields["Valid Till (NT)"] = nt_raw
            fields["Valid Till"] = nt_raw
            fields.pop("NT Expiry Review", None)

    # Sync back to generic field if missing
    if nt_raw and not normalize_text(fields.get("Valid Till")):
        fields["Valid Till"] = nt_raw
    elif tr_raw and not normalize_text(fields.get("Valid Till")):
        fields["Valid Till"] = tr_raw

    selected = normalize_text(fields.get("Screening Category", "Automatic")).upper()

    if selected not in ("AUTOMATIC", "NT", "TR"):
        return {
            "expiry_status": "review",
            "expiry_date": "",
            "expiry_category": selected,
            "expiry_reason": "CATEGORY_REQUIRED",
            "expiry_message": build_error_message(
                "Validity Category Required",
                "The selected validity category could not be resolved.",
                "Choose 'NT' (Non-Transport) or 'TR' (Transport) in the options accordion.",
            ),
        }

    if selected == "AUTOMATIC":
        if nt_raw or fields.get("NT Expiry Review"):
            target_category = "NT"
        elif tr_raw or fields.get("TR Expiry Review"):
            target_category = "TR"
        else:
            target_category = printed_cats[0] if len(printed_cats) == 1 else "NT"
    else:
        target_category = selected

    if target_category == "NT":
        raw_date_str = nt_raw
        review_note = None if nt_raw else fields.get("NT Expiry Review")
        missing_code = "MISSING_NT_DATE"
        missing_msg = build_error_message(
            "Missing NT Validity Date",
            "The Non-Transport (NT) validity date was not detected on the card.",
            "Upload a clear horizontal photo showing the 'Valid Till' date clearly.",
        )
    else:
        raw_date_str = tr_raw
        review_note = None if tr_raw else fields.get("TR Expiry Review")
        missing_code = "MISSING_TR_DATE"
        missing_msg = build_error_message(
            "Missing TR Validity Date",
            "The Transport (TR) validity date was not detected on the card.",
            "Upload the reverse side or a clear photo showing the TR expiry date.",
        )

    if not raw_date_str:
        if review_note:
            return {
                "expiry_status": "review",
                "expiry_date": "",
                "expiry_category": target_category,
                "expiry_reason": "INVALID_DATE",
                "expiry_message": build_error_message(
                    f"Unreadable {target_category} Validity Date",
                    f"The {target_category} validity date text is blurry or unreadable.",
                    "Ensure indirect lighting and upload a focused photo of the date region.",
                ),
            }
        return {
            "expiry_status": "review",
            "expiry_date": "",
            "expiry_category": target_category,
            "expiry_reason": missing_code,
            "expiry_message": missing_msg,
        }

    parsed_expiry = parse_date_safely(raw_date_str)
    if parsed_expiry is None or review_note:
        return {
            "expiry_status": "review",
            "expiry_date": raw_date_str,
            "expiry_category": target_category,
            "expiry_reason": "INVALID_DATE",
            "expiry_message": build_error_message(
                f"Invalid {target_category} Validity Date",
                f"The date string ('{raw_date_str}') is not a valid calendar date.",
                "Retake the photo to ensure digits are not obscured by camera noise.",
            ),
        }

    if parsed_expiry < today:
        days_expired = (today - parsed_expiry).days
        return {
            "expiry_status": "expired",
            "expiry_date": parsed_expiry.isoformat(),
            "expiry_category": target_category,
            "expiry_reason": "EXPIRED",
            "expiry_message": build_error_message(
                f"Expired {target_category} Validity Date",
                f"The {target_category} validity date ({parsed_expiry:%d-%m-%Y}) expired {days_expired} days ago.",
                "Licence renewal is required. Visit the official Parivahan Sarathi portal.",
                include_rto_link=True,
            ),
        }

    return {
        "expiry_status": "valid",
        "expiry_date": parsed_expiry.isoformat(),
        "expiry_category": target_category,
        "expiry_reason": "",
        "expiry_message": (
            f"Active validity confirmed ({target_category}): {parsed_expiry:%d-%m-%Y}."
        ),
    }


# ==============================================================================
# 3. MASTER EVALUATION ENGINE
# ==============================================================================

def evaluate_licence_validation(
    fields: Dict[str, Any],
    today: Optional[date] = None,
    DL_NUMBER_PATTERN: Optional[re.Pattern] = None,
) -> Dict[str, Any]:
    """Execute all deterministic rules and compile the final validation decision."""
    today = today if today is not None else date.today()
    quality = fields.get("Image Quality Report") or {}
    soft_flags = quality.get("soft_flags") or {}

    # Pre-flight Gate
    if quality.get("hard_fail"):
        return {
            "status": "NOT VALID",
            "message": quality.get("message")
            or build_error_message(
                "Image Quality Check Failed",
                "Image quality metrics fall below minimum optical OCR thresholds.",
                "Upload a sharp, horizontal, well-lit photo of the driving licence.",
            ),
            "missing_fields": ["DL No", "Name", "DOB"],
            "dl_number_ok": False,
            "reason_codes": [quality.get("reason_code", "IMAGE_QUALITY_FAILED")],
            "official_portal_url": PARIVAHAN_PORTAL_URL,
            "expiry_status": "review",
            "expiry_date": "",
            "expiry_category": "",
        }

    recover_missing_core_fields(fields, today=today)
    full_ocr = str(fields.get("Full Image OCR", ""))
    detected_state = fields.get("State")

    # Extract DL Number
    raw_number = normalize_text(fields.get("DL No") or fields.get("ALT"))
    if not raw_number and DL_NUMBER_PATTERN is not None:
        matches = {m.group(1) for m in DL_NUMBER_PATTERN.finditer(full_ocr)}
        if matches:
            raw_number = sorted(matches, key=len, reverse=True)[0]
    if raw_number:
        fields.setdefault("Raw DL No", raw_number)

    # Validate DL Number with Contextual State Fallback
    dl_ok, normalized_dl, dl_code, dl_msg = validate_licence_number(
        raw_number, today=today, detected_state=detected_state
    )
    if dl_ok:
        fields["DL No"] = normalized_dl
        recover_missing_core_fields(fields, today=today)

    expiry = validate_selected_category_expiry(fields, today=today)
    missing = [label for label in ("DL No", "Name", "DOB") if not normalize_text(fields.get(label))]

    result: Dict[str, Any] = {
        "status": "NOT VALID",
        "message": "",
        "missing_fields": missing,
        "dl_number_ok": bool(dl_ok),
        "reason_codes": [],
        "official_portal_url": PARIVAHAN_PORTAL_URL,
        **expiry,
    }

    def fail(code: str, msg: str) -> Dict[str, Any]:
        result["reason_codes"] = [code]
        result["message"] = msg
        return result

    if fields.get("Processing Error"):
        return fail(
            "PROCESSING_ERROR",
            build_error_message(
                "Unreadable Image File",
                "An unexpected image array processing exception occurred.",
                "Re-upload the photo in standard JPEG or PNG format.",
            ),
        )

    # Document Type Classification Gate
    if OTHER_ID_KEYWORDS.search(full_ocr) and not DL_DOCUMENT_KEYWORDS.search(full_ocr):
        return fail(
            "WRONG_DOCUMENT",
            build_error_message(
                "Wrong Document Type",
                "The image contains an unsupported identity card (PAN, Aadhaar, or Voter ID).",
                "Upload a valid Indian Driving Licence photograph.",
            ),
        )

    detections = fields.get("OCR Detections") or []
    if not raw_number and not DL_DOCUMENT_KEYWORDS.search(full_ocr):
        if soft_flags.get("blurry"):
            return fail(
                "BLURRY_IMAGE",
                build_error_message(
                    "Blurry Image",
                    "The photo is too blurry to detect driving licence fields.",
                    "Retake the photo with the camera focused on the card.",
                ),
            )
        if len(detections) < 2:
            return fail(
                "WRONG_DOCUMENT",
                build_error_message(
                    "Unrecognized Document",
                    "No driving licence text or structure could be detected.",
                    "Upload a clear photo of an Indian Driving Licence.",
                ),
            )

    if missing and is_licence_cropped(fields):
        return fail(
            "INCOMPLETE_IMAGE",
            build_error_message(
                "Incomplete / Cropped Image",
                "Card borders are clipped, cutting off necessary verification fields.",
                "Capture all four corners of the licence within the camera frame.",
            ),
        )

    if not raw_number:
        return fail("NUMBER_UNREADABLE", dl_msg)

    dl_conf = fields.get("Field Confidence", {}).get("DL No", 1.0)
    if dl_conf is not None and dl_conf < 0.45:
        return fail(
            "NUMBER_LOW_CONFIDENCE",
            build_error_message(
                "Licence Number Unreadable",
                "OCR confidence on the licence number is below acceptable accuracy levels.",
                "Take a sharper picture under balanced lighting without glare.",
            ),
        )

    if not dl_ok:
        return fail(dl_code, dl_msg)

    # Holder Name Gate
    holder_name = normalize_text(fields.get("Name"))
    if not holder_name:
        return fail(
            "NAME_UNREADABLE",
            build_error_message(
                "Missing Required Field (Name)",
                "The licence holder name could not be extracted.",
                "Ensure the name field is unobstructed and clearly lit.",
            ),
        )

    # DOB Gate
    dob_raw = normalize_text(fields.get("DOB"))
    if not dob_raw:
        return fail(
            "DOB_MISSING",
            build_error_message(
                "Missing Required Field (DOB)",
                "The Date of Birth (DOB) field was not detected.",
                "Ensure the DOB is visible and free of reflections.",
            ),
        )

    dob_parsed = parse_date_safely(dob_raw)
    if dob_parsed is None or dob_parsed >= today or (today.year - dob_parsed.year) < 16:
        return fail(
            "DATE_INVALID",
            build_error_message(
                "Invalid Date of Birth",
                f"Extracted DOB ('{dob_raw}') is invalid or reflects an age under 16.",
                "Confirm that the birth date is clearly readable on the card.",
            ),
        )

    # Chronology Check: DOB vs Issue Dates
    dates: Dict[str, date] = {"DOB": dob_parsed}
    for field in ("Date of Issue", "Date of First Issue"):
        val = normalize_text(fields.get(field))
        if val:
            parsed = parse_date_safely(val)
            if parsed is None or parsed > today:
                return fail(
                    "DATE_INVALID",
                    build_error_message(
                        f"Invalid {field}",
                        f"Extracted date ('{val}') is invalid or listed in the future.",
                        "Upload a glare-free photo with clear issue dates.",
                    ),
                )
            dates[field] = parsed

    if any(dates["DOB"] >= dates[f] for f in ("Date of Issue", "Date of First Issue") if f in dates):
        return fail(
            "DATE_INCONSISTENT",
            build_error_message(
                "Chronological Date Inconsistency",
                "The extracted Date of Birth occurs on or after the Date of Issue.",
                "Ensure dates are not obscured, creating conflicting character reads.",
            ),
        )

    # Expiry Gate
    if expiry["expiry_status"] != "valid":
        return fail(expiry["expiry_reason"], expiry["expiry_message"])

    expiry_date = parse_date_safely(expiry["expiry_date"])
    if expiry_date and any(dates[f] > expiry_date for f in ("Date of Issue", "Date of First Issue", "DOB") if f in dates):
        return fail(
            "DATE_INCONSISTENT",
            build_error_message(
                "Inconsistent Validity Date",
                "The licence validity date precedes the Date of Issue or Date of Birth.",
                "Verify that all printed card dates are legible and re-upload.",
            ),
        )

    cat_label = expiry.get("expiry_category") or "NT"
    exp_str = expiry_date.strftime("%d-%m-%Y") if expiry_date else expiry.get("expiry_date", "")
    days_remaining = (expiry_date - today).days if expiry_date else 0

    result.update(
        status="VALID",
        message=(
            f"VALID | Verification Successful: Indian Driving Licence ({normalized_dl}) for {holder_name} "
            f"passed all image quality, licence number, mandatory field, and validity checks. "
            f"Category ({cat_label}) Valid Till: {exp_str} ({days_remaining} days remaining). "
            f"| {AUTHENTICITY_SCOPE_NOTE}"
        ),
    )
    return result