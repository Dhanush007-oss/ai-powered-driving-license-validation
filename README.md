# AI-Powered Indian Driving Licence Validation

A local Gradio web application that reads an Indian driving licence image, extracts printed details, checks image quality, and evaluates the selected licence validity date. The application combines YOLO-based field detection, PaddleOCR text recognition, and deterministic Python validation rules. When a check fails, the interface explains the detected issue and suggests how to fix it.

> **Important scope limitation:** `VALID` means the uploaded image passed this application's automated image, field, number-format, date, and expiry checks. It does **not** prove that the physical licence is genuine, untampered, active in government records, or free of suspension. The application does not perform live government-record verification. For an official check, use the [Parivahan Sarathi portal](https://sarathi.parivahan.gov.in/) or contact the relevant RTO.

## What the application does

- Checks whether an uploaded image is readable enough for automated processing, including basic resolution, blur, lighting, and glare indicators.
- Uses a YOLO detection model to locate printed licence fields.
- Uses PaddleOCR to read detected regions and text across the card, with additional OCR extraction when the initial result is uncertain.
- Extracts and displays fields such as licence number, name, date of birth, issue dates, validity dates, blood group, relation, address, RTO, state, and vehicle type when detected.
- Applies deterministic rules to the licence number, required fields, date parsing, date chronology, and selected expiry date.
- Supports **Automatic**, **NT** (Non-Transport), and **TR** (Transport) validity-category selection.
- Shows a highlighted image, an extracted-details table, a `VALID` or `NOT VALID` result, and user-friendly guidance when a check fails.
- Can save extracted fields and validation details to a JSON file when run through the command-line interface.

## How validation works

1. **Image input:** The user uploads the front of the card and can optionally upload the back. The selected validity category is passed into the processing pipeline.
2. **Image-quality gate:** The rules screen the image for unreadable or unsuitable input before treating it as a licence image.
3. **Field detection:** The configured YOLO model identifies regions such as the licence number, name, DOB, address, blood group, relation, RTO, state, and vehicle type.
4. **Text extraction:** PaddleOCR reads the detected crops and the full card. Image preprocessing and a second OCR pass can be used when important fields remain uncertain.
5. **Field recovery and normalization:** The code normalizes OCR text and tries to recover relevant dates and core fields from the recognized text.
6. **Rule evaluation:** The validation engine checks document indicators, licence-number format and state prefix, required fields, date validity and chronology, and the expiry date for the selected category.
7. **Result display:** The UI renders the result and a useful explanation. A failed automated check is reported as `NOT VALID`; this is a screening outcome, not a government decision.

The rules are implemented in Python rather than being left entirely to a generative model, so the final outcome follows explicit validation conditions.

## Project structure

The code imports files by their exact names. Keep the filenames `app.py`, `inference.py`, `validation_rules.py`, and `paddleocr_vl_worker.py` as shown below; changing a name can break imports or the optional worker path.

```text
indian-id-validator/
├── app.py                              # Gradio web interface and result rendering
├── inference.py                        # Model loading, image processing, OCR, field merging
├── validation_rules.py                 # Image, number, mandatory-field, date and expiry rules
├── config.json                         # Detector checkpoint path and class-name mapping
├── models/
│   └── Driving_License.pt               # Optional local YOLO checkpoint
├── samples/
│   └── driving_license_front.jpg        # Optional UI example image
├── paddleocr_vl_worker.py               # Optional PaddleOCR-VL subprocess worker
├── install_paddleocr_vl_integration.py  # Optional integration/patch utility
├── detected_license.json                # Example/output JSON; may contain personal data
└── vl_runtime/                           # Optional isolated PaddleOCR-VL environment
```

Some entries are optional and may not be present in a fresh checkout:

| File or folder | Purpose |
|---|---|
| `app.py` | Builds the Gradio interface, accepts front/back images and the NT/TR choice, calls the inference pipeline, and displays the verdict and extracted fields. |
| `inference.py` | Loads the YOLO and PaddleOCR components; preprocesses images; performs detection and OCR; merges fields from both sides; runs the validation engine; and optionally writes JSON output. |
| `validation_rules.py` | Contains the deterministic validation logic and user-facing failure messages. It includes image-quality checks, Indian licence-number checks, mandatory-field checks, and category-specific expiry evaluation. |
| `config.json` | Supplies the detector model path and the class labels in the order expected by that model. The labels must match the trained checkpoint. |
| `models/Driving_License.pt` | Local detector checkpoint. If the configured local checkpoint is absent, `inference.py` attempts to download the configured model file from the Hugging Face repository `logasanjeev/indian-id-validator`. |
| `samples/driving_license_front.jpg` | Optional sample image. The UI shows the sample example only when this exact path exists. |
| `paddleocr_vl_worker.py` | Worker implementation for PaddleOCR-VL. The current inference code defines `_should_run_paddleocr_vl(...)` to return `False`, so this worker is **not used in the default validation flow**. |
| `install_paddleocr_vl_integration.py` | Maintenance utility that patches an existing `inference.py` and saves a backup before editing. It is not needed to launch the current application when the integration code is already present. |
| `vl_runtime/` | Separate Python environment expected by the optional PaddleOCR-VL worker on Windows. It is not required for the default flow while that fallback remains disabled. |
| `detected_license.json` | Example or generated extraction output. Do not commit real users' licence details; redact personal data before sharing test output. |

### Model configuration

The supplied `config.json` points to:

```json
{
  "models": {
    "Driving_License": {
      "type": "detection",
      "path": "models/Driving_License.pt",
      "classes": [
        "Address",
        "Blood Group",
        "DL No",
        "DOB",
        "Name",
        "Relation With",
        "RTO",
        "State",
        "Vehicle Type"
      ]
    }
  }
}
```

If you use a different detector checkpoint, update the `path` and `classes` to match that checkpoint. The class order is significant. Keep `config.json` in the same project directory as `inference.py`.

## Requirements

- Python 3.11 is a practical starting point for this project.
- A working pip installation and internet access for the first-run model/config downloads, unless the required files are already available locally.
- The Python packages imported by the application: Gradio, Ultralytics, PaddleOCR, PaddlePaddle, OpenCV, NumPy, PyTorch, and `huggingface_hub`.

The supplied files do not include a pinned `requirements.txt`, so the install command below is a baseline rather than a lockfile. For a reproducible deployment, install and record the versions confirmed in your working environment.

## Setup on Windows

Run the following commands in PowerShell from the project root—the directory containing `app.py` and `inference.py`.

### 1. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation because of the execution policy, either activate the environment using the policy approved for your machine or invoke the environment's Python directly, for example `.\.venv\Scripts\python.exe -m pip install ...`.

### 2. Install the dependencies

```powershell
python -m pip install gradio ultralytics paddleocr paddlepaddle opencv-python numpy torch huggingface_hub
```

PaddleOCR and PaddlePaddle versions must be compatible with each other and with your Python version. If installation reports a wheel or dependency conflict, use a compatible version combination rather than randomly upgrading individual packages. The optional PaddleOCR-VL worker has its own environment requirement and is not needed for the default flow.

### 3. Check the project files

At minimum, confirm these files are in the same project root:

```text
app.py
inference.py
validation_rules.py
config.json
```

The detector checkpoint can be placed at `models/Driving_License.pt`. If it is not there, the inference code attempts to fetch the configured checkpoint from Hugging Face. The optional sample image should be placed at `samples/driving_license_front.jpg` if you want the UI to display it.

### 4. Start the web application

```powershell
python app.py
```

The application starts the Gradio interface and asks the browser to open it. Gradio normally uses `http://127.0.0.1:7860` for a local session; use the local address printed in the terminal if it differs.

The first run can take longer while the detector or OCR models are downloaded and initialized. Keep the terminal open while using the app.

## How to use the web interface

1. Open the local Gradio page.
2. Upload a clear photo of the **front** of an Indian driving licence.
3. If the validity date is on the reverse side, expand the optional section and upload the **back** photo too.
4. Choose the validity category:
   - **Automatic** — lets the rules determine which detected category/date to evaluate.
   - **NT** — checks the Non-Transport expiry date.
   - **TR** — checks the Transport expiry date.
5. Select **Validate licence**.
6. Review the highlighted image, the extracted fields, the verdict, and any displayed fix guidance.
7. If the result is `NOT VALID`, follow the message and upload a clearer or more complete image when appropriate. If the printed details appear correct but the automated result is unexpected, confirm the licence through the official Parivahan portal or an RTO.

For best OCR results, use a sharp, well-lit photo, avoid flash reflections, keep the text unobstructed, and include all card edges. Upload the side that actually contains the required validity details.

## Run extraction from the command line

`inference.py` also exposes a command-line interface. From the project root:

```powershell
python inference.py "samples/driving_license_front.jpg" --category Automatic --output-json "detected_license.json"
```

Check a specific validity category:

```powershell
python inference.py "samples/driving_license_front.jpg" --category NT --output-json "detected_license.json"
python inference.py "samples/driving_license_front.jpg" --category TR --output-json "detected_license.json"
```

Provide the reverse side when required:

```powershell
python inference.py "samples/driving_license_front.jpg" --reverse-image "samples/driving_license_back.jpg" --category TR --output-json "detected_license.json"
```

To print the extracted result without writing the JSON file, add `--no-save-json`:

```powershell
python inference.py "samples/driving_license_front.jpg" --no-save-json
```

Replace the example paths with real paths to your test images. These commands assume the relevant image files exist.

## Main validation rules

The validation engine uses the following checks. Exact outcomes depend on the data extracted from the uploaded image.

| Check | What it evaluates | Typical next step if it fails |
|---|---|---|
| Image integrity and quality | Whether an image was supplied, can be read, and meets basic image-quality thresholds for detection/OCR. | Upload a readable JPG or PNG, retake the photo in better light, reduce glare, and keep the card in focus. |
| Document type | Uses recognized text and licence indicators to distinguish a driving licence from unsupported identity documents. | Upload an Indian driving licence rather than Aadhaar, PAN, Voter ID, or another document. |
| Licence number | Checks the detected number against configured Indian state/UT prefixes and supported number patterns. | Retake the image so the complete number and its leading state code are sharp and unobstructed. |
| Required identity fields | Requires core details such as licence number, name, and DOB for the automated checks. | Ensure those fields are visible and legible. |
| Date parsing and chronology | Parses recognized dates and checks relevant relationships between DOB, issue dates, and validity dates. | Verify the printed dates and retake the photo if digits are blurred or incorrectly read. |
| NT/TR expiry | Evaluates the selected category's expiry date and checks whether that date is current. | Upload the side showing the correct category and validity date, or choose the intended category. |
| Card completeness | Checks for signs that required fields may have been clipped from the image. | Include all four card corners and avoid cutting off printed details. |

A `NOT VALID` result can mean the image or extracted data failed one of these programmed checks; it is not by itself proof that the licence is legally invalid.

## Troubleshooting

**`ModuleNotFoundError` when starting the app**  
Activate the project's virtual environment and install the missing package in that same environment. Confirm that `python` points to `.venv` by running `python -c "import sys; print(sys.executable)"`.

**`No module named inference` or an import error from `app.py`**  
Make sure the inference file is named exactly `inference.py` and sits beside `app.py`. The code will not import a file left under a downloaded name such as `inference(5).py`.

**Detector model download fails**  
Check your internet connection and Hugging Face access, or place the compatible checkpoint at the path specified in `config.json`. Make sure the checkpoint and the class list belong together.

**OCR misses a date or field**  
Use a sharper image with balanced lighting, remove reflections, include the relevant side of the card, and select NT/TR explicitly when the intended category is known. OCR can still misread unusual layouts, faint print, or legacy formats.

**PaddleOCR-VL runtime error**  
The optional worker expects `paddleocr_vl_worker.py` and a separate Windows environment at `vl_runtime/.venv/Scripts/python.exe`. The current code disables this fallback by default, so this setup should not be necessary for a standard run. Do not run the integration installer unless you deliberately intend to patch `inference.py` and have a backup.

## Privacy and safe testing

- Use synthetic, public test samples, or licence images you are authorized to process.
- OCR results and JSON output may contain names, dates of birth, addresses, licence numbers, and other personal information. Do not publish raw output files or screenshots containing those values.
- Model/config downloads may contact Hugging Face during first-time setup. This application does not contact a government system to confirm a licence's legal status.
- Test representative cards from different states and layouts. A rule-based screen can produce false rejections when OCR extracts a field incorrectly or a card format differs from the model's training data.

## Future Improvements

The following enhancements could make the Indian driving licence screening workflow more accurate, robust, and easier to integrate into other systems. These are **planned ideas, not features currently implemented**.

- **Transformer-based OCR:** Evaluate a transformer OCR model such as TrOCR for difficult licence text, while benchmarking it against the current PaddleOCR pipeline on samples from different states, card layouts, and issue years.
- **Layout-aware field extraction:** Explore document-understanding models such as LayoutLMv3 to use text positions and relationships when identifying licence numbers, names, dates, vehicle classes, and issuing authorities, reducing reliance on regular expressions alone.
- **Licence portrait and face matching:** Where appropriate and with explicit user consent, investigate extracting the portrait printed on a driving licence and comparing it with a submitted selfie. Treat any match as an additional screening signal, not proof of identity or legal validity.
- **Signature-region analysis:** Detect the licence signature area and assess whether it is present and legible. Any future signature comparison would need suitable reference data and must not be presented as conclusive verification on its own.
- **Driving-licence QR or barcode checks:** Detect and decode QR codes or barcodes when they are present on supported licence formats. Validate decoded information only against a documented, trusted source; a readable code alone does not establish that a licence is genuine.
- **Tampering and alteration indicators:** Research checks for inconsistent fonts, suspicious editing boundaries, unusual card edges, portrait-region irregularities, and layout inconsistencies. Report these as indicators for further review rather than definitive proof of fraud.
- **Multilingual OCR:** Extend text recognition to Hindi, Tamil, and other Indian languages that may appear on driving licences, and test field extraction across regional scripts and mixed-language cards.
- **Trained document-type classification:** Add a dedicated image-classification model to distinguish Indian driving licences from unsupported documents, complementing the existing YOLO field detector and keyword-based document checks.
- **REST API:** Provide a FastAPI service alongside the current Gradio interface so other applications can submit licence images and receive structured extraction and validation results.
- **PDF input support:** Accept PDF scans of driving licences, safely convert pages to images, and process the relevant page or pages through the existing pipeline.
- **Automatic orientation and perspective correction:** Detect rotation, tilt, and perspective distortion before OCR so photos taken at an angle can be normalized more reliably.
- **Broader automated testing:** Build a test suite for image-quality failures, missing or misread mandatory fields, Indian state/UT prefixes, legacy licence-number formats, date parsing, and independent NT/TR expiry selection.

Any future enhancement should preserve the distinction between **automated image screening** and **official licence verification**. Without a trusted and authorized government verification source, the application should not claim to confirm that a licence is genuine, active, or free from suspension.

## Current implementation notes

- The result interface supports the binary statuses `VALID` and `NOT VALID`.
- The OCR and validation pipeline is in `inference.py` and `validation_rules.py`; the Gradio layout and user guidance are in `app.py`.
- The optional PaddleOCR-VL code exists, but the `_should_run_paddleocr_vl(...)` gate currently returns `False`, so it is not part of normal processing.
- The UI prints OCR/date diagnostics to the terminal during validation. Avoid using a shared terminal log for real personal data unless its access and retention are controlled.

## License and model provenance

The supplied Python files do not specify a project-wide software licence. Add a `LICENSE` file before distributing the project if you intend to license it to others. The detector fallback is configured to download from the Hugging Face repository [`logasanjeev/indian-id-validator`](https://huggingface.co/logasanjeev/indian-id-validator); check that repository's model terms and provenance before redistribution or commercial use.
