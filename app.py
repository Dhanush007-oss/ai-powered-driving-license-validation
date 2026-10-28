import html
import os
import re

import gradio as gr

from inference import DISPLAY_FIELDS, normalize_field_value, process_driving_license

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_LICENCE = os.path.join(APP_DIR, "samples", "driving_license_front.jpg")

STATUS_STYLES = {
    "VALID": ("#d7ffe8", "linear-gradient(135deg, #0d7a4f, #14b8a6)"),
    "NOT VALID": ("#ffe4e6", "linear-gradient(135deg, #9f1239, #ea580c)"),
}


def _format_links(text):
    escaped = html.escape(str(text or "").strip())
    return re.sub(
        r"(https?://[^\s|<]+)",
        r"<a href='\1' target='_blank' class='result-link'>\1 ↗</a>",
        escaped,
    )


def _build_message_spotlight(status, raw_message):
    """Render the user-friendly message in a dedicated visual spotlight card above the table."""
    parts = [part.strip() for part in str(raw_message or "").split("|") if part.strip()]
    title = "Card Validity Confirmed" if status == "VALID" else "Action Needed to Complete Verification"
    what_happened = raw_message
    how_to_fix = ""

    for part in parts:
        if part.upper() in ("VALID", "NOT VALID"):
            continue
        if part.lower().startswith("error (") and "):" in part:
            header, detail = part.split("):", 1)
            title = header.replace("Error (", "Issue Detected: ").strip()
            what_happened = detail.strip()
        elif part.lower().startswith("verification successful:"):
            title = "All Card & Validity Checks Passed"
            what_happened = part.split(":", 1)[1].strip()
        elif part.lower().startswith("how to fix:"):
            how_to_fix = part.split(":", 1)[1].strip()
        elif not part.lower().startswith("scope & future") and not part.lower().startswith("official rto"):
            what_happened = part

    if status == "VALID":
        return f"""
        <div class="uf-spotlight uf-valid">
          <span class="uf-badge uf-badge-valid">Verification Outcome</span>
          <h3 class="uf-heading">{html.escape(title)}</h3>
          <p class="uf-text">{_format_links(what_happened)}</p>
        </div>
        """

    fix_html = ""
    if how_to_fix:
        fix_html = f"""
        <div class="uf-subbox uf-fix">
          <span class="uf-sublabel uf-fix-label">How to Fix It</span>
          <p class="uf-text">{_format_links(how_to_fix)}</p>
        </div>
        """

    return f"""
    <div class="uf-spotlight uf-invalid">
      <span class="uf-badge uf-badge-invalid">User Guidance</span>
      <h3 class="uf-heading">{html.escape(title)}</h3>
      <div class="uf-subbox uf-issue">
        <span class="uf-sublabel uf-issue-label">What We Found</span>
        <p class="uf-text">{_format_links(what_happened)}</p>
      </div>
      {fix_html}
    </div>
    """


def render_results(fields, assessment):
    # Only the validator's explicit binary status can reach the result badge.
    status = assessment.get("status")
    status = status if status in STATUS_STYLES else "NOT VALID"
    color, background = STATUS_STYLES[status]

    # Get the primary message cleanly without duplication
    raw_message = (
        assessment.get("message")
        or "The licence could not be screened. Upload a clearer image showing the licence details and validity information."
    )

    portal_url = fields.get("Official Portal") or "https://sarathi.parivahan.gov.in/"
    spotlight_html = _build_message_spotlight(status, raw_message)

    rows = []
    for label in DISPLAY_FIELDS:
        value = normalize_field_value(fields.get(label, ""))
        if label == "Valid Till (TR)" and not value:
            value = normalize_field_value(fields.get("Valid Till (T)", ""))
        display_value = html.escape(value) if value else "—"
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td>{display_value}</td></tr>"
        )

    return f"""
    <div class="result-card" data-outcome="{status}">
      <div class="result-sheen"></div>
      <div class="result-head">
        <div class="status-pill" data-outcome="{status}" style="color:{color};background:{background};">{status}</div>
        <span class="result-kicker">Card Verification Summary</span>
      </div>
      {spotlight_html}
      <div class="table-header-bar">
        <span class="table-title">Extracted Driving Licence Details</span>
      </div>
      <table>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      <div class="legal-disclaimer">
        <strong>Important Note — Not a Genuine / Legal Original-Document Verification:</strong>
        This application performs an automated visual check of the uploaded card photo to read printed details and check validity dates. Because private applications do not have access to restricted Government API keys or cryptographic signatures, a photo check cannot confirm whether a physical licence is an original government document, a duplicate, tampered with, or suspended. For official legal verification or licence renewal, please visit your local RTO or the official <a href="{portal_url}" target="_blank" class="result-link">Parivahan Sarathi Portal ({portal_url}) ↗</a>.
      </div>
    </div>
    """


def validate_licence(image):
    if image is None or (isinstance(image, dict) and image.get("front") is None and image.get("back") is None):
        return None, render_results({}, {
            "status": "NOT VALID",
            "message": "NOT VALID | Error (Image Not Uploaded): No driving licence photo was uploaded. | How to fix: Upload a clear, upright horizontal photo of your driving licence, including the side showing its validity date.",
        })

    fields, annotated, assessment = process_driving_license(image)

    print("\n--- DATE EXTRACTION DEBUG ---")
    print("Full OCR:", fields.get("Full Image OCR"))
    print("Detected dates:", fields.get("Detected Dates"))
    print("Validity categories:", fields.get("Validity Categories"))
    print("NT expiry:", fields.get("Valid Till (NT)"))
    print("TR expiry:", fields.get("Valid Till (TR)"))
    print("Date of issue:", fields.get("Date of Issue"))
    print("Field review:", fields.get("Field Review"))
    print("Reason codes:", assessment.get("reason_codes"))
    print("Assessment:", assessment.get("message"))
    print("--- END DEBUG ---\n")

    return annotated, render_results(fields, assessment)


def validate_uploaded_licence(image, reverse_image, category):
    return validate_licence({"front": image, "back": reverse_image, "category": category})


def clear_licence():
    return None, None, ""


CUSTOM_CSS = """
:root {
  --ink: #07080f;
  --gold: #e7c36a;
  --gold-2: #f6e2a8;
  --saffron: #ff8a3c;
  --teal: #3ee0c8;
  --glass: rgba(12, 16, 28, .62);
  --stroke: rgba(231, 195, 106, .22);
}
html, body {
  margin: 0;
  min-height: 100%;
  background: #07080f;
}
.gradio-container.gradio-container {
  position: relative;
  isolation: isolate;
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh;
  margin: 0 auto !important;
  padding: clamp(16px, 2.4vw, 36px) clamp(16px, 4vw, 64px) 48px !important;
  box-sizing: border-box;
  font-family: "Outfit", "Segoe UI", system-ui, sans-serif !important;
  color: #eef3ff !important;
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(255, 138, 60, .22), transparent 55%),
    radial-gradient(900px 520px at 92% 8%, rgba(62, 224, 200, .16), transparent 50%),
    radial-gradient(700px 500px at 70% 90%, rgba(88, 80, 220, .18), transparent 55%),
    linear-gradient(180deg, #0a0c16 0%, #07080f 42%, #0b1020 100%) !important;
  --body-text-color: #eef3ff !important;
  --body-text-color-subdued: #b7c0d6 !important;
  --block-label-text-color: #f6e2a8 !important;
  --block-label-background-fill: rgba(231, 195, 106, .12) !important;
  --block-info-text-color: #c9d2ea !important;
  --input-text-color: #f4f7ff !important;
  --input-placeholder-color: #8e99b8 !important;
  --link-text-color: #e7c36a !important;
  --background-fill-primary: rgba(10, 13, 24, .8) !important;
  --background-fill-secondary: rgba(18, 22, 38, .72) !important;
  --border-color-primary: rgba(231, 195, 106, .18) !important;
  --input-background-fill: rgba(8, 10, 18, .72) !important;
}
.gradio-container.gradio-container::before,
.gradio-container.gradio-container::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}
.gradio-container.gradio-container::before {
  background-image:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse at 50% 20%, #000 30%, transparent 78%);
  animation: grid-drift 28s linear infinite;
}
.gradio-container.gradio-container::after {
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.08) 3px);
  opacity: .35;
}
.gradio-container .main, .gradio-container .contain {
  position: relative;
  z-index: 1;
  width: min(1280px, 100%) !important;
  max-width: 1280px !important;
  margin-inline: auto !important;
  padding: 0 !important;
}
.glass-reset, #site-nav, #site-hero, #site-features, #site-steps, #site-footer,
#site-nav .block, #site-hero .block, #site-features .block, #site-steps .block, #site-footer .block,
#workspace-label .block, #results-panel {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}
#site-nav .html-container, #site-hero .html-container, #site-features .html-container,
#site-steps .html-container, #site-footer .html-container, #workspace-label .html-container {
  padding: 0 !important;
  margin: 0 !important;
}
.dl-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 22px;
  margin-bottom: 18px;
  border: 1px solid var(--stroke);
  border-radius: 999px;
  background: rgba(8, 10, 18, .6);
  backdrop-filter: blur(18px);
  animation: licence-enter .6s ease both;
}
.dl-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  font-size: .78rem;
  color: #f6e2a8;
}
.brand-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #e7c36a, #3ee0c8);
  box-shadow: 0 0 12px rgba(62, 224, 200, .55);
}
.dl-nav span { color: #b0bdd8; font-size: .82rem; }
.dl-hero {
  position: relative;
  overflow: hidden;
  padding: clamp(34px, 5vw, 68px) clamp(24px, 4vw, 60px);
  margin-bottom: 22px;
  border-radius: 36px;
  border: 1px solid rgba(255,255,255,.08);
  background:
    linear-gradient(120deg, rgba(12,16,32,.75), rgba(18, 12, 8, .58)),
    radial-gradient(circle at 80% 20%, rgba(231, 195, 106, .18), transparent 32%);
  box-shadow: 0 30px 80px rgba(0,0,0,.35);
  animation: licence-enter .8s ease both;
}
.dl-hero::before, .dl-hero::after {
  content: "";
  position: absolute;
  border-radius: 50%;
  filter: blur(8px);
  pointer-events: none;
}
.dl-hero::before {
  width: 280px;
  height: 280px;
  right: -40px;
  top: -70px;
  background: radial-gradient(circle, rgba(255,138,60,.45), transparent 70%);
  animation: orb-float 10s ease-in-out infinite;
}
.dl-hero::after {
  width: 220px;
  height: 220px;
  left: 8%;
  bottom: -80px;
  background: radial-gradient(circle, rgba(62,224,200,.35), transparent 70%);
  animation: orb-float 13s ease-in-out infinite reverse;
}
.dl-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 18px;
  padding: 7px 14px;
  border-radius: 999px;
  border: 1px solid rgba(231,195,106,.35);
  color: #f6e2a8;
  font-size: .78rem;
  letter-spacing: .16em;
  text-transform: uppercase;
  background: rgba(231,195,106,.08);
  animation: pulse-glow 3.4s ease-in-out infinite;
}
.dl-hero h1 {
  max-width: 18ch;
  margin: 0 0 18px;
  font-family: "Cormorant Garamond", Georgia, serif;
  font-size: clamp(2.6rem, 6vw, 4.8rem);
  line-height: 1.02;
  letter-spacing: -.025em;
  color: #fff8ea;
  text-shadow: 0 10px 40px rgba(0,0,0,.35);
}
.dl-hero h1 em {
  font-style: italic;
  background: linear-gradient(90deg, #f6e2a8, #ff8a3c, #3ee0c8, #f6e2a8);
  background-size: 220% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: gold-flow 8s linear infinite;
}
.dl-hero p {
  max-width: 64ch;
  margin: 0 0 26px;
  color: #ced8ee;
  font-size: clamp(1rem, 1.4vw, 1.16rem);
  line-height: 1.75;
}
.dl-pills { display: flex; flex-wrap: wrap; gap: 10px; }
.dl-pills i {
  font-style: normal;
  padding: 8px 15px;
  border-radius: 999px;
  border: 1px solid rgba(231,195,106,.24);
  background: rgba(255,255,255,.04);
  color: #eef3ff;
  font-size: .86rem;
}
.dl-features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 8px 0 28px;
}
.dl-card {
  position: relative;
  overflow: hidden;
  padding: 22px 20px 20px;
  border-radius: 22px;
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  backdrop-filter: blur(16px);
  animation: licence-enter .9s ease both;
  transition: transform .35s ease, border-color .35s ease, box-shadow .35s ease;
}
.dl-card:hover {
  transform: translateY(-6px);
  border-color: rgba(231,195,106,.5);
  box-shadow: 0 18px 40px rgba(0,0,0,.28);
}
.dl-card b {
  display: block;
  margin-bottom: 8px;
  color: #f6e2a8;
  font-size: .92rem;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.dl-card p { margin: 0; color: #c5cde0; line-height: 1.65; font-size: .95rem; }
.section-label {
  margin: 8px 0 12px;
  color: #f6e2a8;
  letter-spacing: .22em;
  text-transform: uppercase;
  font-size: .76rem;
  animation: licence-enter .5s ease both;
}
.gradio-container .block {
  background: rgba(10, 13, 24, .72) !important;
  border: 1px solid rgba(231, 195, 106, .16) !important;
  border-radius: 24px !important;
  box-shadow: 0 18px 50px rgba(0,0,0,.22);
  backdrop-filter: blur(18px);
  transition: border-color .3s ease, transform .3s ease, box-shadow .3s ease;
}
.gradio-container .image-container.block:hover {
  border-color: rgba(62, 224, 200, .4) !important;
  box-shadow: 0 22px 50px rgba(62, 224, 200, .08);
}
.gradio-container .block label,
.gradio-container .block span.float-label {
  color: #f6e2a8 !important;
  background-color: rgba(12, 16, 28, .92) !important;
  font-weight: 600 !important;
  border: 1px solid rgba(231,195,106,.2) !important;
}
.gradio-container [data-testid="upload-text"],
.gradio-container [data-testid="upload-text"] *,
.gradio-container .upload-container > .wrap,
.gradio-container .upload-container > .wrap * {
  color: #f6e2a8 !important;
}
.gradio-container [data-testid="upload-text"] .or,
.gradio-container .upload-container .or,
.gradio-container .empty,
.gradio-container .empty .icon,
.gradio-container .empty svg {
  color: #8ea0c6 !important;
}
.gradio-container footer,
.gradio-container footer a,
.gradio-container footer button,
.gradio-container footer span {
  color: #8b95b0 !important;
}

/* Action Row: Forces Validate licence and Clear onto the exact same line */
#action-row {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 16px !important;
  margin: 14px 0 !important;
}
#action-row > * {
  margin: 0 !important;
}
.gradio-container button.primary,
#clear-button {
  position: relative;
  overflow: hidden;
  height: 58px !important;
  min-height: 58px !important;
  max-height: 58px !important;
  margin: 0 !important;
  border-radius: 18px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: .04em;
  text-transform: uppercase;
  box-sizing: border-box !important;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.gradio-container button.primary {
  border: 0 !important;
  background: linear-gradient(110deg, #c9842a, #e7c36a, #3ee0c8) !important;
  background-size: 180% 100% !important;
  color: #14120c !important;
  box-shadow: 0 12px 30px rgba(231, 195, 106, .28);
  animation: btn-shift 8s ease infinite;
}
.gradio-container button.primary::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.38) 50%, transparent 70%);
  transform: translateX(-120%);
}
.gradio-container button.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 36px rgba(231, 195, 106, .4);
}
.gradio-container button.primary:hover::after { animation: licence-shine .7s ease; }
.gradio-container button.primary:active { transform: translateY(0); }
.gradio-container :is(button, a, input, textarea, select):focus-visible {
  outline: 3px solid #3ee0c8 !important;
  outline-offset: 3px;
}
#clear-button {
  background: rgba(255,255,255,.05) !important;
  color: #e8eefc !important;
  border: 1px solid rgba(231, 195, 106, .25) !important;
}
#clear-button:hover {
  border-color: #e7c36a !important;
  color: #f6e2a8 !important;
  transform: translateY(-2px);
}

/* Result Card, User-Friendly Message Spotlight & Table */
.result-card {
  position: relative;
  overflow: hidden;
  box-sizing: border-box;
  width: 100%;
  background: linear-gradient(180deg, rgba(18, 22, 38, .92), rgba(8, 10, 18, .94));
  border: 1px solid rgba(231,195,106,.28);
  border-radius: 28px;
  padding: clamp(22px, 3vw, 40px);
  box-shadow: 0 24px 60px rgba(0,0,0,.35);
  animation: licence-enter .55s ease both;
}
.result-sheen {
  position: absolute;
  inset: 0 auto 0 -40%;
  width: 40%;
  background: linear-gradient(90deg, transparent, rgba(246, 226, 168, .12), transparent);
  animation: sheen-sweep 5.5s ease-in-out infinite;
  pointer-events: none;
}
.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 18px;
}
.result-kicker {
  color: #9fb0d0;
  letter-spacing: .16em;
  text-transform: uppercase;
  font-size: .74rem;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  font-weight: 750;
  letter-spacing: .12em;
  border-radius: 999px;
  padding: 12px 20px;
  line-height: 1.5;
  font-size: clamp(1rem, 1.8vw, 1.28rem);
  box-shadow: 0 10px 28px rgba(0,0,0,.25);
  animation: pill-pulse 2.8s ease-in-out infinite;
}
.uf-spotlight {
  border-radius: 20px;
  padding: 20px 22px;
  margin-bottom: 24px;
  border: 1px solid;
}
.uf-valid {
  background: linear-gradient(135deg, rgba(13, 122, 79, .22), rgba(20, 184, 166, .08));
  border-color: rgba(62, 224, 200, .4);
}
.uf-invalid {
  background: linear-gradient(135deg, rgba(159, 18, 57, .22), rgba(234, 88, 12, .08));
  border-color: rgba(255, 138, 60, .45);
}
.uf-badge {
  display: inline-block;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 999px;
  margin-bottom: 8px;
}
.uf-badge-valid { background: rgba(62, 224, 200, .18); color: #7cf0dc; }
.uf-badge-invalid { background: rgba(255, 138, 60, .2); color: #ffb37a; }
.uf-heading {
  margin: 0 0 12px;
  color: #fff8ea;
  font-size: clamp(1.15rem, 1.7vw, 1.38rem);
  font-weight: 700;
}
.uf-subbox {
  background: rgba(7, 8, 15, .55);
  border-radius: 14px;
  padding: 14px 16px;
  margin-top: 10px;
  border: 1px solid rgba(255,255,255,.08);
}
.uf-fix { border-left: 4px solid #3ee0c8; }
.uf-sublabel {
  display: block;
  font-size: .75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: 4px;
}
.uf-issue-label { color: #ffb37a; }
.uf-fix-label { color: #3ee0c8; }
.uf-text {
  margin: 0;
  color: #eef3ff;
  font-size: .98rem;
  line-height: 1.68;
}
.table-header-bar {
  margin: 8px 0 10px;
}
.table-title {
  color: #f6e2a8;
  font-size: .8rem;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.result-link { color: #e7c36a !important; text-decoration: underline; font-weight: 600; }
.result-card table { width: 100%; border-collapse: collapse; margin-top: 8px; table-layout: fixed; }
.result-card th, .result-card td {
  text-align: left;
  padding: 15px 14px;
  border-bottom: 1px solid rgba(255,255,255,.08);
  vertical-align: top;
  overflow-wrap: anywhere;
  line-height: 1.6;
}
.result-card th { width: 32%; color: #b8c4de; font-weight: 500; }
.result-card td { color: #fff8ea; font-weight: 650; }
.result-card tr:nth-child(odd) { background: rgba(255,255,255,.03); }
.result-card tr { transition: background .2s ease; }
.result-card tr:hover { background: rgba(231, 195, 106, .08); }
.legal-disclaimer {
  margin-top: 22px;
  padding: 16px 18px;
  border-radius: 16px;
  background: rgba(255, 255, 255, .03);
  border: 1px solid rgba(231, 195, 106, .22);
  color: #b7c3de;
  font-size: .88rem;
  line-height: 1.65;
}
.legal-disclaimer strong { color: #f6e2a8; }

#upload-panel { border-color: rgba(255, 138, 60, .28) !important; }
#detected-panel { border-color: rgba(62, 224, 200, .28) !important; }
#results-panel { margin-top: 8px; }
#upload-panel label { color: #ffb37a !important; }
#detected-panel label { color: #7cf0dc !important; }
.dl-steps {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin: 28px 0 18px;
}
.dl-step {
  padding: 18px 16px;
  border-radius: 20px;
  border: 1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03);
  animation: licence-enter 1s ease both;
}
.dl-step strong {
  display: block;
  margin-bottom: 8px;
  color: #fff8ea;
  font-family: "Cormorant Garamond", Georgia, serif;
  font-size: 1.35rem;
}
.dl-step p { margin: 0; color: #aeb8d0; font-size: .9rem; line-height: 1.6; }
.dl-footer {
  margin-top: 28px;
  padding: 22px 8px 8px;
  border-top: 1px solid rgba(255,255,255,.08);
  color: #8e99b8;
  text-align: center;
  font-size: .88rem;
}
.gradio-container .accordion {
  border-radius: 22px !important;
  overflow: hidden;
  border: 1px solid rgba(231, 195, 106, .22) !important;
}
.optional-guide {
  margin: 0 0 12px;
  color: #c9d3ea;
  font-size: .92rem;
  line-height: 1.55;
}
#validity-category {
  --input-background-fill: rgba(8, 10, 18, .85) !important;
  --input-background-fill-focus: rgba(8, 10, 18, .95) !important;
  --background-fill-primary: rgba(8, 10, 18, .85) !important;
  --background-fill-secondary: rgba(18, 22, 38, .9) !important;
  --body-text-color: #eef3ff !important;
  --input-text-color: #eef3ff !important;
}
@keyframes grid-drift { to { background-position: 56px 56px; } }
@keyframes orb-float { 50% { transform: translateY(24px) scale(1.08); } }
@keyframes gold-flow { to { background-position: 220% 0; } }
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 rgba(231,195,106,0); }
  50% { box-shadow: 0 0 22px rgba(231,195,106,.22); }
}
@keyframes licence-enter { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes licence-shine { to { transform: translateX(120%); } }
@keyframes btn-shift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes sheen-sweep { 0% { left: -40%; } 100% { left: 120%; } }
@keyframes pill-pulse { 50% { transform: scale(1.03); } }
@media (max-width: 900px) {
  .dl-features, .dl-steps { grid-template-columns: 1fr 1fr; }
  .dl-hero { min-height: auto; border-radius: 28px; }
}
@media (max-width: 640px) {
  .dl-nav { border-radius: 22px; flex-direction: column; text-align: center; }
  .dl-features, .dl-steps { grid-template-columns: 1fr; }
  .dl-hero h1 { font-size: 2.4rem; }
  #action-row { flex-direction: column !important; }
  .row { flex-direction: column !important; }
  .row > .block {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
  }
  .result-card { padding: 20px 14px; }
  .result-card th, .result-card td { padding: 12px 8px; font-size: .9rem; }
  .result-card th { width: 38%; }
}
@media (prefers-reduced-motion: reduce) {
  .dl-hero, .dl-hero::before, .dl-hero::after, .dl-kicker, .result-card,
  .result-sheen, .status-pill, button.primary, .gradio-container.gradio-container::before {
    animation: none !important;
  }
  *, *::before, *::after { transition: none !important; }
  button.primary:hover, .dl-card:hover { transform: none; }
}
"""

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
"""

with gr.Blocks(
    title="AI Powered Driving License Validation",
    theme=gr.themes.Base(primary_hue="amber", secondary_hue="teal", neutral_hue="zinc"),
    css=CUSTOM_CSS,
    head=HEAD_HTML,
    fill_width=True,
) as demo:
    gr.HTML(
        """
        <div class="dl-nav">
          <div class="dl-brand"><span class="brand-dot" aria-hidden="true"></span> Smart Card Intelligence · Indian Driving Licence</div>
          <span>Automated Visual Card Check · Not an Official Government Record Verification</span>
        </div>
        """,
        elem_id="site-nav",
    )
    gr.HTML(
        """
        <section class="dl-hero">
          <div class="dl-kicker">Intelligent Document Reader</div>
          <h1>AI Powered<br><em>Driving License Validation</em></h1>
          <p>Upload a clear, upright horizontal photograph of an Indian driving licence. The workspace inspects photo clarity, highlights printed card details, and checks whether the printed licence validity is active.</p>
          <div class="dl-pills">
            <i>Instant Photo Clarity Check</i>
            <i>Automated Card Detail Reading</i>
            <i>Active Validity Date Check</i>
          </div>
        </section>
        """,
        elem_id="site-hero",
    )
    gr.HTML(
        """
        <div class="dl-features">
          <article class="dl-card"><b>01 · Photo Quality Check</b><p>Ensures your uploaded licence photo is upright, sharp, well-lit, and free from flash glare before reading.</p></article>
          <article class="dl-card"><b>02 · Smart Detail Reading</b><p>Highlights key regions on the card and reads the licence number, holder name, birth date, relation, and address.</p></article>
          <article class="dl-card"><b>03 · Clear Status &amp; Guidance</b><p>Presents an easy-to-read validation verdict with step-by-step guidance if a clearer photo is needed.</p></article>
        </div>
        """,
        elem_id="site-features",
    )
    gr.HTML('<p class="section-label">Verification Workspace</p>', elem_id="workspace-label")

    with gr.Row():
        image_input = gr.Image(
            label="Driving licence photo (Upload upright horizontal card)",
            type="numpy",
            height=420,
            sources=["upload"],
            elem_id="upload-panel",
        )
        annotated_output = gr.Image(
            label="Highlighted card fields",
            height=420,
            interactive=False,
            elem_id="detected-panel",
        )

    with gr.Accordion("Need to upload the back side of your card or select a specific validity type?", open=False):
        gr.HTML(
            '<p class="optional-guide">Open this section if your validity dates are printed on the back of your card, or if your licence lists both Non-Transport and Transport dates and you want to check a specific one.</p>'
        )
        with gr.Row():
            reverse_input = gr.Image(
                label="Back side photo (optional)",
                type="numpy",
                sources=["upload"],
                height=240,
                elem_id="reverse-panel",
            )
            category_input = gr.Dropdown(
                choices=["Automatic", "NT", "TR"],
                value="Automatic",
                label="Validity type to check",
                info="Leave on 'Automatic' for standard cards, or pick 'NT' (Non-Transport) / 'TR' (Transport).",
                elem_id="validity-category",
            )

    with gr.Row(elem_id="action-row"):
        submit = gr.Button("Validate licence", variant="primary", scale=3)
        clear = gr.Button("Clear", variant="secondary", elem_id="clear-button", scale=1)
    results = gr.HTML(elem_id="results-panel")

    submit.click(
        fn=validate_uploaded_licence,
        inputs=[image_input, reverse_input, category_input],
        outputs=[annotated_output, results],
    )
    image_input.change(
        fn=lambda: (None, ""),
        inputs=[],
        outputs=[annotated_output, results],
        queue=False,
    )
    reverse_input.change(fn=lambda: (None, ""), inputs=[], outputs=[annotated_output, results], queue=False)
    category_input.change(fn=lambda: (None, ""), inputs=[], outputs=[annotated_output, results], queue=False)
    clear.click(
        fn=clear_licence,
        inputs=[],
        outputs=[image_input, annotated_output, results],
        queue=False,
    ).then(fn=lambda: (None, "Automatic"), inputs=[], outputs=[reverse_input, category_input], queue=False)

    if os.path.exists(SAMPLE_LICENCE):
        gr.Examples(
            examples=[[SAMPLE_LICENCE]],
            inputs=image_input,
            label="Try a sample licence",
        )

    gr.HTML(
        """
        <div class="dl-steps">
          <div class="dl-step"><strong>01 · Upload</strong><p>Drop a clear, horizontal photo of the front side of the driving licence.</p></div>
          <div class="dl-step"><strong>02 · Back Side (If Needed)</strong><p>Add the reverse side photo if your validity dates are printed on the back.</p></div>
          <div class="dl-step"><strong>03 · Review Details</strong><p>Check the highlighted card fields, user guidance banner, and extracted details table.</p></div>
          <div class="dl-step"><strong>04 · Official Check</strong><p>Visit the official Parivahan Sarathi portal or your local RTO for legal verification.</p></div>
        </div>
        <p class="dl-footer">AI Powered Driving License Validation · Automated Visual Card Reading Only · Not an Official Government Verification</p>
        """,
        elem_id="site-footer",
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        inbrowser=False,
        show_error=True,
    )