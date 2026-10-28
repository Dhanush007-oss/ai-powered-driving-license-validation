"""
Production Gradio Web Application for Indian Driving Licence Validation.
Ensures instant server startup and port binding to satisfy cloud deployment health checks.
"""

import os
import gradio as gr

# Import the core processing pipeline
from inference import process_driving_license, DISPLAY_FIELDS

def validate_licence_ui(front_image, back_image, category_selection):
    """Gradio wrapper for driving licence validation pipeline."""
    if front_image is None:
        return (
            "NOT VALID | Error: No image provided. Please upload the front side of your driving licence.",
            None,
            {"error": "No image uploaded"}
        )
    
    payload = {
        "front": front_image,
        "back": back_image,
        "category": category_selection if category_selection else "Automatic"
    }
    
    try:
        fields, annotated_image, assessment = process_driving_license(payload, save_json=False)
        status_msg = assessment.get("message", "Processing complete.")
        return status_msg, annotated_image, fields
    except Exception as e:
        return f"NOT VALID | Processing Error: {str(e)}", front_image, {"error": str(e)}

# Build the dark glassmorphism Gradio interface
with gr.Blocks(title="AI-Powered Indian Driving Licence Validator") as demo:
    gr.Markdown("# 🪪 Indian Driving Licence Validation Engine")
    gr.Markdown("Upload your driving licence photo to verify image quality, number format, required fields, and category validity.")
    
    with gr.Row():
        with gr.Column(scale=1):
            front_input = gr.Image(type="numpy", label="Licence Front Side (Required)")
            back_input = gr.Image(type="numpy", label="Licence Back Side (Optional)")
            category_dropdown = gr.Dropdown(
                choices=["Automatic", "NT", "TR"],
                value="Automatic",
                label="Validity Screening Category"
            )
            validate_btn = gr.Button("Validate Licence", variant="primary")
            
        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Validation Status & Guidance", lines=3)
            annotated_output = gr.Image(label="Detected Field Bounding Boxes")
    
    with gr.Row():
        json_output = gr.JSON(label="Extracted Card Fields & Telemetry")

    validate_btn.click(
        fn=validate_licence_ui,
        inputs=[front_input, back_input, category_dropdown],
        outputs=[status_output, annotated_output, json_output]
    )

if __name__ == "__main__":
    # Dynamically grab the port assigned by the hosting platform (Render/HuggingFace/Railway), defaulting to 7860 locally
    port = int(os.environ.get("PORT", 7860))
    
    print(f"Binding Gradio server to 0.0.0.0:{port}...")
    
    # CRITICAL: server_name="0.0.0.0" and server_port=port ensure the platform port scanner detects an open port immediately
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=False,
        show_error=True
    )