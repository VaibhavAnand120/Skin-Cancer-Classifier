import gradio as gr
import requests
import json
from PIL import Image
import io

API_URL = "http://localhost:8000"

MODELS = ["ResNet50", "EfficientNetB0"]

def predict(image, model_name):
    try:
        # convert PIL image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = requests.post(
            f"{API_URL}/predict",
            params={"model_name": model_name},
            files={"file": ("image.jpg", img_bytes, "image/jpeg")}
        )

        if response.status_code != 200:
            return f"Error: {response.json().get('detail', 'Unknown error')}", ""

        result = response.json()

        # format top 3
        top3_text = "Top 3 Predictions:\n"
        for i, pred in enumerate(result["top_3_predictions"], 1):
            top3_text += f"  {i}. {pred['class'].title()} — {pred['confidence']}%\n"

        summary = (
            f"Model: {result['model_used']}\n"
            f"Predicted class: {result['predicted_class'].title()}\n"
            f"Confidence: {result['confidence']}%\n\n"
            f"{top3_text}"
        )

        return summary

    except Exception as e:
        return f"Could not connect to API: {str(e)}\nMake sure FastAPI is running."

with gr.Blocks(title="Skin Cancer Classifier") as demo:
    gr.Markdown("""
    # 🔬 Skin Cancer Classifier
    Upload a dermoscopic skin lesion image to classify it across 9 diagnostic categories.
    
    **Best model: ResNet50 — 90.7% accuracy | F1: 93.5%**
    
    ⚠️ *This is a research tool only. Not a substitute for medical diagnosis.*
    """)

    with gr.Row():
        with gr.Column():
            image_input  = gr.Image(type="pil", label="Upload skin lesion image")
            model_select = gr.Dropdown(choices=MODELS, value="ResNet50", label="Select model")
            predict_btn  = gr.Button("Classify", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Classification Result", lines=12)

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_select],
        outputs=output
    )

    gr.Markdown("""
    ### Classes
    Actinic Keratosis · Basal Cell Carcinoma · Dermatofibroma · Melanoma · Nevus · 
    Pigmented Benign Keratosis · Seborrheic Keratosis · Squamous Cell Carcinoma · Vascular Lesion
    """)

demo.launch()
