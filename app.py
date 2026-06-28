import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

# load both models at startup
print("Loading models...")
efficientnet_model = load_model(
    "efficientnet_model.keras",
    custom_objects={"preprocess_input": efficientnet_preprocess}
)
resnet_model = load_model(
    "resnet_model.keras",
    custom_objects={"preprocess_input": resnet_preprocess}
)
print("Models loaded!")

MODELS = {
    "ResNet50 (90.7% accuracy)":       resnet_model,
    "EfficientNetB0 (87.7% accuracy)": efficientnet_model
}

def predict(image, model_choice):
    try:
        # preprocess
        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        model       = MODELS[model_choice]
        predictions = model.predict(img_array, verbose=0)[0]

        # top 3
        top3_indices = np.argsort(predictions)[::-1][:3]

        result = f"{'─'*40}\n"
        result += f"Predicted: {CLASS_NAMES[np.argmax(predictions)]}\n"
        result += f"Confidence: {float(np.max(predictions))*100:.1f}%\n"
        result += f"{'─'*40}\n\nTop 3 Predictions:\n"
        for i, idx in enumerate(top3_indices, 1):
            result += f"  {i}. {CLASS_NAMES[idx]:<35} {float(predictions[idx])*100:.1f}%\n"

        result += f"\n⚠️ Research tool only. Not a medical diagnosis."
        return result

    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Skin Cancer Classifier") as demo:
    gr.Markdown("""
    # 🔬 Skin Cancer Classifier
    Upload a dermoscopic skin lesion image for classification across 9 diagnostic categories.
    
    | Model | Accuracy | F1 Score |
    |-------|----------|----------|
    | ResNet50 | 90.7% | 93.5% |
    | EfficientNetB0 | 87.7% | 87.8% |
    
    > ⚠️ **Research tool only. Not a substitute for medical diagnosis.**
    """)

    with gr.Row():
        with gr.Column():
            image_input  = gr.Image(type="pil", label="Upload skin lesion image")
            model_select = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="ResNet50 (90.7% accuracy)",
                label="Select model"
            )
            predict_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Classification Result", lines=12)

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_select],
        outputs=output
    )

    gr.Markdown("""
    ### Diagnostic Categories
    Actinic Keratosis · Basal Cell Carcinoma · Dermatofibroma · Melanoma · Nevus · 
    Pigmented Benign Keratosis · Seborrheic Keratosis · Squamous Cell Carcinoma · Vascular Lesion
    """)

demo.launch()
