from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = FastAPI(title="Skin Cancer Classifier API")

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

# load both models at startup
models = {}


from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

@app.on_event("startup")
def load_models():
    global models
    custom_objects = {"preprocess_input": resnet_preprocess}
    for name, path in [
        ("EfficientNetB0", r"C:\Users\anand\skin-cancer-classifier\models\efficientnet_model.keras"),
        ("ResNet50",       r"C:\Users\anand\skin-cancer-classifier\models\resnet_model.keras")
    ]:
        if os.path.exists(path):
            models[name] = load_model(path, custom_objects=custom_objects)
            print(f"{name} loaded successfully")
        else:
            print(f"WARNING: {name} not found at {path}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys())
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "ResNet50"
):
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available. Choose from: {list(models.keys())}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    img_array   = preprocess_image(image_bytes)

    model       = models[model_name]
    predictions = model.predict(img_array, verbose=0)[0]

    # top 3 predictions
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3 = [
        {
            "class":      CLASS_NAMES[i],
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top3_indices
    ]

    return {
        "model_used":       model_name,
        "predicted_class":  CLASS_NAMES[np.argmax(predictions)],
        "confidence":       round(float(np.max(predictions)) * 100, 2),
        "top_3_predictions": top3
    }
