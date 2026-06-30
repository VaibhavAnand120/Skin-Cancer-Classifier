# Skin Cancer Classification on ISIC Dataset

Comparing 5 deep learning models for multi-class skin lesion classification using the ISIC dataset, progressively improving accuracy through augmentation, class balancing, and transfer learning. Best model deployed live with a FastAPI backend and Gradio frontend.

**Live demo:** https://huggingface.co/spaces/VaibhavAnand120/Skin-Cancer-Classifier

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CNN without augmentation | 39.6% | 38.9% | 39.6% | 34.6% |
| CNN with augmentation | 51.2% | 51.4% | 51.2% | 48.9% |
| CNN with class balance (Augmentor) | 61.4% | 64.9% | 61.4% | 60.5% |
| EfficientNetB0 | 87.7% | 88.9% | 87.7% | 87.8% |
| ResNet50 (BEST) | 90.7% | 90.7% | 90.7% | 93.5% |

---

## Key Findings

- Augmentation alone improved accuracy by **+11.6%** over baseline CNN
- Class balancing with Augmentor added a further **+10.2%**
- Transfer learning (EfficientNet/ResNet) dramatically outperformed custom CNNs
- ResNet50 achieved the best overall performance at **90.7% accuracy**

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Models | TensorFlow, Keras, CNN, EfficientNetB0, ResNet50 |
| Augmentation | Augmentor, Keras ImageDataGenerator |
| API | FastAPI, Uvicorn |
| Frontend | Gradio |
| Deployment | HuggingFace Spaces |
| Platform | Kaggle Notebooks (GPU) |

---

## Project Structure

```
skin-cancer-classifier/
├── api/
│   └── main.py          # FastAPI backend — image upload + prediction endpoint
├── app.py               # Gradio frontend — image upload UI
├── requirements.txt
└── README.md

Note: Model files (efficientnet_model.keras, resnet_model.keras) are hosted
on HuggingFace Spaces due to GitHub's 100MB file size limit.
```

---

## Models Compared

**1. CNN without augmentation** — baseline model trained on raw imbalanced data

**2. CNN with augmentation** — added random flips, rotations, zoom to reduce overfitting

**3. CNN with class balance (Augmentor)** — oversampled minority classes to 1000 samples each, addressing severe dataset imbalance

**4. EfficientNetB0** — pretrained on ImageNet, fine-tuned on ISIC using transfer learning

**5. ResNet50** — pretrained on ImageNet, fine-tuned on ISIC, best performing model

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/VaibhavAnand120/Skin-Cancer-Classifier.git
cd Skin-Cancer-Classifier
```

**2. Download model files from HuggingFace**
```
https://huggingface.co/spaces/VaibhavAnand120/Skin-Cancer-Classifier/tree/main
```
Place them in a `models/` folder.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the API**
```bash
python -m uvicorn api.main:app --reload
```

**5. Launch the Gradio UI**
```bash
python app.py
```
Opens at `http://localhost:7860`

---

## API Endpoint

```
POST /predict
- Input:  image file + model_name (ResNet50 or EfficientNetB0)
- Output: predicted class, confidence score, top 3 predictions
```

Sample response:
```json
{
  "model_used": "ResNet50",
  "predicted_class": "melanoma",
  "confidence": 94.3,
  "top_3_predictions": [
    {"class": "melanoma", "confidence": 94.3},
    {"class": "nevus", "confidence": 4.1},
    {"class": "pigmented benign keratosis", "confidence": 1.2}
  ]
}
```

---

## Dataset

[ISIC 2019 Skin Lesion Dataset](https://www.isic-archive.com/) — 9 diagnostic categories of dermoscopic images.

Available on Kaggle: [Skin Cancer ISIC 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

### Classes
Actinic Keratosis · Basal Cell Carcinoma · Dermatofibroma · Melanoma · Nevus · Pigmented Benign Keratosis · Seborrheic Keratosis · Squamous Cell Carcinoma · Vascular Lesion

---

> ⚠️ This is a research tool only and is not a substitute for professional medical diagnosis.
