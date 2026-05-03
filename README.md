# Smart Waste Classifier

A deep learning project that classifies waste images into 6 categories. Built end-to-end — from training a model to serving it via an API with a simple UI on top.

---

## Screenshots

**Upload an image:**

![Upload](assets/streamlit_upload.png)

**Prediction result:**

![Result](assets/streamlit_result.png)

---

## What it does

You upload a photo of a waste item and the model tells you what category it belongs to — cardboard, glass, metal, paper, plastic, or trash. It runs as a FastAPI backend with a Streamlit frontend.

---

## Dataset

Used the TrashNet dataset from Kaggle — around 2,500 images across 6 waste categories. Split it 80/20 for training and validation.

| Class | Examples |
|---|---|
| Cardboard | Boxes, packaging |
| Glass | Bottles, jars |
| Metal | Cans, tins |
| Paper | Newspapers, sheets |
| Plastic | Bottles, containers |
| Trash | General non-recyclable waste |

---

## Model

Built on top of EfficientNetB0 pretrained on ImageNet. Kept the base frozen initially, trained a custom head, then fine-tuned the top layers in a second phase.

Custom head:
```
EfficientNetB0 (pretrained, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256) + ReLU
        ↓
Dropout(0.3)
        ↓
Dense(6) + Softmax
```

One thing worth noting — EfficientNet expects inputs scaled to -1 to 1, not 0 to 1. Using the wrong preprocessing was causing poor accuracy early on. Fixed it using `tf.keras.applications.efficientnet.preprocess_input`.

**Final validation accuracy: 87.47%**

---

## Training

Used a two-phase approach:

- **Phase 1** — Freeze the EfficientNetB0 base, train only the classification head. Learning rate: `1e-3`
- **Phase 2** — Unfreeze the top 20 layers and fine-tune with a low learning rate of `1e-5`

The reason for two phases is to avoid catastrophic forgetting — if you start fine-tuning with random weights in the head, the noisy gradients destroy the pretrained weights.

Used EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks throughout.

| Metric | Value |
|---|---|
| Training accuracy | ~98% |
| Validation accuracy | 87.47% |
| Best epoch | 7 |

---

## Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model training |
| FastAPI | Serving predictions via REST API |
| Streamlit | Simple UI for uploading images and seeing results |
| DVC | Versioning the dataset and model |
| GitHub Actions | Runs basic checks on every push |

---

## Running it locally

```bash
git clone https://github.com/UchiaObito004/smart-waste-classifier.git
cd smart-waste-classifier

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the API:
```bash
uvicorn api:app --reload
```

Start the UI in a separate terminal:
```bash
streamlit run app.py
```

Test a prediction from terminal:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/your/image.jpg"
```

Example response:
```json
{
  "predicted_class": "plastic",
  "confidence": "91.16%",
  "all_probabilities": {
    "cardboard": "0.19%",
    "glass": "6.82%",
    "metal": "0.08%",
    "paper": "0.52%",
    "plastic": "91.16%",
    "trash": "1.22%"
  }
}
```

---

## Project structure

```
├── api.py                  ← FastAPI inference endpoint
├── app.py                  ← Streamlit UI
├── code.ipynb              ← Training notebook
├── data.dvc                ← DVC tracking for dataset
├── assets/
│   ├── streamlit_upload.png
│   └── streamlit_result.png
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml          ← GitHub Actions CI pipeline
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Classify a waste image |

---

## Results

Validation accuracy of 87.47% on ~500 unseen images. Training accuracy reached ~98% which shows some overfitting — expected for a 2,500 image dataset. The model still generalizes well to real-world images as shown in the screenshots above.

---

## What I'd improve with more time

- Deploy it to Streamlit Cloud or HuggingFace Spaces
- Add Docker support for easier deployment
- Collect more training data, especially for the trash category which is the hardest to classify
- Add confidence threshold — if model is less than 60% confident, say "unclear"

---

## Author

Parivat — [GitHub](https://github.com/UchiaObito004)
