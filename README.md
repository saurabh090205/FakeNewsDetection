# Veritas AI — Fake News Detection & Explainability Platform

> **"Understand the truth behind every headline."**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-blue.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](https://www.typescriptlang.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org)
[![LIME XAI](https://img.shields.io/badge/LIME-Explainable_AI-cyan.svg)](https://github.com/marcotcr/lime)

Veritas AI is an advanced deep learning and explainability platform that analyzes news text to classify articles as **REAL** or **FAKE** while providing transparent, word-level local surrogate attribution using **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## Key Features

1. **Neural Text Classification Pipeline**:
   - Conservative text normalization (HTML strip, uniform `URLTOKEN` replacement, Reuters dateline suppression to avoid shortcut learning).
   - Fixed 500-token sequence padding and tokenization using a fitted vocabulary of 30,000 words.
   - Comparative evaluation across five neural architectures: **Dense Neural Baseline**, **SimpleRNN**, **LSTM (Long Short-Term Memory)**, **GRU (Gated Recurrent Unit)**, and **DistilBERT Transformer**.

2. **Transparent LIME Explainability Engine**:
   - 160 local surrogate perturbations per analysis with reproducible random seed.
   - Interactive article text display with dynamic word-level highlighting:
     - **Cyan/Green**: Words supporting the predicted classification.
     - **Rose/Red**: Words opposing the predicted classification.
   - Hover tooltips showing exact contribution weights (surrogate ridge regression coefficients).
   - Horizontal feature contribution bar chart with directional weights.
   - Plain-English narrative interpretation of key contributing factors.

3. **Production-Ready Results Dashboard**:
   - Classification verdict badge: `PREDICTED REAL`, `PREDICTED FAKE`, or `PREDICTION UNCERTAIN` (enforcing a 20% uncertainty margin).
   - Calibrated probability distribution bar (Real vs. Fake percentage split).
   - Article analysis summary (word count, character count, tokenization preview, and timestamps).
   - Multi-model comparative prediction mode.
   - Session-based analysis history with one-click reload.

4. **Technical Transparency & Evaluation**:
   - Empirical test metrics loaded exclusively from untouched held-out test splits (70/15/15 train/val/test).
   - Confusion matrix breakdown (True Positive, True Negative, False Positive, False Negative).
   - Training and validation loss/accuracy curves.
   - Diagnostic overfit and underfit experiment demonstrations.

---

## Technology Stack

- **Backend**: FastAPI (Python 3.11), Uvicorn, Pydantic v2
- **Deep Learning**: TensorFlow 2.x / Keras, PyTorch, Hugging Face Transformers
- **Explainability (XAI)**: LIME (`lime.lime_text.LimeTextExplainer`)
- **Data Science**: Scikit-Learn, Pandas, NumPy
- **Frontend**: React 18, TypeScript, Vite, Recharts, Lucide Icons
- **Styling**: Tailored modern AI analytics design system (Vanilla CSS with dark navy/charcoal palette)

---

## Directory Structure

```text
FakeNewsDetection/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py             # FastAPI REST endpoints & LIME caching
│   │   ├── core/
│   │   │   └── config.py             # Configuration loader & filesystem paths
│   │   ├── ml/
│   │   │   └── predictor.py          # Model loading, inference, & validation
│   │   ├── preprocessing/
│   │   │   └── text.py               # Conservative text normalization
│   │   ├── services/
│   │   │   └── artifacts.py          # Model registry & inventory
│   │   ├── main.py                   # FastAPI app, CORS, & security middleware
│   │   └── schemas.py                # Pydantic data schemas
│   ├── config.yaml                   # Core hyperparameter & CORS config
│   ├── requirements.txt              # Production backend dependencies
│   └── tests/                        # Pytest suite (17 tests)
│       ├── test_api.py
│       ├── test_data_pipeline.py
│       ├── test_prediction.py
│       ├── test_preprocessing.py
│       └── test_veritas_ai.py
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts                # Typed frontend API client
│   │   ├── types/
│   │   │   └── api.ts                # TypeScript interfaces
│   │   ├── App.tsx                   # Main React platform UI
│   │   ├── main.tsx                  # React DOM root
│   │   └── styles.css                # Veritas AI design tokens & styles
│   ├── index.html                    # HTML entry point with metadata
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── raw/                          # Raw True.csv and Fake.csv datasets
│   └── processed/                    # Cleaned news.csv & label_mapping.json
├── models/                           # Trained model checkpoints & tokenizers
│   ├── dense/
│   └── lstm/
├── results/                          # Evaluation metrics & confusion matrices
│   └── metrics/
├── training/
│   ├── data_pipeline.py              # Ingestion, cleaning, & stratified split
│   ├── train_text_model.py           # Model training & test evaluation
│   ├── train_experiments.py          # Overfit/underfit diagnostic curves
│   └── settings.py
├── .env.example
├── .gitignore
└── README.md
```

---

## How It Works

### 1. Linguistic Classification Pipeline
```text
Raw News Article ──> Conservative Cleaning ──> Tokenization & Sequence Padding
                            │                               │
                            ▼                               ▼
                 HTML Strip / URLTOKEN           500-Token Tensor (Post-padded)
                                                            │
                                                            ▼
                                                Neural Network Inference
                                                            │
                                                            ▼
                                              Sigmoid Probability Output
                                                            │
                                                            ▼
                                                [REAL %] vs [FAKE %]
```

### 2. LIME Local Surrogate Explanation
```text
Input Article ──> Perturb Text (160 samples) ──> Run Model on Perturbations
                         │                                  │
                         ▼                                  ▼
               Random Word Deletions                 Class Probabilities
                                                            │
                                                            ▼
                                            Fit Local Ridge Regression
                                                            │
                                                            ▼
                                                Feature Contribution Weights
                                                (Supports vs. Opposes)
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- Git

### 1. Clone Repository
```bash
git clone https://github.com/saurabh090205/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2. Backend Setup
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Dataset Preparation & Model Training
To prepare the dataset and train the baseline neural models:
```bash
# Prepare dataset and train the Dense baseline
python -c "from training.data_pipeline import prepare_dataset; prepare_dataset()"
python training/train_dense.py

# Optional: Train the LSTM recurrent model
python training/train_lstm.py

# Optional: Generate overfit/underfit diagnostic curves
python training/train_experiments.py
```

### 4. Run the Backend API Server
```bash
# From the project root:
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
The FastAPI documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 5. Frontend Setup & Startup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open your browser at [http://localhost:5173](http://localhost:5173).

---

## API Endpoints Reference

### `GET /api/health`
Returns system status and model inventory.
```json
{
  "status": "ok",
  "platform": "Veritas AI — Fake News Detection & Explainability Platform",
  "available_models": 2,
  "models": [
    { "name": "dense", "available": true, "parameter_count": 3856641, "description": "DENSE neural text classifier" },
    { "name": "lstm", "available": true, "parameter_count": 3889601, "description": "LSTM neural text classifier" }
  ]
}
```

### `POST /api/predict`
Evaluates a headline and article using a selected model.
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Federal Reserve Holds Interest Rates Steady",
    "article": "The central bank voted unanimously to maintain its benchmark federal funds rate during its scheduled policy meeting on Wednesday. Officials noted that consumer prices have moderated toward target thresholds...",
    "model": "dense"
  }'
```
Response:
```json
{
  "prediction": "REAL",
  "predicted_label_display": "Predicted Real",
  "confidence": 0.7685,
  "real_probability": 0.7685,
  "fake_probability": 0.2315,
  "model": "dense",
  "article_words": 57,
  "article_chars": 440,
  "timestamp": "2026-10-26T05:27:12.911434+00:00",
  "explanation_note": "Model-based classification from learned text patterns; not independent fact verification."
}
```

### `POST /api/explain`
Generates LIME feature attribution and plain-English narrative for the prediction.
```bash
curl -X POST http://127.0.0.1:8000/api/explain \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Federal Reserve Holds Interest Rates Steady",
    "article": "The central bank voted unanimously to maintain its benchmark federal funds rate during its scheduled policy meeting on Wednesday...",
    "model": "dense"
  }'
```
Response includes `words`, `positive_words`, `negative_words`, `plain_english_summary`, `methodology`, and `cached`.

---

## Running Automated Tests

Run the full backend test suite (unit tests, validation, data pipeline, and LIME explainability):
```bash
# Windows PowerShell:
$env:PYTHONPATH="backend;."; pytest backend/tests

# Linux/macOS:
PYTHONPATH="backend:." pytest backend/tests
```

Run frontend production build verification:
```bash
cd frontend
npm run build
```

---

## Responsible AI & Ethical Use Disclaimer

Veritas AI is designed for academic demonstration and research into natural language processing and explainability. 

> [!IMPORTANT]
> **Probabilistic Classification vs. Objective Factuality**
> - A **REAL** prediction indicates that the submitted text shares stylistic, syntactic, and structural traits with verified news articles in the training set. It does **not** prove that the underlying claims are factually authentic.
> - A **FAKE** prediction indicates high resemblance to sensationalist, hyperbolic, or conspiratorial writing styles in the training distribution.
> - Veritas AI does **not** crawl the live web, access real-time databases, or verify external facts. Users should always consult primary sources and accredited journalistic reporting before drawing conclusions.

---

## License

This project is released under the MIT License for academic and research purposes.
