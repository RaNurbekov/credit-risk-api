# 🏦 Credit Risk API — Full MLOps Pipeline

> **Full ML Engineering cycle for bank credit scoring:**
> Training → MLflow Tracking → FastAPI → SHAP Explainability → Drift Monitoring → Docker

🔗 **Live API:** https://credit-scoring-ml-api.onrender.com/predict

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | Logged in MLflow per experiment |
| **Algorithm** | LightGBM + Class Imbalance handling |
| **Decision threshold** | 0.15 (configurable) |
| **Dataset** | Home Credit Default Risk (Kaggle) |
| **Explainability** | SHAP TreeExplainer (Top-5 factors) |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Machine Learning** | Python, Pandas, Scikit-Learn, LightGBM |
| **Experiment Tracking** | MLflow (autolog, Model Registry, SQLite backend) |
| **Explainable AI** | SHAP (TreeExplainer, Top-5 risk factors) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Drift Monitoring** | Evidently AI (Data Drift Detection) |
| **Frontend** | Streamlit (interactive scoring dashboard) |
| **DevOps** | Docker, Git |
| **Audit Logging** | SQLite (full prediction history) |
| **Deployment** | Render |

---

## ⚙️ Architecture

```
📂 Home Credit Dataset (Kaggle)
        │
        ▼
🔬 src/train.py ──► MLflow autolog() ──► mlflow.db (SQLite)
        │                                      │
        │                               (metrics, ROC-AUC,
        │                                hyperparams, artifacts)
        ▼
🚀 api.py (FastAPI)
        │
        ├── lifespan: mlflow.lightgbm.load_model(RUN_ID)
        │            ← dynamic model loading from Registry
        │
        ├── /predict ──► SHAP TreeExplainer
        │            └──► Top-5 decision factors
        │            └──► log_request() ──► SQLite audit log
        │
        └── app.py (Streamlit UI) ──► visual scoring dashboard

📊 src/monitor_drift.py ──► Evidently AI ──► reports/data_drift_report.html
```

---

## 🔑 Key Features

### 1. MLflow Model Registry
Модель не "зашита" в код — при старте сервер **динамически загружает** нужную версию из MLflow по `RUN_ID`. Это позволяет переключаться между версиями модели без изменения кода API:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
ml_models["lgbm"] = mlflow.lightgbm.load_model(f"runs:/{RUN_ID}/model")
```

### 2. Explainable AI (SHAP)
Каждое решение по кредиту сопровождается **объяснением** — топ-5 факторов которые повлияли на результат. Это требование банковских регуляторов (BASEL III):

```json
{
  "probability_of_default": 0.73,
  "decision": "Reject",
  "explanation": [
    {"feature": "AMT_CREDIT", "impact": +0.42},
    {"feature": "DAYS_EMPLOYED", "impact": +0.31},
    {"feature": "AMT_INCOME_TOTAL", "impact": -0.18},
    {"feature": "DAYS_BIRTH", "impact": +0.15},
    {"feature": "EXT_SOURCE_2", "impact": -0.12}
  ]
}
```

### 3. Data Drift Monitoring (Evidently AI)
`src/monitor_drift.py` сравнивает референсные и текущие данные по 5 ключевым фичам модели и генерирует HTML-дашборд с алертами о дрейфе. Симулируется сценарий кризиса (рост доходов и кредитов в 3x):

```bash
python src/monitor_drift.py
# → reports/data_drift_report.html
```

### 4. Prediction Audit Log
Каждый `/predict` запрос логируется в SQLite: фичи клиента, вероятность дефолта, решение, timestamp. Полная воспроизводимость и аудируемость — обязательное требование для финансовых сервисов.

---

## 🚀 Quick Start

### 1. Prepare data
Download [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) from Kaggle.
Place `application_train.csv` and `application_test.csv` in `data/raw/`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train model + MLflow tracking
```bash
python src/train.py
```
MLflow automatically saves hyperparameters, ROC-AUC metrics and model artifacts to `mlflow.db`.

### 4. View experiments in MLflow UI
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```
Open [http://localhost:5000](http://localhost:5000), copy the best `RUN_ID` and paste it into `api.py`.

### 5. Run API
```bash
# Local
uvicorn api:app --reload

# Docker
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### 6. Run Streamlit UI
```bash
streamlit run app.py
```

### 7. Monitor Data Drift
```bash
python src/monitor_drift.py
# Open reports/data_drift_report.html in browser
```

---

## 📁 Project Structure

```
credit-risk-api/
├── src/
│   ├── train.py              # Training + MLflow autolog
│   ├── database.py           # SQLite prediction audit log
│   └── monitor_drift.py      # Evidently AI Data Drift
├── models/                   # Saved artifacts
├── notebooks/                # EDA and experiments
├── reports/                  # Evidently HTML reports
├── api.py                    # FastAPI microservice
├── app.py                    # Streamlit UI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔗 Resources

- [Home Credit Dataset on Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

## 🔗 Related Projects

Part of a Fintech ML ecosystem:

- [**fraud-detection-api**](https://github.com/RaNurbekov/fraud-detection-api) — Real-time fraud detection with Redis + A/B Testing
- [**fraud-gnn**](https://github.com/RaNurbekov/fraud-gnn) — Graph Neural Networks for fraud detection
- [**pfm-ai-assistant**](https://github.com/RaNurbekov/pfm_ai_assistant) — Personal Finance Manager with AI advisor

> 💡 **MLOps progression:** this project covers the full cycle —
> training → experiment tracking → model registry → API → monitoring.
> That's exactly what senior ML Engineers do in production fintech systems.

---

## 📫 Author

**Rashid Nurbekov** — ML Engineer | Fintech & Generative AI | Almaty, Kazakhstan 🇰🇿

[![Telegram](https://img.shields.io/badge/Telegram-@RaNurbek-2CA5E0?style=flat&logo=telegram&logoColor=white)](https://t.me/RaNurbek)
[![Email](https://img.shields.io/badge/Email-nurbekovrashidjob@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nurbekovrashidjob@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-RaNurbekov-181717?style=flat&logo=github&logoColor=white)](https://github.com/RaNurbekov)
