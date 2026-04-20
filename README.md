# 🌸 Pink Cycle — AI-Powered Menstrual Cycle & Ovulation Predictor

A cross-platform mobile application for menstrual cycle tracking, ovulation prediction, and reproductive health monitoring — powered by machine learning and wearable data integration.

## 👥 Team

| Name | Role |
|------|------|
| Meruert (meruertd876) |  Wearable Integration |
| Samira Mur | 2026 Updates — ML engineer Flutter & FastAPI |
| Gaukhar Samat | 2026 business model |

---

## 🎯 Project Aim

Pink Cycle predicts ovulation day and cycle phases using historical biological and wearable data. The app helps women plan pregnancies, track symptoms, set health goals, and compete with friends through a gamification system.

---

## 📱 Mobile App — Flutter

Built with Flutter for iOS and Android.

**Key Features:**
- Cycle phase tracking with visual ring indicator
- Ovulation and period day predictions
- Symptom logging
- AI assistant integration
- Wearable data display (HRV, Heart Rate, Skin Temperature)
- Gamification: goals, achievements, friend leaderboard
- Multilingual support

**Tech Stack:**
- Flutter + Riverpod (state management)
- Shared Preferences (local storage)
- Custom theming with AppColors

---

## ⚙️ Backend — FastAPI

REST API serving ML model predictions and user data.

**Endpoints include:**
- `/predict/ovulation` — ovulation day prediction
- `/predict/phase` — current cycle phase
- `/user/cycle` — cycle history

**Tech Stack:**
- Python + FastAPI
- Uvicorn
- Pydantic for data validation

---

## 🤖 Machine Learning — Google Colab

Models trained on a Kaggle dataset with feature engineering.

**Features used:**
`CycleNumber`, `LengthofCycle`, `Age`, `BMI`, `SkinTemp`, `HRV_Index`

### Models

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble model, high stability, main predictor |
| **XGBoost** | Gradient boosting, handles non-linear patterns |
| **LSTM** | Recurrent neural network for sequential cycle data |
| **Autoencoder** | Anomaly detection in cycle patterns |

### Evaluation Metrics
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² Score — primary accuracy metric

### Model Artifacts
- `ovulation_model.sav` — trained Random Forest model for inference

---

## 🚀 How to Run

### Flutter App
```bash
flutter pub get
flutter run
```

### FastAPI Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

ML Models
Open notebooks in Google Colab:
- `MenstrualCyclePrediction.ipynb`

---

## 📂 Repository Structure

```
pinkcycle/
├── lib/                  # Flutter app source
│   ├── features/         # Screens and widgets
│   ├── models/           # Data models
│   ├── repositories/     # State & business logic
│   └── app/              # Theme, routing
├── backend/              # FastAPI backend
│   ├── main.py
│   └── requirements.txt
├── ml/                   # Google Colab notebooks
│   ├── MenstrualCyclePrediction.ipynb
│   └── ovulation_model.sav
└── README.md
```
