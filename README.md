# Clinical Decision Support System for Heart Failure Mortality Risk Prediction

This project predicts whether a patient is at risk of a heart failure death event using the Kaggle Heart Failure Clinical Data dataset.

Dataset: [Heart Failure Prediction on Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data)

## Project Goal

Build a clinical decision support model that estimates mortality risk from patient clinical records.

## Final Method

The final model is a **SMOTE Tuned Stacking Ensemble**.

Base learners:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine

Meta-model:

- Random Forest

The base learners are tuned with `RandomizedSearchCV`, using F1 score as the main optimization target while tracking recall and precision. SMOTE is applied inside the model pipelines so oversampling happens only during training and cross-validation folds. The stacking model uses `passthrough=True`, so the Random Forest meta-model can learn from both base-model probabilities and the original patient features.

The notebook also tunes the decision threshold using out-of-fold training predictions. The selected threshold maximizes F1, with recall and precision used as tie-breakers. Results are displayed directly in the notebook.

## Dataset

Expected CSV file:

```text
heart_failure_prediction.csv
```

The notebook also accepts the original Kaggle filename:

```text
heart_failure_clinical_records_dataset.csv
```

Target column:

```text
DEATH_EVENT
```

The `time` feature is excluded by default because it represents follow-up duration and may cause target leakage for baseline clinical decision support.

The notebook uses a stratified 70/30 train-test split, so 30% of the dataset is kept as the final test set.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Project

Open and run:

```text
Heart_Failure_Mortality_Risk_Prediction.ipynb
```

The notebook displays:

- exploratory data analysis
- base learner tuning results
- final stacking ensemble cross-validation results
- optimized threshold table for F1, recall, and precision
- SMOTE tuned stacking ensemble metrics
- ROC curve
- confusion matrix
- final notebook report
- new patient prediction example

## Backend API

Train and save the model artifact:

```powershell
python backend\train_model.py
```

Saved model:

```text
backend/model/heart_failure_model.joblib
```

Run the API:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Prediction endpoint:

```text
POST /predict
```

Example request body:

```json
{
  "age": 65,
  "anaemia": 0,
  "creatinine_phosphokinase": 250,
  "diabetes": 1,
  "ejection_fraction": 35,
  "high_blood_pressure": 1,
  "platelets": 263000,
  "serum_creatinine": 1.3,
  "serum_sodium": 136,
  "sex": 1,
  "smoking": 0
}
```

Other endpoints:

- `GET /health`
- `GET /model-info`

## Railway Deployment Notes

This project includes a root `main.py` file and `railway.json` config for Railway.

Railway start command:

```text
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
```

`main.py` re-exports the FastAPI app from `backend/api.py`, so Railway can import `main:app` correctly. Keep the Railway root directory as `/` so it can see `main.py`, `railway.json`, `requirements.txt`, and the bundled model artifact.

After deploying to Railway, generate a public domain and test these URLs:

```text
https://your-railway-domain.up.railway.app/health
https://your-railway-domain.up.railway.app/model-info
https://your-railway-domain.up.railway.app/docs
```

## Vercel Deployment Notes

This project includes a root `app.py` file:

```text
app.py
```

It re-exports the FastAPI app from `backend/api.py`, which gives Vercel a standard Python FastAPI entrypoint.

After deploying to Vercel, test these URLs:

```text
https://your-vercel-domain.vercel.app/health
https://your-vercel-domain.vercel.app/model-info
https://your-vercel-domain.vercel.app/docs
```

If `/health` returns `model_loaded: false`, the model artifact was not included or failed to load. Make sure this file exists in the deployed repo:

```text
backend/model/heart_failure_model.joblib
```
