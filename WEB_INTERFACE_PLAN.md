# Web Interface Requirement and Planning

## Web Interface Goal

Build a user-friendly web application where users can enter patient clinical data, submit it to the backend prediction API, and see the predicted heart failure mortality risk.

## Main Users

- Doctor or clinical staff
- Medical student or researcher
- Project evaluator or lecturer

## Core Features

- Patient input form
- Input validation
- Submit button to call backend `/predict`
- Display prediction result
- Display mortality risk probability
- Display predicted class
- Display risk level: low, medium, high
- Show short explanation of what each result means
- Reset form button
- Loading state while prediction is running
- Error message if backend is unavailable or input is invalid

## Input Fields

The form should include:

```text
age
anaemia
creatinine_phosphokinase
diabetes
ejection_fraction
high_blood_pressure
platelets
serum_creatinine
serum_sodium
sex
smoking
```

Suggested field types:

- `age`: number input
- `anaemia`: dropdown or radio, Yes/No
- `creatinine_phosphokinase`: number input
- `diabetes`: dropdown or radio, Yes/No
- `ejection_fraction`: number input
- `high_blood_pressure`: dropdown or radio, Yes/No
- `platelets`: number input
- `serum_creatinine`: number input
- `serum_sodium`: number input
- `sex`: dropdown or radio
- `smoking`: dropdown or radio, Yes/No

## Result Display

After prediction, show:

```text
Mortality Risk Probability: 0.3459
Predicted Death Event: No
Risk Level: Low
Threshold Used: 0.49
```

User-friendly output mapping:

- `predicted_death_event = 0`: Death event not predicted
- `predicted_death_event = 1`: Death event predicted
- `risk_level = low`: green indicator
- `risk_level = medium`: yellow or orange indicator
- `risk_level = high`: red indicator

## Pages

For a simple project, one page is enough:

```text
Prediction Page
```

Optional pages:

- About Model
- Dataset Information
- API Status

## Recommended Tech Stack

Frontend:

- React or Next.js
- Tailwind CSS
- Axios or Fetch API

Backend:

- FastAPI
- Endpoint: `POST /predict`
- Saved machine learning model loaded from `backend/model/heart_failure_model.joblib`

## System Architecture

```text
User Input Form
      ↓
Frontend fetch/axios request
      ↓
FastAPI /predict endpoint
      ↓
Saved ML model
      ↓
Prediction response
      ↓
Result card on frontend
```

## API Request Format

Frontend sends this JSON to `POST /predict`:

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

## API Response Format

Backend returns:

```json
{
  "mortality_risk_probability": 0.3459,
  "predicted_death_event": 0,
  "threshold": 0.49,
  "risk_level": "low"
}
```

## Development Plan

1. Design the page layout.
2. Build the patient input form.
3. Add field validation.
4. Connect the form to FastAPI `/predict`.
5. Display the prediction result.
6. Add loading and error states.
7. Improve UI styling.
8. Test with sample patient data.
9. Prepare the final demo.

## Suggested UI Layout

Top section:

- Project title
- Short subtitle

Main section:

- Left side: patient input form
- Right side: prediction result card

Bottom section:

- Model information
- Disclaimer

## Important Disclaimer

Because this is a medical prediction project, the interface should show:

```text
This prediction is for educational and research purposes only. It should not be used as a final clinical decision without professional medical evaluation.
```

## Minimum Viable Product

For the first version, build only:

- One prediction page
- Patient input form
- Result card
- Backend API connection
- Basic validation
- Loading state
- Error state

This is enough for a complete working demo.
