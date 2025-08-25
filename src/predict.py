import pandas as pd
from joblib import load

THRESHOLD = 0.33  # tuned threshold

bundle = load('../models/injury_risk_rf_pipeline.joblib')
pipe = bundle['pipeline']
feature_names = bundle['feature_names']

# example input (edit values)
row = {
  'Age': 28, 'Gender': 1, 'Height_cm': 175.0, 'Weight_kg': 72.0, 'BMI': 23.5,
  'Training_Frequency': 4, 'Training_Duration': 60, 'Warmup_Time': 8, 'Sleep_Hours': 6.5,
  'Flexibility_Score': 0.55, 'Muscle_Asymmetry': 0.22, 'Recovery_Time': 24,
  'Injury_History': 1, 'Stress_Level': 6, 'Training_Intensity': 6.8
}

X = pd.DataFrame([row], columns=feature_names)
proba = pipe.predict_proba(X)[:, 1][0]
label = int(proba >= THRESHOLD)

print(f"Probability: {proba:.3f}")
print(f"Predicted label (threshold={THRESHOLD}): {label}")
