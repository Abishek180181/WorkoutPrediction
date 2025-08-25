# app.py
# Simple Streamlit UI for Workout Injury Risk Prediction
# Run: streamlit run app.py

import os
import pandas as pd
import streamlit as st
from joblib import load

THRESHOLD = 0.33  # tuned threshold you found earlier
MODEL_BUNDLE_PATH = os.path.join("models", "injury_risk_rf_pipeline.joblib")

st.set_page_config(page_title="Workout Injury Risk Predictor", page_icon="💪", layout="wide")
st.title("💪 Workout Injury Risk Predictor")
st.write("Enter your details to estimate your injury risk, see a simple reason, and get quick tips.")

@st.cache_resource(show_spinner=True)
def load_bundle(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found at `{path}`. Train and save your pipeline first.")
        st.stop()
    return load(path)

bundle = load_bundle(MODEL_BUNDLE_PATH)
pipe = bundle["pipeline"]
feature_names = bundle["feature_names"]

# --- Input form ---
with st.form("risk_form"):
    st.subheader("Your details")

    c1, c2, c3 = st.columns(3)

    Age = c1.number_input("Age", min_value=16, max_value=80, value=28, step=1)
    Gender = c2.selectbox("Gender (0=female, 1=male)", [0, 1], index=1)
    Height_cm = c3.number_input("Height (cm)", min_value=130.0, max_value=210.0, value=175.0, step=0.1)
    Weight_kg = c1.number_input("Weight (kg)", min_value=35.0, max_value=160.0, value=72.0, step=0.1)

    # BMI auto-calc
    height_m = max(Height_cm / 100.0, 0.5)
    BMI = round(Weight_kg / (height_m * height_m), 1)
    c2.metric("BMI (auto-calculated)", f"{BMI:.1f}")

    Training_Frequency = c3.number_input("Training Frequency (sessions/week)", min_value=0, max_value=14, value=4, step=1)
    Training_Duration = c1.number_input("Training Duration (minutes/session)", min_value=0, max_value=240, value=60, step=5)
    Warmup_Time = c2.number_input("Warmup Time (minutes)", min_value=0, max_value=60, value=8, step=1)
    Sleep_Hours = c3.number_input("Sleep Hours (per night)", min_value=0.0, max_value=12.0, value=6.5, step=0.1)

    Flexibility_Score = c1.number_input("Flexibility Score (0–1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    Muscle_Asymmetry = c2.number_input("Muscle Asymmetry (0–1)", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
    Recovery_Time = c3.number_input("Recovery Time (hours between hard sessions)", min_value=0, max_value=120, value=24, step=1)

    Injury_History = c1.number_input("Injury History (0=none, higher=more)", min_value=0, max_value=10, value=1, step=1)
    Stress_Level = c2.number_input("Stress Level (0–10)", min_value=0, max_value=10, value=6, step=1)
    Training_Intensity = c3.number_input("Training Intensity (1–10)", min_value=1.0, max_value=10.0, value=6.8, step=0.1)

    submitted = st.form_submit_button("Predict")

def simple_reasons(row):
    """Return short, human reasons based on common-sense thresholds."""
    reasons = []

    if row["Injury_History"] > 0:
        reasons.append("You have past injuries, which increases your risk.")
    if row["Training_Intensity"] >= 7:
        reasons.append("Your training intensity is high.")
    if row["Training_Frequency"] >= 5:
        reasons.append("You train many times per week.")
    if row["Sleep_Hours"] < 7:
        reasons.append("You are not sleeping enough.")
    if row["Warmup_Time"] < 8:
        reasons.append("Your warmup time is short.")
    if row["Muscle_Asymmetry"] > 0.25:
        reasons.append("You have noticeable muscle asymmetry.")
    if row["Recovery_Time"] < 24:
        reasons.append("Your recovery time between hard sessions is short.")
    if row["Stress_Level"] >= 7:
        reasons.append("Your stress level is high.")
    if row["BMI"] >= 30:
        reasons.append("Your BMI is high, which can add joint stress.")
    if row["BMI"] < 18.5:
        reasons.append("Your BMI is low; inadequate nutrition can increase injury risk.")

    if not reasons:
        reasons.append("Your inputs look balanced overall.")

    # Keep it concise: show up to 3 reasons
    return reasons[:3]

def simple_actions(label, row):
    """Return practical actions based on what’s off."""
    actions = []
    if label == 1:
        if row["Injury_History"] > 0:
            actions.append("Check in with a physio and address past injury areas.")
        if row["Training_Intensity"] >= 7 or row["Training_Frequency"] >= 5:
            actions.append("Reduce training intensity/volume for 1–2 weeks and reassess.")
        if row["Sleep_Hours"] < 7:
            actions.append("Aim for at least 7 hours of sleep.")
        if row["Warmup_Time"] < 8:
            actions.append("Increase warmup to 8–12 minutes with mobility and activation.")
        if row["Muscle_Asymmetry"] > 0.25:
            actions.append("Add unilateral strength work to correct imbalances.")
        if row["Recovery_Time"] < 24:
            actions.append("Increase recovery time to 24–36 hours between hard sessions.")
        if row["Stress_Level"] >= 7:
            actions.append("Add stress management: breathwork, walks, or a lighter week.")
        if not actions:
            actions.append("Take a deload week: reduce volume by ~30% and monitor how you feel.")
    else:
        actions.append("Maintain your routine and keep tracking sleep, warmup, and recovery.")
        actions.append("Progress gradually (no more than ~10% weekly load increase).")
        actions.append("Add a recovery week every 4–6 weeks.")

    # Keep it concise: show up to 4 actions
    return actions[:4]

if submitted:
    # Build the row with BMI auto-filled
    row = {
        "Age": Age, "Gender": Gender, "Height_cm": Height_cm, "Weight_kg": Weight_kg, "BMI": BMI,
        "Training_Frequency": Training_Frequency, "Training_Duration": Training_Duration, "Warmup_Time": Warmup_Time,
        "Sleep_Hours": Sleep_Hours, "Flexibility_Score": Flexibility_Score, "Muscle_Asymmetry": Muscle_Asymmetry,
        "Recovery_Time": Recovery_Time, "Injury_History": Injury_History, "Stress_Level": Stress_Level,
        "Training_Intensity": Training_Intensity
    }

    # Ensure column order matches training
    X_user = pd.DataFrame([row], columns=feature_names)

    # Predict
    proba = float(pipe.predict_proba(X_user)[:, 1][0])
    label = int(proba >= THRESHOLD)

    st.subheader("Result")
    st.metric(
        label="Injury Risk Probability",
        value=f"{proba:.2%}",
        delta="High risk" if label == 1 else "Low risk",
        delta_color="inverse" if label == 1 else "normal"
    )

    st.subheader("Why this result?")
    for r in simple_reasons(row):
        st.write(f"- {r}")

    st.subheader("What you can do next")
    for a in simple_actions(label, row):
        st.write(f"- {a}")

    st.caption("This tool provides educational guidance and is not a medical diagnosis.")
