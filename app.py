# app.py
# Run from repo root: streamlit run src/app.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Workout Injury Risk Predictor", page_icon="💪", layout="wide")
st.title("💪 Workout Injury Risk Predictor")
st.write("Enter your details to estimate injury risk, see reasons, and get quick tips.")

MODEL_CAL_PATH = Path("models") / "injury_risk_rf_pipeline.joblib"   # calibrated pipeline for predictions
MODEL_BEST_PATH = Path("models") / "injury_risk_rf_best.joblib"      # best RF pipeline (for SHAP explain)
THRESH_PATH     = Path("models") / "threshold.json"

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=True)
def load_calibrated_pipeline():
    if not MODEL_CAL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_CAL_PATH}`. Train & calibrate first.")
        st.stop()
    return load(MODEL_CAL_PATH)

@st.cache_resource(show_spinner=False)
def load_threshold(default=0.45):
    if THRESH_PATH.exists():
        try:
            with open(THRESH_PATH) as f:
                return float(json.load(f)["best_threshold"])
        except Exception:
            pass
    return float(default)

pipe_cal = load_calibrated_pipeline()
THRESHOLD = load_threshold()

# Extract exact input columns the calibrated pipeline expects
try:
    pre_cal = pipe_cal.named_steps["pre"]  # ColumnTransformer
    num_cols_cal = list(pre_cal.transformers_[0][2])
    cat_cols_cal = list(pre_cal.transformers_[1][2])
    expected_cols = num_cols_cal + cat_cols_cal
except Exception:
    st.error("Could not introspect pipeline input columns. Retrain using ColumnTransformer with explicit column lists.")
    st.stop()

# ---------- SHAP artifacts (using best RF pipeline) ----------
try:
    import shap

    @st.cache_resource(show_spinner=False)
    def load_shap_artifacts():
        """Load best RF pipeline for explanations and build a reliable mapping from transformed indices to readable names."""
        if not MODEL_BEST_PATH.exists():
            return None

        pipe_best = load(MODEL_BEST_PATH)
        if "pre" not in pipe_best.named_steps or "clf" not in pipe_best.named_steps:
            return None

        pre_best  = pipe_best.named_steps["pre"]   # ColumnTransformer
        rf_best   = pipe_best.named_steps["clf"]   # RandomForestClassifier

        # Raw input columns used by the best pipeline
        num_cols_best = list(pre_best.transformers_[0][2])
        cat_cols_best = list(pre_best.transformers_[1][2])

        # OHE categories for each categorical column, in output order
        ohe = pre_best.named_transformers_["cat"]
        categories = ohe.categories_

        # Build transformed feature names: [num cols] + ["col = cat", ...]
        ohe_names = []
        ohe_map = []  # list of (col, cat) in transformed order
        for col, cats in zip(cat_cols_best, categories):
            for cat in cats:
                cat_str = str(cat)
                # Pretty-map gender 0/1
                if col.lower().startswith("gender") and cat_str in {"0", "1"}:
                    cat_str = "female" if cat_str == "0" else "male"
                ohe_names.append(f"{col} = {cat_str}")
                ohe_map.append((col, cat))  # keep raw cat for truth-check

        feat_names = list(num_cols_best) + ohe_names

        # Build a fast index->(col,cat) resolver starting at num feature count
        num_count = len(num_cols_best)
        idx_to_ohe = {num_count + i: ohe_map[i] for i in range(len(ohe_map))}

        # SHAP explainer
        try:
            explainer = shap.TreeExplainer(rf_best)
        except Exception:
            explainer = shap.Explainer(rf_best)

        return {
            "explainer": explainer,
            "pre": pre_best,
            "feat_names": feat_names,    # len == transformed features
            "num_cols_best": num_cols_best,
            "cat_cols_best": cat_cols_best,
            "idx_to_ohe": idx_to_ohe,    # transformed index -> (col, raw_cat)
        }

    shap_bundle = load_shap_artifacts()
except Exception:
    shap_bundle = None

# ---------- Helpers ----------
def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())

def align_to_expected(row_dict: dict, expected: list[str]) -> pd.DataFrame:
    """Map user-provided keys to exact training columns (handles Height_cm/Weight_kg and spacing)."""
    ukeys = { _norm(k): k for k in row_dict.keys() }
    alias = {
        "height": ["heightcm", "height_m", "heightinmeters", "height_in_cm", "height_in_m"],
        "weight": ["weightkg", "weight_kg"],
        "trainingfrequency": ["training_days", "trainingsessionsperweek", "trainingfreq"],
        "trainingduration": ["sessionduration", "duration"],
        "warmuptime": ["warmup", "warm_up_time"],
        "sleephours": ["sleep", "sleep_per_night"],
        "flexibilityscore": ["flexibility", "flex_score"],
        "muscleasymmetry": ["asymmetry", "muscle_imbalance"],
        "recoverytime": ["recovery_hours", "recovery_between_sessions"],
        "injuryhistory": ["injury_hist", "prior_injury"],
        "stresslevel": ["stress", "stress_score"],
        "trainingintensity": ["intensity"],
        "bmi": ["bodymassindex"],
        "tli": ["trainingloadindex", "training_load_index"],
        "gender": ["sex"],
        "age": ["years"],
    }
    out = {}
    for exp in expected:
        ne = _norm(exp)
        if ne in ukeys:
            out[exp] = row_dict[ukeys[ne]]; continue
        matched = False
        for base, alts in alias.items():
            if ne == base:
                for a in [base] + alts:
                    if a in ukeys:
                        out[exp] = row_dict[ukeys[a]]; matched = True; break
            if matched: break
        if not matched and exp in row_dict:
            out[exp] = row_dict[exp]; matched = True
        if not matched:
            out[exp] = np.nan
    return pd.DataFrame([out], columns=expected)

def risk_bucket(p: float, thresh: float) -> str:
    if p < 0.20: return "Low"
    if p < thresh: return "Medium"
    return "High"

FRIENDLY = {
    "Age": "Age",
    "Gender": "Gender",
    "Height": "Height",
    "Height_cm": "Height",
    "Weight": "Weight",
    "Weight_kg": "Weight",
    "BMI": "Body Mass Index (BMI)",
    "TLI": "Training Load Index (TLI)",
    "Training_Frequency": "Training frequency (sessions/week)",
    "Training_Duration": "Training duration (min/session)",
    "Warmup_Time": "Warm-up time (min)",
    "Sleep_Hours": "Sleep hours",
    "Flexibility_Score": "Flexibility score",
    "Muscle_Asymmetry": "Muscle asymmetry",
    "Recovery_Time": "Recovery time (hours)",
    "Injury_History": "Injury history",
    "Stress_Level": "Stress level",
    "Training_Intensity": "Training intensity",
}

def simple_reasons(row):
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
    return reasons[:3]

def simple_actions(high_risk: bool, row):
    actions = []
    if high_risk:
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
    return actions[:4]

def human_message_numeric(col, val, shap_sign):
    base = FRIENDLY.get(col, col)
    effect = "increased" if shap_sign > 0 else "reduced"
    return f"{base}: your value ({val}) {effect} your risk."

def human_message_ohe(col, cat_raw, active, shap_sign):
    base = FRIENDLY.get(col, col)
    cat_str = str(cat_raw)
    if col.lower().startswith("gender") and cat_str in {"0", "1"}:
        cat_str = "female" if cat_str == "0" else "male"
    effect = "increased" if shap_sign > 0 else "reduced"
    if active:
        return f"{base}: being **{cat_str}** {effect} your risk."
    else:
        # Usually inactive categories matter little; include only if strong
        other = f"not {cat_str}"
        return f"{base}: being **{other}** {effect} your risk."

# ---------- Input form ----------
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
    Training_Duration  = c1.number_input("Training Duration (minutes/session)", min_value=0, max_value=240, value=60, step=5)
    Warmup_Time        = c2.number_input("Warmup Time (minutes)", min_value=0, max_value=60, value=8, step=1)
    Sleep_Hours        = c3.number_input("Sleep Hours (per night)", min_value=0.0, max_value=12.0, value=6.5, step=0.1)

    Flexibility_Score  = c1.number_input("Flexibility Score (0–1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    Muscle_Asymmetry   = c2.number_input("Muscle Asymmetry (0–1)", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
    Recovery_Time      = c3.number_input("Recovery Time (hours between hard sessions)", min_value=0, max_value=120, value=24, step=1)

    Injury_History     = c1.number_input("Injury History (0=none, higher=more)", min_value=0, max_value=10, value=1, step=1)
    Stress_Level       = c2.number_input("Stress Level (0–10)", min_value=0, max_value=10, value=6, step=1)
    Training_Intensity = c3.number_input("Training Intensity (1–10)", min_value=1.0, max_value=10.0, value=6.8, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Compute TLI
    TLI = float(Training_Intensity) * float(Training_Frequency)

    # Build raw user row
    user_row = {
        "Age": Age, "Gender": Gender, "Height_cm": Height_cm, "Weight_kg": Weight_kg, "BMI": BMI,
        "TLI": TLI,
        "Training_Frequency": Training_Frequency, "Training_Duration": Training_Duration, "Warmup_Time": Warmup_Time,
        "Sleep_Hours": Sleep_Hours, "Flexibility_Score": Flexibility_Score, "Muscle_Asymmetry": Muscle_Asymmetry,
        "Recovery_Time": Recovery_Time, "Injury_History": Injury_History, "Stress_Level": Stress_Level,
        "Training_Intensity": Training_Intensity
    }

    # Align to calibrated pipeline expected columns
    X_user_cal = align_to_expected(user_row, expected_cols)

    # Predict with calibrated pipeline
    proba = float(pipe_cal.predict_proba(X_user_cal)[:, 1][0])
    label = int(proba >= THRESHOLD)
    bucket = risk_bucket(proba, THRESHOLD)

    st.subheader("Result")
    st.metric(
        label=f"Injury Risk Probability (threshold = {THRESHOLD:.2f})",
        value=f"{proba:.2%}",
        delta=f"{bucket} risk",
        delta_color="inverse" if label == 1 else "normal"
    )

    # Human-readable reasons & actions
    st.subheader("Why this result (rule-of-thumb)?")
    for r in simple_reasons(user_row):
        st.write(f"- {r}")

    st.subheader("What you can do next")
    for a in simple_actions(label == 1, user_row):
        st.write(f"- {a}")

    # ---------- SHAP Top Factors (plain-English) ----------
    if shap_bundle is not None:
        try:
            explainer     = shap_bundle["explainer"]
            pre_best      = shap_bundle["pre"]
            feat_names    = shap_bundle["feat_names"]
            num_cols_best = shap_bundle["num_cols_best"]
            idx_to_ohe    = shap_bundle["idx_to_ohe"]

            # Align user row for the best pipeline (same raw columns as pre_best)
            exp_cols_best = num_cols_best + list(pre_best.transformers_[1][2])
            X_user_best = align_to_expected(user_row, exp_cols_best)

            # Transform and ensure dense
            X_user_pre = pre_best.transform(X_user_best)
            X_user_pre = X_user_pre.toarray() if hasattr(X_user_pre, "toarray") else np.asarray(X_user_pre)

            # SHAP values -> 1D vector for sample 0
            shap_vals = explainer.shap_values(X_user_pre)
            if isinstance(shap_vals, list):  # older SHAP returns [class0, class1]
                sv = np.asarray(shap_vals[min(1, len(shap_vals) - 1)])[0]
            else:
                sv = np.asarray(shap_vals)[0]
            sv = np.asarray(sv, dtype=float).ravel()

            # Safety: if names and vector length mismatch, fallback generic labels
            if len(feat_names) != len(sv):
                feat_names = [f"feat_{i}" for i in range(len(sv))]

            # Build human messages for top |contribution| indices
            num_count = len(num_cols_best)
            active = X_user_pre[0]  # transformed features
            k = 5
            top_idx = np.argsort(np.abs(sv))[::-1]
            messages = []

            for j in top_idx:
                sign = np.sign(sv[j])
                if j < num_count:
                    col = num_cols_best[j]
                    val = X_user_best.iloc[0][col]
                    messages.append(human_message_numeric(col, val, sign))
                else:
                    # OHE feature: use index->(col, cat) map; include only active categories
                    if j in idx_to_ohe and active[j] > 0.5:
                        col, cat_raw = idx_to_ohe[j]
                        messages.append(human_message_ohe(col, cat_raw, True, sign))
                    else:
                        continue  # skip inactive or unmapped categories

                if len(messages) >= k:
                    break

            if messages:
                st.subheader("Top factors (data-driven)")
                for m in messages:
                    st.write(f"- {m}")
            else:
                st.info("No strong data-driven factors detected for this input.")

        except Exception as e:
            st.info(f"SHAP explanation unavailable ({e}).")

    st.caption("This tool provides educational guidance and is not a medical diagnosis.")
