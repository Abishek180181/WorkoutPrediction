# src/calibrate_and_save.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA = Path("data")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"

# Load data
train_df = pd.read_csv(DATA / "sirp600_train_aug.csv")
test_df  = pd.read_csv(DATA / "sirp600_test.csv")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].astype(int)
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL].astype(int)

# Load best RF pipeline (smote -> pre -> clf)
best = load(MODELS / "injury_risk_rf_best.joblib")

# Replace final classifier with CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline

steps = list(best.steps)
last_name, base_clf = steps[-1]

def make_calibrator(estimator, method="isotonic"):
    # Compatibility across sklearn versions (estimator vs base_estimator)
    try:
        return CalibratedClassifierCV(estimator=estimator, cv=3, method=method)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, cv=3, method=method)

cal = make_calibrator(base_clf, method="isotonic")
steps[-1] = (last_name, cal)
calibrated_pipe = Pipeline(steps=steps)

# Fit; if isotonic fails (rare, e.g. not enough unique probs per fold), fallback to sigmoid
try:
    calibrated_pipe.fit(X_train, y_train)
except ValueError:
    cal = make_calibrator(base_clf, method="sigmoid")
    steps[-1] = (last_name, cal)
    calibrated_pipe = Pipeline(steps=steps)
    calibrated_pipe.fit(X_train, y_train)

# Evaluate on untouched test
y_prob = calibrated_pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

# Save artifacts
dump(calibrated_pipe, MODELS / "injury_risk_rf_pipeline.joblib")
with open(REPORTS / "calibrated_metrics.json", "w") as f:
    json.dump({"roc_auc": float(auc), "brier_score": float(brier)}, f, indent=2)

print("[OK] Saved calibrated pipeline -> models/injury_risk_rf_pipeline.joblib")
print(f"[OK] Test AUC = {auc:.3f} | Brier = {brier:.4f}")
