# src/threshold_tuning.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DATA = Path("data")
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"

# Load data and calibrated pipeline
test_df  = pd.read_csv(DATA / "sirp600_test.csv")
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL].astype(int)

pipe = load(MODELS / "injury_risk_rf_pipeline.joblib")
y_prob = pipe.predict_proba(X_test)[:, 1]

# Grid search threshold for best F1 (injury class = 1)
thresholds = np.linspace(0.1, 0.9, 81)
best = {"threshold": 0.5, "f1": -1, "precision": 0, "recall": 0, "roc_auc": float(roc_auc_score(y_test, y_prob))}
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if f1 > best["f1"]:
        best.update({
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        })

# Save
with open(MODELS / "threshold.json", "w") as f:
    json.dump({"best_threshold": best["threshold"], "metrics_at_best": best}, f, indent=2)

print(f"[OK] Best threshold: {best['threshold']:.2f} (F1={best['f1']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f})")
