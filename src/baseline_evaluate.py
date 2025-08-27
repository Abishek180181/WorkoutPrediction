# src/baseline_evaluate.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA = Path("data")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"
RANDOM_STATE = 42
N_SPLITS = 5

# Load augmented train and untouched test
train_df = pd.read_csv(DATA / "sirp600_train_aug.csv")
test_df  = pd.read_csv(DATA / "sirp600_test.csv")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].astype(int)
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL].astype(int)

# ---- Holdout baseline: predict the majority class found in the TEST set ----
maj_class_test = int(y_test.mode()[0])
y_pred_holdout = np.full_like(y_test, maj_class_test)

baseline_holdout = {
    "majority_class": maj_class_test,
    "accuracy": float(accuracy_score(y_test, y_pred_holdout)),
    "precision": float(precision_score(y_test, y_pred_holdout, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_holdout, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred_holdout, zero_division=0)),
    "roc_auc_note": "AUC not defined for constant predictions (skipped)."
}

with open(REPORTS / "baseline_holdout.json", "w") as f:
    json.dump(baseline_holdout, f, indent=2)

# ---- CV baseline on TRAIN (augmented): majority per fold ----
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rows = []
for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
    y_va = y_train.iloc[va]
    maj = int(y_train.iloc[tr].mode()[0])
    y_pred = np.full_like(y_va, maj)
    rows.append({
        "fold": fold,
        "majority_class": maj,
        "accuracy": float(accuracy_score(y_va, y_pred)),
        "precision": float(precision_score(y_va, y_pred, zero_division=0)),
        "recall": float(recall_score(y_va, y_pred, zero_division=0)),
        "f1": float(f1_score(y_va, y_pred, zero_division=0)),
    })

baseline_cv = {
    "folds": rows,
    "summary": {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
        "precision_mean": float(np.mean([r["precision"] for r in rows])),
        "recall_mean": float(np.mean([r["recall"] for r in rows])),
        "f1_mean": float(np.mean([r["f1"] for r in rows])),
    }
}

with open(REPORTS / "baseline_cv.json", "w") as f:
    json.dump(baseline_cv, f, indent=2)

print("[OK] Wrote reports/baseline_holdout.json and reports/baseline_cv.json")
