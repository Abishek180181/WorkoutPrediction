# src/cv_evaluate.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

DATA = Path("data")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"
RANDOM_STATE = 42
N_SPLITS = 5

def detect_cats_nums(df: pd.DataFrame):
    cats = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for c in df.columns:
        if c not in cats and pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10:
            cats.append(c)
    cats = list(dict.fromkeys([c for c in cats if c != TARGET_COL]))
    nums = [c for c in df.columns if c not in cats and c != TARGET_COL]
    return cats, nums

def smote_indices(df: pd.DataFrame, cat_cols):
    return [df.columns.get_loc(c) for c in cat_cols]

# ---- load augmented train (12k) ----
train_path = DATA / "sirp600_train_aug.csv"
df = pd.read_csv(train_path)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

cat_cols, num_cols = detect_cats_nums(X)
cat_idx = smote_indices(X, cat_cols)

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

models = {
    "Logistic_Regression": LogisticRegression(max_iter=2000),
    "Decision_Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random_Forest":       RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost":             XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, clf in models.items():
    fold_id = 0
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline(steps=[
            ("smote", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
            ("pre", pre),
            ("clf", clf),
        ])
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_va)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "fold": fold_id,
            "accuracy": float(accuracy_score(y_va, y_pred)),
            "precision": float(precision_score(y_va, y_pred, zero_division=0)),
            "recall": float(recall_score(y_va, y_pred, zero_division=0)),
            "f1": float(f1_score(y_va, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_va, y_prob)),
        }
        results.append(metrics)
        fold_id += 1

# Save per-fold CSV
cv_df = pd.DataFrame(results)
cv_df.to_csv(REPORTS / "cv_results.csv", index=False)

# Save summary (mean ± std) per model
summary = {}
for name, group in cv_df.groupby("model"):
    summary[name] = {
        "accuracy": {"mean": group["accuracy"].mean(), "std": group["accuracy"].std()},
        "precision":{"mean": group["precision"].mean(),"std": group["precision"].std()},
        "recall":   {"mean": group["recall"].mean(),   "std": group["recall"].std()},
        "f1":       {"mean": group["f1"].mean(),       "std": group["f1"].std()},
        "roc_auc":  {"mean": group["roc_auc"].mean(),  "std": group["roc_auc"].std()},
        "n_folds": int(group.shape[0]),
    }

with open(REPORTS / "cv_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[OK] Wrote reports/cv_results.csv and reports/cv_summary.json")
