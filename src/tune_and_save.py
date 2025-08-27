# src/tune_and_save.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

DATA = Path("data")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"
RANDOM_STATE = 42

# Load data
train_df = pd.read_csv(DATA / "sirp600_train_aug.csv")
test_df  = pd.read_csv(DATA / "sirp600_test.csv")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].astype(int)
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL].astype(int)

# Detect cats/nums + SMOTENC indices
def detect_cats_nums(df: pd.DataFrame):
    cats = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for c in df.columns:
        if c not in cats and pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10:
            cats.append(c)
    cats = list(dict.fromkeys(cats))
    nums = [c for c in df.columns if c not in cats]
    return cats, nums

cat_cols, num_cols = detect_cats_nums(X_train)
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

pipe = Pipeline(steps=[
    ("smote", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
    ("pre", pre),
    ("clf", rf),
])

param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 10],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",   # optimize injury-class F1
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)

best = gs.best_estimator_
dump(best, MODELS / "injury_risk_rf_best.joblib")

# Evaluate on untouched test
y_prob = best.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "best_params": gs.best_params_,
    "cv_best_score_f1": float(gs.best_score_),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
}

with open(REPORTS / "tuned_rf_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("[OK] Saved best RF to models/injury_risk_rf_best.joblib and reports/tuned_rf_test_metrics.json")
