# src/train_models.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

# -------- paths ----------
DATA = Path("data")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Injury_Risk"
RANDOM_STATE = 42

# -------- helpers ----------
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

def agg_feature_importances_rf(rf_pipeline, preprocessor, cat_cols, num_cols):
    """Sum OHE importances back to original feature names for RF."""
    rf = rf_pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    # Build expanded feature names from ColumnTransformer
    ohe = preprocessor.named_transformers_["cat"]
    try:
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
    except Exception:
        # sklearn < 1.0 fallback
        cat_feature_names = ohe.get_feature_names(cat_cols)
    num_feature_names = np.array(num_cols, dtype=object)
    all_names = np.concatenate([num_feature_names, cat_feature_names])
    # Map back
    agg = {}
    for name, val in zip(all_names, importances):
        base = name.split("_")[0] if name in num_feature_names else name.split("__")[0]
        agg[base] = agg.get(base, 0.0) + float(val)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

# -------- load data ----------
train_path = DATA / "sirp600_train_aug.csv"
test_path  = DATA / "sirp600_test.csv"
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].astype(int)
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL].astype(int)

cat_cols, num_cols = detect_cats_nums(X_train)
cat_idx = smote_indices(X_train, cat_cols)

# Preprocessor AFTER SMOTE
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# -------- models ----------
models = {
    "Logistic_Regression": LogisticRegression(max_iter=2000, n_jobs=None if hasattr(LogisticRegression(), "n_jobs") else None),
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

pipelines = {}
results = []

# -------- train/evaluate ----------
for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("smote", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
        ("pre", pre),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    results.append({
        "model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "confusion_matrix": cm
    })

    # Save model
    dump(pipe, MODELS / f"{name}.joblib")
    pipelines[name] = pipe

# Save metrics JSON
with open(REPORTS / "baseline_model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# -------- figures ----------
# ROC curves
plt.figure(figsize=(8,6))
for name, pipe in pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name)
plt.title("ROC Curves")
plt.tight_layout()
plt.savefig(FIGS / "model_comparison_roc.png", dpi=160)
plt.close()

# Confusion matrices (simple text table saved as PNG for brevity)
fig, ax = plt.subplots(figsize=(6,0.6+0.4*len(results)))
ax.axis("off")
lines = ["Confusion Matrices (TN FP / FN TP)"]
for r in results:
    cm = np.array(r["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    lines.append(f"{r['model']}: [[{tn} {fp}] / [{fn} {tp}]]")
ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")
plt.tight_layout()
plt.savefig(FIGS / "model_confusion_matrices.png", dpi=160)
plt.close()

# Bar chart (F1 comparison)
labels = [r["model"] for r in results]
f1s    = [r["f1"] for r in results]
plt.figure(figsize=(8,4))
plt.bar(labels, f1s)
plt.ylabel("F1 (injury class = 1)")
plt.title("Model Comparison (F1)")
plt.tight_layout()
plt.savefig(FIGS / "model_comparison_bars.png", dpi=160)
plt.close()

# RF feature importance (aggregated)
if "Random_Forest" in pipelines:
    rf_pipe = pipelines["Random_Forest"]
    # The ColumnTransformer inside the fitted pipeline is at step "pre"
    pre_fitted = rf_pipe.named_steps["pre"]
    agg = agg_feature_importances_rf(rf_pipe, pre_fitted, cat_cols, num_cols)
    # Save CSV
    pd.DataFrame(agg, columns=["feature","importance"]).to_csv(REPORTS / "rf_feature_importances.csv", index=False)
    # Plot top 15
    top = agg[:15]
    plt.figure(figsize=(8,5))
    plt.barh([a for a,b in reversed(top)], [b for a,b in reversed(top)])
    plt.title("Random Forest – Aggregated Feature Importances (Top 15)")
    plt.tight_layout()
    plt.savefig(FIGS / "rf_feature_importance.png", dpi=160)
    plt.close()

# Individual RF ROC for report parity
if "Random_Forest" in pipelines:
    plt.figure(figsize=(6,5))
    RocCurveDisplay.from_estimator(pipelines["Random_Forest"], X_test, y_test, name="Random_Forest")
    plt.title("Random Forest ROC")
    plt.tight_layout()
    plt.savefig(FIGS / "rf_roc_curve.png", dpi=160)
    plt.close()

print("[OK] Models trained and saved to models/. Metrics in reports/. Figures in figures/.")
