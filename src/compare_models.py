import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_preprocess

# ------- Paths -------
FIG_DIR = '../figures'
REPORT_DIR = '../reports'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ------- Data -------
X_train, X_test, y_train, y_test = load_and_preprocess('../data/sirp600.xlsx')

# ------- Models -------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# ------- Train & Evaluate -------
summary_rows = []
y_probas = {}   # for ROC curves
cms = {}        # confusion matrices

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Some models need predict_proba for ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function if available, else zeros (won't plot ROC)
        y_proba = getattr(model, "decision_function", lambda X: np.zeros(len(X)))(X_test)
        # map to [0,1] roughly
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    summary_rows.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    })

    y_probas[name] = y_proba
    cms[name] = confusion_matrix(y_test, y_pred)

summary_df = pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
summary_path = os.path.join(REPORT_DIR, "model_comparison_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved metrics → {summary_path}")
print(summary_df)

# ------- Bar chart: Accuracy / Precision / Recall / F1 -------
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10,6))
for i, m in enumerate(metrics):
    plt.bar(x + i*width - (1.5*width), summary_df[m].values, width, label=m)
plt.xticks(x, summary_df["Model"].values, rotation=0)
plt.ylabel("Score")
plt.title("Model Comparison (Higher is Better)")
plt.legend()
plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "model_comparison_bars.png")
plt.savefig(bar_path, dpi=200)
plt.close()
print(f"Saved figure → {bar_path}")

# ------- ROC Curves -------
plt.figure(figsize=(8,6))
for name in summary_df["Model"].values:
    y_proba = y_probas[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(FIG_DIR, "model_comparison_roc.png")
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"Saved figure → {roc_path}")

# ------- Confusion Matrices -------
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for ax, name in zip(axes, summary_df["Model"].values):
    ConfusionMatrixDisplay(cms[name]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
cm_path = os.path.join(FIG_DIR, "model_confusion_matrices.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Saved figure → {cm_path}")
