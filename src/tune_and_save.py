import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from preprocess import load_and_preprocess

DATA_PATH = '../data/sirp600.xlsx'

# 1) Load data (scaled train/test from your preprocess.py)
X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

# Also load feature names for importance plotting
df_cols = pd.read_excel(DATA_PATH).drop(columns=['Injury_Risk']).columns

# 2) Define base model + param grid for tuning
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 6, 10, 14],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 3) Grid search (5-fold CV)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',      # you can switch to 'f1' (injury is class 1) or 'accuracy'
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)
best_model = grid.best_estimator_

# 4) Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ROC-AUC (needs probabilities)
y_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC:", auc)

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Random Forest ROC Curve")
plt.show()

# 5) Feature importance plot
importances = best_model.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10,5))
plt.bar(range(len(importances)), importances[idx])
plt.xticks(range(len(importances)), df_cols[idx], rotation=45, ha='right')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# 6) Save the tuned model
dump(best_model, '../models/injury_risk_rf_best.joblib')
print("\nSaved model to ../models/injury_risk_rf_best.joblib")
