import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

PATH = '../data/sirp600.xlsx'
BEST = dict(n_estimators=400, max_depth=None, min_samples_leaf=1,
            min_samples_split=10, max_features='sqrt', random_state=42, n_jobs=-1)

df = pd.read_excel(PATH)
X = df.drop(columns=['Injury_Risk'])
y = df['Injury_Risk']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(**BEST))
])

pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
y_proba = pipe.predict_proba(X_te)[:,1]

print("Accuracy:", accuracy_score(y_te, y_pred))
print(classification_report(y_te, y_pred))
print("ROC-AUC:", roc_auc_score(y_te, y_proba))

dump({'pipeline': pipe, 'feature_names': X.columns.tolist()}, '../models/injury_risk_rf_pipeline.joblib')
print("Saved ../models/injury_risk_rf_pipeline.joblib")
