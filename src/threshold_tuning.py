import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import load

PATH = '../data/sirp600.xlsx'
bundle = load('../models/injury_risk_rf_pipeline.joblib')
pipe = bundle['pipeline']

df = pd.read_excel(PATH)
X = df.drop(columns=['Injury_Risk'])
y = df['Injury_Risk']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

proba = pipe.predict_proba(X_te)[:,1]

best = (0.5, 0, 0, 0)  # (thr, f1_1, prec_1, rec_1)
for thr in np.linspace(0.2, 0.8, 25):
    y_hat = (proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_hat, average=None, labels=[1])
    if f1[0] > best[1]:
        best = (thr, f1[0], p[0], r[0])

print(f"Best threshold: {best[0]:.2f} | F1(1)={best[1]:.3f}, P(1)={best[2]:.3f}, R(1)={best[3]:.3f}")
