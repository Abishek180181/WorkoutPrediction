# src/preprocess.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC, RandomOverSampler

RANDOM_STATE = 42
TARGET_COL = "Injury_Risk"
DATA_XLSX = Path("data/sirp600.xlsx")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- helpers ---------------------------------------------------------------

def _col(df, name):
    """Return exact column if present; else case/space-insensitive match."""
    if name in df.columns:
        return name
    key = name.strip().lower().replace(" ", "_")
    mapping = {c: c for c in df.columns}
    alt = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    for orig, norm in alt.items():
        if norm == key:
            return orig
    raise KeyError(f"Column `{name}` not found in {list(df.columns)}")

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # BMI = Weight / Height(m)^2  (auto-detect cm vs m)
    try:
        h_col = _col(df, "Height")
        w_col = _col(df, "Weight")
        h = pd.to_numeric(df[h_col], errors="coerce")
        w = pd.to_numeric(df[w_col], errors="coerce")
        h_m = np.where(h > 3.0, h / 100.0, h)
        df["BMI"] = w / (h_m ** 2)
    except KeyError:
        pass

    # TLI = Training Intensity × Training Frequency
    try:
        ti = pd.to_numeric(df[_col(df, "Training Intensity")], errors="coerce")
        tf = pd.to_numeric(df[_col(df, "Training Frequency")], errors="coerce")
        df["TLI"] = ti * tf
    except KeyError:
        pass

    return df

def _detect_cat_cols(X: pd.DataFrame):
    cat = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    # also treat small-cardinality ints (e.g., Gender encoded 0/1) as categorical
    for c in X.columns:
        if c not in cat and pd.api.types.is_integer_dtype(X[c]) and X[c].nunique() <= 10:
            cat.append(c)
    # common categorical names that might be ints
    for name in ["Gender", "Injury_History"]:
        if name in X.columns and name not in cat:
            cat.append(name)
    cat = list(dict.fromkeys(cat))
    num = [c for c in X.columns if c not in cat]
    return cat, num

def _augment_to_12k(X_train: pd.DataFrame, y_train: pd.Series, cat_cols, target_total=12000):
    """SMOTENC to balance, then oversample to hit ~12k (≈6k/6k)."""
    # Factorize categoricals to integer codes for SMOTENC
    X_enc = X_train.copy()
    cat_maps = {}
    for c in cat_cols:
        codes, uniques = pd.factorize(X_enc[c], sort=True)
        X_enc[c] = codes
        cat_maps[c] = list(uniques)

    cat_idx = [X_enc.columns.get_loc(c) for c in cat_cols]
    smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_enc, y_train)

    per_class = target_total // 2
    ros = RandomOverSampler(sampling_strategy={0: per_class, 1: per_class}, random_state=RANDOM_STATE)
    X_big, y_big = ros.fit_resample(X_bal, y_bal)

    # decode categoricals back to original labels
    X_out = X_big.copy()
    for c in cat_cols:
        inv = cat_maps[c]
        X_out[c] = X_out[c].map(lambda i: inv[int(i)] if 0 <= int(i) < len(inv) else inv[0])

    aug = X_out.copy()
    aug[TARGET_COL] = y_big
    return aug

# --- main API --------------------------------------------------------------

def load_and_preprocess(path=DATA_XLSX):
    """Returns split sets (no scaling here)."""
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target `{TARGET_COL}` not found. Columns: {list(df.columns)}")

    df = _engineer_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load raw, engineer features, split
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_XLSX)

    # Build ~12k augmented train set
    cat_cols, num_cols = _detect_cat_cols(X_train)
    aug_df = _augment_to_12k(X_train, y_train, cat_cols, target_total=12000)

    # Persist outputs exactly as referenced in your report
    train_aug_path = OUT_DIR / "sirp600_train_aug.csv"
    test_path = OUT_DIR / "sirp600_test.csv"
    meta_path = OUT_DIR / "sirp600_aug_summary.json"

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    aug_df.to_csv(train_aug_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "random_state": RANDOM_STATE,
        "target_total_rows": int(len(aug_df)),
        "class_balance": aug_df[TARGET_COL].value_counts().to_dict(),
        "categorical_columns": cat_cols,
        "numeric_columns": [c for c in X_train.columns if c not in cat_cols],
        "source_file": str(DATA_XLSX),
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote {train_aug_path}, {test_path}, {meta_path}")
