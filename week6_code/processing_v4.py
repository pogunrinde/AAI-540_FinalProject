# - Goal: always emit a single CSV named "train.csv" in XGBoost format:
#         [label, feat1, feat2, ...] with NO header.
# - Why "v4"? Earlier failures showed three different input schemas reaching this step.
#   This script auto-detects and handles all 3.
#     (A) RAW Kaggle (has 'RiskLevel')  -> engineer features + bin labels to 0/1
#     (B) Headerless [label, features]  -> ensure binary label + passthrough
#     (C) Headered engineered with 'label' column -> reorder so label is first
# - We print a lot so we can see the chosen branch in CloudWatch.

import os, glob
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES: List[str] = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

def _first_csv(dirpath: str) -> str:
    cands = sorted([p for p in glob.glob(os.path.join(dirpath, "*.csv"))])
    assert cands, f"No CSV found under {dirpath}"
    return cands[0]

def _ensure_binary_label(s: pd.Series) -> pd.Series:
    """Collapse to binary: 'high risk'->1 else 0; numeric 2->1 (treat mid as positive)."""
    if s.dtype == object:
        sl = s.astype(str).str.lower().str.strip()
        return sl.map({"high risk": 1}).fillna(0).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).replace({2: 1})

def _looks_headerless_label_first(df: pd.DataFrame) -> bool:
    """If columns are 0..n-1 and first col small ints (0/1[/2]), assume [label, feats...]."""
    int_like = all(isinstance(c, (int,)) for c in df.columns)
    if not int_like:
        return False
    first = df.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(first):
        return False
    uniq = set(pd.Series(first).dropna().unique().tolist())
    return uniq.issubset({0, 1, 2})

def engineer_raw(df: pd.DataFrame) -> pd.DataFrame:
    """RAW Kaggle schema (has RiskLevel) -> engineer features and return [label, features...]"""
    X = df.copy()
    X.columns = [str(c).strip() for c in X.columns]  # normalize headers

    missing = [c for c in RAW_FEATURES + ["RiskLevel"] if c not in X.columns]
    if missing:
        raise ValueError(f"[RAW mode] Missing columns: {missing}. Found: {list(X.columns)}")

    # explainable features for clinical intuition
    X["PulsePressure"]    = X["SystolicBP"] - X["DiastolicBP"]
    X["SBP_to_DBP"]       = X["SystolicBP"] / X["DiastolicBP"].replace(0, pd.NA)
    X["Fever"]            = (X["BodyTemp"] > 99.5).astype(int)
    X["Tachycardia"]      = (X["HeartRate"] >= 100).astype(int)
    X["HypertensionFlag"] = ((X["SystolicBP"] >= 140) | (X["DiastolicBP"] >= 90)).astype(int)

    cont = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","PulsePressure"]
    X[[f"z_{c}" for c in cont]] = StandardScaler().fit_transform(X[cont])

    y = _ensure_binary_label(X["RiskLevel"])
    X = X.drop(columns=["RiskLevel"])
    out = pd.concat([y.rename("label"), X], axis=1)
    return out

def reorder_headered_with_label(df: pd.DataFrame) -> pd.DataFrame:
    """Headered engineered schema: has 'label' column (often last). Move to col 0 and ensure binary."""
    X = df.copy()
    X.columns = [str(c).strip() for c in X.columns]
    assert "label" in X.columns, "Expected 'label' column in headered engineered schema."
    y = _ensure_binary_label(X["label"])
    X = X.drop(columns=["label"])
    out = pd.concat([y.rename("label"), X], axis=1)
    return out

def passthrough_label_first(df: pd.DataFrame) -> pd.DataFrame:
    """Headerless [label, feats...] -> ensure binary label; keep the rest."""
    out = df.copy()
    out.iloc[:, 0] = _ensure_binary_label(out.iloc[:, 0])
    return out

if __name__ == "__main__":
    print("==== processing_v4 START ====")
    in_dir  = "/opt/ml/processing/input"
    out_dir = "/opt/ml/processing/output"
    os.makedirs(out_dir, exist_ok=True)

    src = _first_csv(in_dir)
    print(f"[v4] reading: {src}")

    # Try headered read first
    df_head = pd.read_csv(src)
    head_cols = [str(c).strip() for c in df_head.columns]
    print(f"[v4] headered cols({len(head_cols)}): {head_cols[:20]}")
    print(f"[v4] head(headered):\n{df_head.head(3)}")

    if any(c == "RiskLevel" for c in head_cols):
        print("[v4] MODE = RAW (RiskLevel present)")
        out = engineer_raw(df_head)
    elif "label" in head_cols:
        print("[v4] MODE = HEADERED+LABEL (engineered with label col)")
        out = reorder_headered_with_label(df_head)
    else:
        print("[v4] MODE = TRY_HEADERLESS")
        df_hless = pd.read_csv(src, header=None)
        print(f"[v4] headerless shape: {df_hless.shape}")
        print(f"[v4] head(headerless):\n{df_hless.head(3)}")
        if _looks_headerless_label_first(df_hless):
            print("[v4] MODE = HEADERLESS label-first (passthrough)")
            out = passthrough_label_first(df_hless)
        else:
            raise KeyError(
                "Schema not recognized. No 'RiskLevel' or 'label', and not headerless label-first.\n"
                f"Headered columns: {head_cols}\nHeaderless shape: {df_hless.shape}"
            )

    out_path = os.path.join(out_dir, "train.csv")
    out.to_csv(out_path, index=False, header=False)
    print(f"[v4] wrote: {out_path} shape={out.shape}")
    print("==== processing_v4 DONE ====")
