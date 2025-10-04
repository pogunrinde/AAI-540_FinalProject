
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def engineer(df: pd.DataFrame):
    X = df.copy()
    # Simple extra features (you can extend from Week 3)
    X["PulsePressure"] = X["SystolicBP"] - X["DiastolicBP"]
    X["SBPtoDBP"]      = X["SystolicBP"] / X["DiastolicBP"].replace(0, np.nan)
    X["Fever"]         = (X["BodyTemp"] > 99.5).astype(int)
    X["Tachycardia"]   = (X["HeartRate"] >= 100).astype(int)
    X["HypertensionFlag"] = ((X["SystolicBP"] >= 140) | (X["DiastolicBP"] >= 90)).astype(int)

    # Scale a subset of continuous vars (z*)
    cont = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","PulsePressure","SBPtoDBP"]
    X[["z"+c for c in cont]] = StandardScaler().fit_transform(X[cont])

    # Binary label mapping from your Week-3 work
    y = X["RiskLevel"].map({"low risk":0, "high risk":1})
    X = X.drop(columns=["RiskLevel"])
    return X, y

def reorder_for_xgb(df):
    cols = ["label"] + [c for c in df.columns if c != "label"]
    return df[cols]

if __name__ == "__main__":
    in_dir  = "/opt/ml/processing/input"
    out_dir = "/opt/ml/processing/output"
    os.makedirs(out_dir, exist_ok=True)

    # Expect exactly one CSV in the input dir
    csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    assert csvs, f"No CSV found in {in_dir}"
    df = pd.read_csv(csvs[0])

    # Quick sanity checks (columns exist)
    req = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","RiskLevel"]
    miss = [c for c in req if c not in df.columns]
    assert not miss, f"Missing columns: {miss}"

    X, y = engineer(df)

    # Splits: 40% production; remaining 60% -> train 40%, val 10%, test 10% (stratified)
    X_tmp, X_prod, y_tmp, y_prod = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    X_train, X_rem, y_train, y_rem = train_test_split(X_tmp, y_tmp, test_size=(1/3), random_state=42, stratify=y_tmp)
    X_val, X_test, y_val, y_test   = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42, stratify=y_rem)

    def dump(name, Xd, yd):
        out = Xd.copy()
        out["label"] = yd.values
        out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    dump("train", X_train, y_train)
    dump("val",   X_val,   y_val)
    dump("test",  X_test,  y_test)
    dump("production", X_prod, y_prod)

    # XGBoost built-in expects label-first CSV without header
    for nm in ["train","val"]:
        d = pd.read_csv(os.path.join(out_dir, f"{nm}.csv"))
        reorder_for_xgb(d).to_csv(os.path.join(out_dir, f"xgb_{nm}.csv"), index=False, header=False)
