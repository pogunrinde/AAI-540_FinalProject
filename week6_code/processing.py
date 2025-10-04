
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = "/opt/ml/processing/output"
LOG = os.path.join(OUT, "_preprocess_log.json")

def log(msg, **kv):
    os.makedirs(OUT, exist_ok=True)
    rec = {"msg": msg, **kv}
    print("[preprocess]", json.dumps(rec))
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

def engineer(df: pd.DataFrame):
    X = df.copy()

    # --- create simple derived features ---
    X["PulsePressure"]    = X["SystolicBP"] - X["DiastolicBP"]
    X["SBPtoDBP"]         = X["SystolicBP"] / X["DiastolicBP"].replace(0, np.nan)
    X["Fever"]            = (X["BodyTemp"] > 99.5).astype(int)
    X["Tachycardia"]      = (X["HeartRate"] >= 100).astype(int)
    X["HypertensionFlag"] = ((X["SystolicBP"] >= 140) | (X["DiastolicBP"] >= 90)).astype(int)

    # --- scale continuous cols (store as z*) ---
    cont = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","PulsePressure","SBPtoDBP"]
    X[["z"+c for c in cont]] = StandardScaler().fit_transform(X[cont])

    # --- binarize label: low -> 0, (mid|high) -> 1 ---
    label_map = {"low risk":0, "mid risk":1, "high risk":1}
    if "RiskLevel" not in X.columns:
        raise ValueError("Column 'RiskLevel' not found. Available cols: %s" % list(X.columns))
    y = X["RiskLevel"].str.lower().map(label_map)

    # guard: drop rows that still ended up NaN (unexpected labels)
    bad = y.isna().sum()
    if bad > 0:
        log("dropping rows with unknown RiskLevel", bad_rows=int(bad))
        keep = ~y.isna()
        X, y = X.loc[keep].copy(), y.loc[keep].copy()

    X = X.drop(columns=["RiskLevel"])
    return X, y.astype(int)

def to_xgb(df):
    # xgboost algorithm container expects: label first column, no header
    cols = ["label"] + [c for c in df.columns if c != "label"]
    return df[cols]

if __name__ == "__main__":
    try:
        in_dir  = "/opt/ml/processing/input"
        out_dir = OUT
        os.makedirs(out_dir, exist_ok=True)

        csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
        assert csvs, f"No CSV in {in_dir}"
        log("found_input_csv", files=csvs)

        df = pd.read_csv(csvs[0])
        needed = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","RiskLevel"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        log("df_shape_before", rows=int(df.shape[0]), cols=int(df.shape[1]))
        X, y = engineer(df)
        log("label_counts", **{str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()})

        # --- split: 40% production holdout; remaining 60% -> 40/10/10 ---
        X_tmp, X_prod, y_tmp, y_prod = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
        X_train, X_rem, y_train, y_rem = train_test_split(X_tmp, y_tmp, test_size=(1/3), random_state=42, stratify=y_tmp)
        X_val, X_test, y_val, y_test   = train_test_split(X_rem, y_rem, test_size=0.5,  random_state=42, stratify=y_rem)

        def dump(name, Xd, yd, header=True):
            out = Xd.copy(); out["label"] = yd.values
            out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, header=header)
            log("wrote_split", name=name, rows=int(out.shape[0]), cols=int(out.shape[1]))

        # human-readable CSVs (header)
        dump("train", X_train, y_train, header=True)
        dump("val",   X_val,   y_val,   header=True)
        dump("test",  X_test,  y_test,  header=True)
        dump("production", X_prod, y_prod, header=True)

        # xgboost files (no header, label first)
        t = pd.read_csv(os.path.join(out_dir, "train.csv"))
        v = pd.read_csv(os.path.join(out_dir, "val.csv"))
        to_xgb(t).to_csv(os.path.join(out_dir, "xgb_train.csv"), index=False, header=False)
        to_xgb(v).to_csv(os.path.join(out_dir, "xgb_val.csv"),   index=False, header=False)
        log("wrote_xgb_files", files=["xgb_train.csv", "xgb_val.csv"])

        # sanity: ensure xgb files exist
        assert os.path.exists(os.path.join(out_dir, "xgb_train.csv"))
        assert os.path.exists(os.path.join(out_dir, "xgb_val.csv"))
        log("success")

    except Exception as e:
        # write message so you can read it directly from S3 if the step fails
        log("fatal_error", error=str(e))
        raise
