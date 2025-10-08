# - Our processing step writes the split file as "train.csv" (even for the test split),
#   so we look for that first; "test.csv" is a fallback.
# - This step must run in the XGBoost container so 'import xgboost' works.

import os, json, tarfile, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

if __name__ == "__main__":
    print("==== evaluate.py START ====")
    model_dir = "/opt/ml/processing/model"
    test_dir  = "/opt/ml/processing/test"
    out_dir   = "/opt/ml/processing/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    # unpack model.tar.gz created by the training job
    model_tar = os.path.join(model_dir, "model.tar.gz")
    print("[evaluate] extracting:", model_tar)
    with tarfile.open(model_tar) as t:
        t.extractall(model_dir)
    model_path = os.path.join(model_dir, "xgboost-model")
    print("[evaluate] model path:", model_path)
    booster = xgb.Booster(model_file=model_path)

    # find the test CSV produced by processing step
    candidate_paths = [
        os.path.join(test_dir, "train.csv"),  # our processors always write train.csv
        os.path.join(test_dir, "test.csv"),   # fallback name
    ]
    test_path = _first_existing(candidate_paths)
    if test_path is None:
        raise FileNotFoundError(f"No test file found among: {candidate_paths}")
    print("[evaluate] reading:", test_path)

    # XGBoost expects: [label, features...] without a header
    test_df = pd.read_csv(test_path, header=None)
    y_true = test_df.iloc[:, 0].values
    X      = test_df.iloc[:, 1:]
    dtest  = xgb.DMatrix(X, label=y_true)

    proba  = booster.predict(dtest)
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, proba)

    report = {
        "binary_classification_metrics": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "roc_auc": float(auc),
        }
    }
    out_json = os.path.join(out_dir, "evaluation.json")
    with open(out_json, "w") as f:
        json.dump(report, f)

    print("[evaluate] metrics:", report)
    print("[evaluate] wrote:", out_json)
    print("==== evaluate.py DONE ====")