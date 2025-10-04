
import os, tarfile, json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

MODEL_TAR = "/opt/ml/processing/model/model.tar.gz"
TEST_CSV  = "/opt/ml/processing/test/test.csv"
EVAL_DIR  = "/opt/ml/processing/evaluation"

def load_model():
    tmp = "/tmp/model"; os.makedirs(tmp, exist_ok=True)
    with tarfile.open(MODEL_TAR, "r:gz") as t: t.extractall(tmp)
    booster = xgb.Booster(); booster.load_model(os.path.join(tmp, "xgboost-model"))
    return booster

if __name__ == "__main__":
    booster = load_model()
    test = pd.read_csv(TEST_CSV)
    y = test["label"].values; X = test.drop(columns=["label"])
    dtest = xgb.DMatrix(X, label=y)
    proba = booster.predict(dtest); pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try: auc = roc_auc_score(y, proba)
    except: auc = float("nan")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR,"evaluation.json"),"w") as f:
        json.dump({"binary_classification_metrics":
                   {"accuracy":acc,"precision":p,"recall":r,"f1":f1,"auc":auc}}, f, indent=2)
