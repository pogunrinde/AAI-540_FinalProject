
import os, glob, json, pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def first_csv_in(d):
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    assert files, f'No CSV found in {d}'
    return files[0]

if __name__ == '__main__':
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    val_dir   = os.environ.get('SM_CHANNEL_VAL',   '/opt/ml/input/data/val')
    model_dir = os.environ.get('SM_MODEL_DIR',     '/opt/ml/model')

    df_tr = pd.read_csv(first_csv_in(train_dir))
    df_va = pd.read_csv(first_csv_in(val_dir))

    Xtr, ytr = df_tr[['Age','SystolicBP']], df_tr['label']
    Xva, yva = df_va[['Age','SystolicBP']], df_va['label']

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

    pred  = clf.predict(Xva)
    proba = clf.predict_proba(Xva)[:,1]
    acc = accuracy_score(yva, pred)
    p,r,f1,_ = precision_recall_fscore_support(yva, pred, average='binary', zero_division=0)
    try: auc = roc_auc_score(yva, proba)
    except: auc = float('nan')

    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, os.path.join(model_dir, 'model.joblib'))
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump({'accuracy':acc,'precision':p,'recall':r,'f1':f1,'roc_auc':auc}, f)
