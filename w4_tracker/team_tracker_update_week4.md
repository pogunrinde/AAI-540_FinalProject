# Week 4 Tracker – Maternal Health Risk (RUN: 20251007-221946)

**Week-3 prefix:** s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week3/20251007-221711  
**Week-4 prefix:** s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week4/20251007-221946

## Benchmark (LogReg on Age + SystolicBP)
Acc: 0.765 | Prec: 0.719 | Rec: 0.697 | F1: 0.708 | AUC: 0.791

# XGBoost (full features)
Acc: 0.988 | Prec: 1.000 | Rec: 0.970 | F1: 0.985 | AUC: 0.999

# Artifacts
- Metrics JSON: s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week4/20251007-221946/metrics_compare.json
- Baseline CM:  s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week4/20251007-221946/baseline_cm.png
- XGBoost CM:   s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week4/20251007-221946/xgb_cm.png
- Model:        s3://sagemaker-us-east-1-849121223812/sagemaker-xgboost-2025-10-07-22-24-35-764/output/model.tar.gz
- Batch Output: s3://sagemaker-us-east-1-849121223812/aai540/maternal-risk/week4/20251007-221946/batch/outputs
