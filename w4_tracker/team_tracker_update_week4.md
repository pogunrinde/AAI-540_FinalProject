# Week 4 Tracker – Maternal Health Risk (RUN: 20250925-102857)

**Week-3 prefix:** s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week3/20250925-102647  
**Week-4 prefix:** s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250925-102857

## Benchmark (LogReg on Age + SystolicBP)
Acc: 0.765 | Prec: 0.719 | Rec: 0.697 | F1: 0.708 | AUC: 0.791

# XGBoost (full features)
Acc: 0.988 | Prec: 1.000 | Rec: 0.970 | F1: 0.985 | AUC: 0.999

# Artifacts
- Metrics JSON: s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250925-102857/metrics_compare.json
- Baseline CM:  s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250925-102857/baseline_cm.png
- XGBoost CM:   s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250925-102857/xgb_cm.png
- Model:        s3://sagemaker-us-east-1-533267301342/sagemaker-xgboost-2025-09-25-10-31-56-289/output/model.tar.gz
- Batch Output: s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250925-102857/batch/outputs
