# Week 4 Tracker – Maternal Health Risk (RUN: 20250930-205405)

**Week-3 prefix:** s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week3/20250930-205143  
**Week-4 prefix:** s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250930-205405

## Benchmark (LogReg on Age + SystolicBP)
Acc: 0.765 | Prec: 0.719 | Rec: 0.697 | F1: 0.708 | AUC: 0.791

# XGBoost (full features)
Acc: 0.988 | Prec: 1.000 | Rec: 0.970 | F1: 0.985 | AUC: 0.999

# Artifacts
- Metrics JSON: s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250930-205405/metrics_compare.json
- Baseline CM:  s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250930-205405/baseline_cm.png
- XGBoost CM:   s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250930-205405/xgb_cm.png
- Model:        s3://sagemaker-us-east-1-533267301342/sagemaker-xgboost-2025-09-30-21-05-51-135/output/model.tar.gz
- Batch Output: s3://sagemaker-us-east-1-533267301342/aai540/maternal-risk/week4/20250930-205405/batch/outputs
