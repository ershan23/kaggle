# S6E8 快速说明

预测智能手机成瘾（ROC-AUC）。完整流程与结论见 [docs/竞赛分析报告.md](docs/竞赛分析报告.md)。

## 一键复现

```powershell
E:\miniconda\envs\playground\python.exe src\baseline.py
E:\miniconda\envs\playground\python.exe src\train_pipeline.py
```

## 当前最优本地结果

- **融合 OOF AUC：0.96471**（XGB 0.6 + LGBM 0.2 + CatBoost 0.2）
- 提交文件：`submission.csv`
