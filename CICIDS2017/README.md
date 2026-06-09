# CICIDS2017 入侵检测系统

这是一个基于 CICIDS2017 数据集的机器学习入侵检测系统项目，适合用于 Kaggle 实验、GitHub 作品集展示和安全方向面试。项目包含从 EDA、数据预处理、特征工程、模型训练、超参数调优、模型融合到预测推理的完整流程。

## 项目任务

- `X`：CICIDS2017 CSV 文件中的网络流量 flow-level 特征。
- `y`：二分类 IDS 标签，`BENIGN = 0`，所有非 `BENIGN` 标签统一视为 `ATTACK = 1`。
- 模型：RandomForest、XGBoost、LightGBM。
- 融合：soft voting 和 stacking。
- 指标：Accuracy、Precision、Recall、F1，其中攻击类作为正类。

## 项目结构

```text
configs/default.json          # 实验配置
src/cicids_ids/               # 可复用 Python 包
scripts/01_eda.py             # EDA 数据概览
scripts/02_train_baseline.py  # 训练 RF/XGBoost/LightGBM 基线模型
scripts/03_tune_models.py     # RandomizedSearchCV 超参数调优
scripts/04_train_ensemble.py  # soft voting / stacking 模型融合
scripts/05_predict.py         # 对 CICIDS 风格 CSV 进行预测
requirements.txt              # Python 依赖
```

原始数据文件较大，不建议提交到 GitHub。请将 CICIDS2017 的 CSV 文件放在：

```text
MachineLearningCVE/
```

## 环境安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果在 Kaggle 中运行，缺少依赖时可执行：

```python
!pip install -q lightgbm xgboost
```

## 运行 EDA

```bash
python scripts/01_eda.py --config configs/default.json
```

输出文件：

- `outputs/eda_profile.json`

EDA 会统计：

- 总样本数
- 特征数量
- 标签分布
- 正常 / 攻击比例
- 缺失值和无穷值
- 重复样本数量

## 训练基线模型

默认配置会从全量数据中分层抽样 300,000 条记录，用于快速实验。如果希望使用全量训练，可以把 `configs/default.json` 中的 `sample_size` 改为 `null`。

```bash
python scripts/02_train_baseline.py --config configs/default.json
```

输出文件：

- `outputs/baseline_metrics.csv`
- `outputs/baseline_metrics.json`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/lightgbm.joblib`

也可以只训练某一个模型：

```bash
python scripts/02_train_baseline.py --model lightgbm --sample-size 100000
```

## 超参数调优

```bash
python scripts/03_tune_models.py --config configs/default.json --n-iter 20
```

快速测试命令：

```bash
python scripts/03_tune_models.py --sample-size 20000 --n-iter 2 --model lightgbm
```

输出文件：

- `outputs/tuned_metrics.csv`
- `outputs/tuned_metrics.json`
- `models/*_tuned.joblib`

## 模型融合

soft voting 融合：

```bash
python scripts/04_train_ensemble.py --config configs/default.json --method soft_voting
```

stacking 融合：

```bash
python scripts/04_train_ensemble.py --config configs/default.json --method stacking
```

输出文件：

- `outputs/*_ensemble_metrics.json`
- `models/*_ensemble.joblib`

soft voting 是对多个模型的攻击概率做平均；stacking 是把多个基模型的预测概率交给二层逻辑回归模型学习。

## 预测新数据

```bash
python scripts/05_predict.py ^
  --model-path models/lightgbm.joblib ^
  --input-csv MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv ^
  --output-csv outputs/monday_predictions.csv
```

预测结果会新增：

- `predicted_target`：`0` 表示正常，`1` 表示攻击。
- `attack_probability`：模型预测为攻击的概率。

## 配置说明

核心配置位于 `configs/default.json`：

- `data_dir`：CSV 数据目录。
- `sample_size`：抽样行数，设为 `null` 表示使用全量。
- `test_size`：测试集比例。
- `deduplicate`：是否去除重复流。
- `split_strategy`：随机切分或文件级 holdout。
- `feature_engineering`：是否启用领域特征工程。
- `drop_zero_variance`：是否删除低信息量字段。
- `models`：需要训练的模型列表。
- `tuning`：调参次数、交叉验证折数和评分指标。

## 评估指标说明

本项目默认把攻击类作为正类：

- `TP`：攻击流量被正确识别为攻击。
- `TN`：正常流量被正确识别为正常。
- `FP`：正常流量被误报为攻击。
- `FN`：攻击流量被漏报为正常。

指标含义：

- `Accuracy`：总体预测正确率。
- `Precision`：模型报攻击时，有多大比例确实是攻击。
- `Recall`：真实攻击中，有多大比例被模型抓住。
- `F1`：Precision 和 Recall 的调和平均。

在 IDS 场景中，Recall 通常非常重要，因为漏报攻击的代价较高；Precision 也不能忽略，因为误报太多会导致告警疲劳。

## 更严格的实验建议

CICIDS2017 随机切分下的指标通常会很高，但这不等于模型已经可以直接用于真实生产环境。更严谨的验证方式包括：

- 去除重复样本后重新训练和评估。
- 使用按日期或按文件的 holdout 切分。
- 报告每种攻击类型的 Precision、Recall、F1。
- 对少数类攻击单独分析，例如 Heartbleed、Infiltration、SQL Injection。
- 根据业务代价调整分类阈值。
- 在新采集流量或其他 IDS 数据集上验证泛化能力。
