import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print('Loading processed data...')
train = pd.read_csv('E:/py_project/playground-series-s6e5/train_processed.csv')
test = pd.read_csv('E:/py_project/playground-series-s6e5/test_processed.csv')

X = train.drop(columns=['PitNextLap'])
y = train['PitNextLap'].astype(int)
X_test = test.copy()

print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'Positive rate: {y.mean():.4f}')

# ===== Special handling for 2023 =====
# 2023 data has 0.96% positive rate - the model might struggle with this in CV
# Strategy: We'll use features including _is_2023 and let the model handle it
# Also create sample weights to balance classes within non-2023 data

weights = np.ones(len(y))
# Don't upweight 2023 since positive samples are ~0 there
# For non-2023, apply class weights
non_2023_mask = X['_is_2023'] == 0
pos_weight = (y[non_2023_mask] == 0).sum() / max((y[non_2023_mask] == 1).sum(), 1)
weights[non_2023_mask & (y == 1)] = pos_weight * 0.5  # dampen weight slightly

print(f'Non-2023 positive weight: {pos_weight:.2f}')

# ===== LightGBM Parameters =====
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 50,
    'random_state': 42,
    'verbose': -1,
    'early_stopping_rounds': 100,
    'device': 'cpu',
    'n_jobs': -1
}

# ===== 5-Fold Cross Validation =====
print('\n=== 5-Fold Stratified CV ===')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    w_tr = weights[train_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)]
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f'Fold {fold+1}: AUC = {auc:.6f}')
    
    # Feature importance
    fold_imp = pd.DataFrame({
        'feature': X.columns,
        f'importance_fold{fold}': model.feature_importances_
    })
    if feature_importance.empty:
        feature_importance = fold_imp
    else:
        feature_importance = feature_importance.merge(fold_imp, on='feature')

# OOF AUC
oof_auc = roc_auc_score(y, oof_preds)
print(f'\n=== Overall OOF AUC: {oof_auc:.6f} ===')

# Feature importance summary
feature_importance['importance_mean'] = feature_importance.filter(like='importance_').mean(axis=1)
feature_importance = feature_importance.sort_values('importance_mean', ascending=False)
print('\n=== TOP 20 FEATURE IMPORTANCE ===')
print(feature_importance[['feature', 'importance_mean']].head(20).to_string(index=False))

# Threshold analysis
print('\n=== THRESHOLD OPTIMIZATION ===')
from sklearn.metrics import f1_score
best_thresh = 0.5
best_f1 = 0
for thresh in np.arange(0.1, 0.6, 0.02):
    f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f'Best threshold: {best_thresh:.4f}, Best F1: {best_f1:.4f}')

# Apply best threshold
pred_binary = (oof_preds > best_thresh).astype(int)
print('\n=== CLASSIFICATION REPORT (OOF) ===')
print(classification_report(y, pred_binary, digits=4))

# ===== Generate Submission =====
print('\n=== GENERATING SUBMISSION ===')
sub = pd.read_csv('E:/py_project/playground-series-s6e5/sample_submission.csv')

sub['PitNextLap'] = (test_preds > best_thresh).astype(int)
sub.to_csv('E:/py_project/playground-series-s6e5/submission_baseline.csv', index=False)
print(f'Submission saved. Positive predictions: {sub.PitNextLap.sum()}, ({sub.PitNextLap.mean():.4f})')

# Also save with probabilities
sub_prob = pd.read_csv('E:/py_project/playground-series-s6e5/sample_submission.csv')
sub_prob['PitNextLap'] = test_preds
sub_prob.to_csv('E:/py_project/playground-series-s6e5/submission_baseline_prob.csv', index=False)
print(f'Prob submission saved.')

# Save OOF predictions for stacking
oof_df = pd.DataFrame({'PitNextLap_OOF': oof_preds, 'PitNextLap_True': y})
oof_df.to_csv('E:/py_project/playground-series-s6e5/oof_predictions.csv', index=False)
