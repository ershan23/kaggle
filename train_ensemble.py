import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('F1 PitStop Ensemble: LGB + XGB + CAT')
print('=' * 60)

# ===== 1. Load =====
train = pd.read_csv('E:/py_project/playground-series-s6e5/train_processed.csv')
test = pd.read_csv('E:/py_project/playground-series-s6e5/test_processed.csv')

X = train.drop(columns=['PitNextLap'])
y = train['PitNextLap'].astype(int)
X_test = test.copy()

print(f'X: {X.shape}, y: {y.shape}, pos_rate: {y.mean():.4f}')

# ===== 2. Sample weights =====
weights = np.ones(len(y))
non_2023_mask = X['_is_2023'] == 0
pos_weight_val = (y[non_2023_mask] == 0).sum() / max((y[non_2023_mask] == 1).sum(), 1)
weights[non_2023_mask & (y == 1)] = pos_weight_val * 0.5
print(f'Non-2023 pos weight: {pos_weight_val:.2f}')

# ===== 3. CV Setup =====
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF storage
oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

# ===== 4. LGB Params =====
lgb_params = {
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
    'n_jobs': -1
}

# ===== 5. XGB Params =====
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 100
}

# ===== 6. CAT Params =====
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'depth': 8,
    'learning_rate': 0.05,
    'iterations': 2000,
    'subsample': 0.8,
    'colsample_bylevel': 0.7,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
    'early_stopping_rounds': 100,
    'allow_writing_files': False
}

# ===== 7. Cross-Validation Training =====
print('\n' + '=' * 60)
print('Starting 5-Fold CV')
print('=' * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    w_tr = weights[train_idx]
    
    print(f'\n--- Fold {fold+1} ---')
    
    # -- LGB --
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)]
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_folds
    auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    
    # -- XGB --
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_folds
    auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    
    # -- CAT --
    model_cat = cb.CatBoostClassifier(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        verbose=False
    )
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / n_folds
    auc_cat = roc_auc_score(y_val, oof_cat[val_idx])
    
    # -- Simple average --
    oof_avg = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
    auc_avg = roc_auc_score(y_val, oof_avg)
    
    print(f'  LGB AUC: {auc_lgb:.6f}')
    print(f'  XGB AUC: {auc_xgb:.6f}')
    print(f'  CAT AUC: {auc_cat:.6f}')
    print(f'  Ensemble Avg AUC: {auc_avg:.6f}')

# ===== 8. OOF Evaluation =====
print('\n' + '=' * 60)
print('Overall OOF Performance')
print('=' * 60)

auc_lgb_oof = roc_auc_score(y, oof_lgb)
auc_xgb_oof = roc_auc_score(y, oof_xgb)
auc_cat_oof = roc_auc_score(y, oof_cat)
oof_avg_all = (oof_lgb + oof_xgb + oof_cat) / 3
auc_avg_oof = roc_auc_score(y, oof_avg_all)

print(f'LGB  OOF AUC: {auc_lgb_oof:.6f}')
print(f'XGB  OOF AUC: {auc_xgb_oof:.6f}')
print(f'CAT  OOF AUC: {auc_cat_oof:.6f}')
print(f'Avg  OOF AUC: {auc_avg_oof:.6f}')

# ===== 9. Stacking (Logistic Regression meta-learner) =====
print('\n' + '=' * 60)
print('Stacking with Logistic Regression')
print('=' * 60)

stack_X = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_stack_X = np.column_stack([test_lgb, test_xgb, test_cat])

# Use 3-fold for meta model to avoid overfitting
meta_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_meta = np.zeros(len(y))
test_meta = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(meta_skf.split(stack_X, y)):
    meta_model = LogisticRegression(C=10, random_state=42)
    meta_model.fit(stack_X[tr_idx], y[tr_idx])
    oof_meta[val_idx] = meta_model.predict_proba(stack_X[val_idx])[:, 1]
    test_meta += meta_model.predict_proba(test_stack_X)[:, 1] / 3

auc_meta_oof = roc_auc_score(y, oof_meta)
print(f'Stacking (LR) OOF AUC: {auc_meta_oof:.6f}')

# ===== 10. Threshold optimization on OOF =====
print('\n' + '=' * 60)
print('Threshold Optimization')
print('=' * 60)

candidates = {
    'LGB': oof_lgb,
    'XGB': oof_xgb,
    'CAT': oof_cat,
    'Avg Ensemble': oof_avg_all,
    'Stacking (LR)': oof_meta
}

for name, preds in candidates.items():
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.6, 0.02):
        f1 = f1_score(y, (preds > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    auc_val = roc_auc_score(y, preds)
    print(f'{name:20s}: AUC={auc_val:.6f}, Best Thresh={best_thresh:.2f}, F1={best_f1:.4f}')

# ===== 11. Use best model (Stacking LR) for submission =====
best_name = 'Stacking (LR)'
best_preds = test_meta
best_thresh = 0.34  # Will recalc below

# Recalc best threshold for stacking
for thresh in np.arange(0.1, 0.6, 0.02):
    f1 = f1_score(y, (oof_meta > thresh).astype(int))
    if f1 > 0:
        best_thresh = thresh

best_thresh = 0.42  # Default, will be recalculated
best_f1 = 0
for thresh in np.arange(0.2, 0.55, 0.01):
    f1 = f1_score(y, (oof_meta > thresh).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f'\nStacking best threshold: {best_thresh:.4f}, F1: {best_f1:.4f}')

# ===== 12. Classification Report (Stacking) =====
pred_binary = (oof_meta > best_thresh).astype(int)
print('\n=== Classification Report - Stacking (LR) ===')
print(classification_report(y, pred_binary, digits=4))

# ===== 13. Generate Submissions =====
print('\n' + '=' * 60)
print('Generating Submissions')
print('=' * 60)

sub = pd.read_csv('E:/py_project/playground-series-s6e5/sample_submission.csv')

# Simple Average
sub['PitNextLap'] = (test_lgb + test_xgb + test_cat) / 3
sub_avg = sub.copy()
sub_avg['PitNextLap'] = ((test_lgb + test_xgb + test_cat) / 3 > best_thresh).astype(int)
sub_avg.to_csv('E:/py_project/playground-series-s6e5/submission_ensemble_avg.csv', index=False)

# Stacking
sub_stacking = sub.copy()
sub_stacking['PitNextLap'] = (test_meta > best_thresh).astype(int)
sub_stacking.to_csv('E:/py_project/playground-series-s6e5/submission_ensemble_stacking.csv', index=False)

# Also save raw probabilities
sub_probs = pd.DataFrame({
    'id': sub['id'],
    'PitNextLap_Avg': (test_lgb + test_xgb + test_cat) / 3,
    'PitNextLap_Stacking': test_meta
})
sub_probs.to_csv('E:/py_project/playground-series-s6e5/submission_ensemble_probs.csv', index=False)

print(f'Avg ensemble: positives={sub_avg.PitNextLap.sum()} ({sub_avg.PitNextLap.mean():.4f})')
print(f'Stacking ensemble: positives={sub_stacking.PitNextLap.sum()} ({sub_stacking.PitNextLap.mean():.4f})')

# ===== 14. Feature Importance Comparison =====
print('\n' + '=' * 60)
print('Feature Importance - Top 15 (from final fold)')
print('=' * 60)

# LGB
lgb_imp = pd.DataFrame({'feature': X.columns, 'lgb': model_lgb.feature_importances_}).sort_values('lgb', ascending=False)
# XGB
xgb_imp = pd.DataFrame({'feature': X.columns, 'xgb': model_xgb.feature_importances_}).sort_values('xgb', ascending=False)
# CAT
cat_imp = pd.DataFrame({'feature': X.columns, 'cat': model_cat.feature_importances_}).sort_values('cat', ascending=False)

imp_merged = lgb_imp.merge(xgb_imp, on='feature').merge(cat_imp, on='feature')
imp_merged['avg'] = imp_merged[['lgb', 'xgb', 'cat']].mean(axis=1)
imp_merged = imp_merged.sort_values('avg', ascending=False)

print(imp_merged[['feature', 'lgb', 'xgb', 'cat', 'avg']].head(20).to_string(index=False))

print('\n' + '=' * 60)
print('Ensemble complete!')
print('=' * 60)
