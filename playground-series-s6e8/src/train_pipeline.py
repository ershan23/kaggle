"""Playground S6E8：多模型 CV 训练与融合提交。"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import CAT_COLS, prepare_xy  # noqa: E402

warnings.filterwarnings("ignore")

DATA_DIR = ROOT
OUT_DIR = ROOT / "output"
SEED = 42
N_SPLITS = 5
LOG_PATH = OUT_DIR / "train.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def encode_cats_for_trees(X: pd.DataFrame, X_test: pd.DataFrame, cats: list[str]):
    """类别列统一为 category，供 LGBM / XGB 使用。"""
    X = X.copy()
    X_test = X_test.copy()
    for c in cats:
        if c not in X.columns:
            continue
        X[c] = X[c].fillna("Missing").astype(str)
        X_test[c] = X_test[c].fillna("Missing").astype(str)
        cats_type = pd.CategoricalDtype(
            categories=sorted(set(X[c].tolist()) | set(X_test[c].tolist()))
        )
        X[c] = X[c].astype(cats_type)
        X_test[c] = X_test[c].astype(cats_type)
    return X, X_test


def train_lgbm(X, y, X_test, folds):
    import lightgbm as lgb

    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    scores = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 3000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }

    cat_features = [c for c in CAT_COLS if c in X.columns]

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y_va, oof[va_idx])
        scores.append(auc)
        log(f"  LGBM fold{fold}: {auc:.6f} (best_iter={model.best_iteration_})")

    return oof, pred, scores


def train_xgb(X, y, X_test, folds):
    import xgboost as xgb

    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    scores = []

    # category 列需 enable_categorical
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 3000,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True,
        "early_stopping_rounds": 100,
    }

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y_va, oof[va_idx])
        scores.append(auc)
        log(f"  XGB  fold{fold}: {auc:.6f} (best_iter={model.best_iteration})")

    return oof, pred, scores


def train_cat(X, y, X_test, folds):
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    scores = []

    # CatBoost 使用字符串类别（encode 后缺失已是 'nan'）
    Xc = X.copy()
    Xtc = X_test.copy()
    cat_features = [c for c in CAT_COLS if c in Xc.columns]
    for c in cat_features:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.6,
        "iterations": 3000,
        "random_seed": SEED,
        "verbose": False,
        "early_stopping_rounds": 100,
        "task_type": "CPU",
    }

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = Xc.iloc[tr_idx], Xc.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_va, y_va, cat_features=cat_features),
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        pred += model.predict_proba(Xtc)[:, 1] / N_SPLITS
        auc = roc_auc_score(y_va, oof[va_idx])
        scores.append(auc)
        log(f"  CAT  fold{fold}: {auc:.6f} (best_iter={model.best_iteration_})")

    return oof, pred, scores


def optimize_blend(oofs: dict[str, np.ndarray], y: pd.Series) -> dict[str, float]:
    """网格搜索简单凸组合权重。"""
    names = list(oofs.keys())
    mats = np.column_stack([oofs[n] for n in names])
    best_auc, best_w = -1.0, None

    # 粗网格：步长 0.1
    if len(names) == 3:
        for i in range(0, 11):
            for j in range(0, 11 - i):
                k = 10 - i - j
                w = np.array([i, j, k], dtype=float) / 10.0
                pred = mats @ w
                auc = roc_auc_score(y, pred)
                if auc > best_auc:
                    best_auc, best_w = auc, w
    else:
        # 等权兜底
        best_w = np.ones(len(names)) / len(names)
        best_auc = roc_auc_score(y, mats @ best_w)

    return {n: float(w) for n, w in zip(names, best_w)}, float(best_auc)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    log("Loading data...")
    train, test = load_data()
    log(f"train={train.shape}, test={test.shape}, pos_rate={train['addicted_label'].mean():.4f}")

    X, y, X_test, feature_cols = prepare_xy(train, test)
    log(f"features={len(feature_cols)}")

    X, X_test = encode_cats_for_trees(X, X_test, CAT_COLS)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    results = {}
    oofs, preds = {}, {}

    log("\n=== LightGBM ===")
    oofs["lgbm"], preds["lgbm"], scores = train_lgbm(X, y, X_test, folds)
    results["lgbm"] = {
        "fold_auc": scores,
        "oof_auc": float(roc_auc_score(y, oofs["lgbm"])),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    log(f"  LGBM OOF: {results['lgbm']['oof_auc']:.6f}")

    log("\n=== XGBoost ===")
    oofs["xgb"], preds["xgb"], scores = train_xgb(X, y, X_test, folds)
    results["xgb"] = {
        "fold_auc": scores,
        "oof_auc": float(roc_auc_score(y, oofs["xgb"])),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    log(f"  XGB  OOF: {results['xgb']['oof_auc']:.6f}")

    log("\n=== CatBoost ===")
    oofs["cat"], preds["cat"], scores = train_cat(X, y, X_test, folds)
    results["cat"] = {
        "fold_auc": scores,
        "oof_auc": float(roc_auc_score(y, oofs["cat"])),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    log(f"  CAT  OOF: {results['cat']['oof_auc']:.6f}")

    weights, blend_auc = optimize_blend(oofs, y)
    blend_pred = sum(weights[n] * preds[n] for n in weights)
    blend_oof = sum(weights[n] * oofs[n] for n in weights)

    results["blend"] = {"weights": weights, "oof_auc": blend_auc}
    log(f"\n=== Blend weights={weights} OOF AUC={blend_auc:.6f} ===")

    # 保存 OOF / 预测
    oof_df = pd.DataFrame({"id": train["id"], "y": y, **{f"oof_{k}": v for k, v in oofs.items()}})
    oof_df["oof_blend"] = blend_oof
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    sub = pd.DataFrame({"id": test["id"], "addicted_label": blend_pred})
    sub_path = ROOT / "submission.csv"
    sub.to_csv(sub_path, index=False)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # 单模型提交备份
    for name, p in preds.items():
        pd.DataFrame({"id": test["id"], "addicted_label": p}).to_csv(
            OUT_DIR / f"submission_{name}.csv", index=False
        )

    with open(OUT_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log(f"\nSaved: {sub_path}")
    log(f"CV summary: {json.dumps(results, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
