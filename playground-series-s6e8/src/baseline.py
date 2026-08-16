"""快速 Baseline：单折 LogisticRegression + LightGBM，验证流水线与特征。"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import CAT_COLS, NUM_COLS, prepare_xy  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    X, y, X_test, _ = prepare_xy(train, test)

    # 数值 baseline：缺失填中位数
    num_feats = [c for c in X.columns if c not in CAT_COLS]
    Xn = X[num_feats]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic
    oof_lr = np.zeros(len(X))
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=0.5, n_jobs=-1)),
        ]
    )
    for tr, va in skf.split(Xn, y):
        pipe.fit(Xn.iloc[tr], y.iloc[tr])
        oof_lr[va] = pipe.predict_proba(Xn.iloc[va])[:, 1]
    print(f"LogisticRegression OOF AUC: {roc_auc_score(y, oof_lr):.6f}")

    # LightGBM quick 1-fold sanity
    import lightgbm as lgb

    Xlg = X.copy()
    for c in CAT_COLS:
        Xlg[c] = Xlg[c].astype(str).astype("category")
    tr, va = next(skf.split(Xlg, y))
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        Xlg.iloc[tr],
        y.iloc[tr],
        eval_set=[(Xlg.iloc[va], y.iloc[va])],
        categorical_feature=[c for c in CAT_COLS if c in Xlg.columns],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict_proba(Xlg.iloc[va])[:, 1]
    print(f"LightGBM fold0 AUC: {roc_auc_score(y.iloc[va], pred):.6f}")

    # 原始强特征简单阈值
    raw = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
    Xr = train[raw]
    oof_raw = np.zeros(len(train))
    for tr, va in skf.split(Xr, y):
        pipe.fit(Xr.iloc[tr], y.iloc[tr])
        oof_raw[va] = pipe.predict_proba(Xr.iloc[va])[:, 1]
    print(f"Raw3-feature LR OOF AUC: {roc_auc_score(y, oof_raw):.6f}")


if __name__ == "__main__":
    main()
