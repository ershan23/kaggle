"""特征工程：缺失指示、比例特征、序数编码。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "addicted_label"
ID_COL = "id"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IMPACT_MAP = {"No": 0, "Yes": 1}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    for c in NUM_COLS + CAT_COLS:
        if c in out.columns:
            out[f"{c}_isna"] = out[c].isna().astype(np.int8)

    out["missing_count"] = out[[c for c in NUM_COLS + CAT_COLS if c in out.columns]].isna().sum(axis=1)

    daily = out["daily_screen_time_hours"]
    social = out["social_media_hours"]
    gaming = out["gaming_hours"]
    work = out["work_study_hours"]
    sleep = out["sleep_hours"]
    weekend = out["weekend_screen_time"]
    notify = out["notifications_per_day"]
    opens = out["app_opens_per_day"]

    out["entertainment_hours"] = social.fillna(0) + gaming.fillna(0)
    out["entertainment_hours"] = out["entertainment_hours"].where(
        social.notna() | gaming.notna(), np.nan
    )
    out["leisure_work_ratio"] = out["entertainment_hours"] / (work + eps)
    out["social_share"] = social / (daily + eps)
    out["gaming_share"] = gaming / (daily + eps)
    out["work_share"] = work / (daily + eps)
    out["entertainment_share"] = out["entertainment_hours"] / (daily + eps)
    out["weekend_weekday_ratio"] = weekend / (daily + eps)
    out["weekend_excess"] = weekend - daily
    out["sleep_deficit"] = 8.0 - sleep
    out["screen_sleep_ratio"] = daily / (sleep + eps)
    out["notify_per_open"] = notify / (opens + eps)
    out["engagement"] = notify.fillna(0) + opens.fillna(0)
    out["engagement"] = out["engagement"].where(notify.notna() | opens.notna(), np.nan)
    out["screen_per_open"] = daily / (opens + eps)
    out["heavy_screen"] = (daily >= 8.5).astype(np.float32)
    out["heavy_weekend"] = (weekend >= 10.5).astype(np.float32)
    out["heavy_social"] = (social >= 2.7).astype(np.float32)

    # 序数编码（保留原始类别列给 CatBoost / LGBM 类别特征）
    out["stress_ord"] = out["stress_level"].map(STRESS_MAP)
    out["impact_ord"] = out["academic_work_impact"].map(IMPACT_MAP)

    # 简单交互
    out["screen_x_social"] = daily * social
    out["screen_x_weekend"] = daily * weekend
    out["social_x_gaming"] = social * gaming
    out["stress_x_screen"] = out["stress_ord"] * daily

    return out


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame):
    train_fe = add_features(train)
    test_fe = add_features(test)

    drop_cols = [ID_COL, TARGET]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]

    X = train_fe[feature_cols]
    y = train_fe[TARGET].astype(int)
    X_test = test_fe[feature_cols]
    return X, y, X_test, feature_cols
