import numpy as np
import pandas as pd

from src.config import MAG_COLS, CAT_COLS
from src.utils import get_logger

logger = get_logger()


def build_features(df: pd.DataFrame, model_type: str = "lgbm") -> pd.DataFrame:
    """
    构造特征。model_type 控制类别编码方式：
      - 'lgbm': pd.Categorical (LightGBM 原生)
      - 'cat': 保留字符串 (CatBoost 原生)
      - 'xgb': OneHot 编码
    返回特征 DataFrame（不含 id / target）。
    """
    out = pd.DataFrame(index=df.index)

    # ─── 原始数值 ────────────────────────────────────────
    for col in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]:
        out[col] = df[col]

    # ─── 颜色指数 ────────────────────────────────────────
    out["u_g"] = df["u"] - df["g"]
    out["g_r"] = df["g"] - df["r"]
    out["r_i"] = df["r"] - df["i"]
    out["i_z"] = df["i"] - df["z"]
    out["u_r"] = df["u"] - df["r"]
    out["u_z"] = df["u"] - df["z"]
    out["g_i"] = df["g"] - df["i"]

    # ─── redshift 派生 ───────────────────────────────────
    out["redshift_log1p"] = np.log1p(df["redshift"].clip(lower=0))
    out["is_negative_z"] = (df["redshift"] < 0).astype(np.int8)
    out["redshift_bin"] = pd.qcut(df["redshift"], q=10, labels=False, duplicates="drop")

    # ─── 亮度聚合 ────────────────────────────────────────
    mag_vals = df[MAG_COLS].values
    out["mag_mean"] = mag_vals.mean(axis=1)
    out["mag_std"] = mag_vals.std(axis=1)
    out["mag_max"] = mag_vals.max(axis=1)
    out["mag_min"] = mag_vals.min(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]

    # ─── 天区坐标环形编码 ────────────────────────────────
    alpha_rad = np.deg2rad(df["alpha"])
    out["sin_alpha"] = np.sin(alpha_rad)
    out["cos_alpha"] = np.cos(alpha_rad)

    # ─── 类别交互 ────────────────────────────────────────
    spectral_x_pop = df["spectral_type"].astype(str) + "_" + df["galaxy_population"].astype(str)

    # ─── 类别编码（按 model_type）────────────────────────
    if model_type == "lgbm":
        out["spectral_type"] = pd.Categorical(df["spectral_type"])
        out["galaxy_population"] = pd.Categorical(df["galaxy_population"])
        out["spectral_x_population"] = pd.Categorical(spectral_x_pop)

    elif model_type == "cat":
        out["spectral_type"] = df["spectral_type"].astype(str)
        out["galaxy_population"] = df["galaxy_population"].astype(str)
        out["spectral_x_population"] = spectral_x_pop.astype(str)

    elif model_type == "xgb":
        # OneHot for spectral_type
        for val in ["M", "A/F", "G/K", "O/B"]:
            out[f"spectral_{val}"] = (df["spectral_type"] == val).astype(np.int8)
        # OneHot for galaxy_population
        out["pop_Red_Sequence"] = (df["galaxy_population"] == "Red_Sequence").astype(np.int8)
        out["pop_Blue_Cloud"] = (df["galaxy_population"] == "Blue_Cloud").astype(np.int8)
        # OneHot for interaction (fixed order to ensure train/test consistency)
        SXP_VALUES = [
            "M_Red_Sequence", "M_Blue_Cloud",
            "A/F_Red_Sequence", "A/F_Blue_Cloud",
            "G/K_Red_Sequence", "G/K_Blue_Cloud",
            "O/B_Red_Sequence", "O/B_Blue_Cloud",
        ]
        for val in SXP_VALUES:
            safe_name = f"sxp_{val}".replace("/", "_")
            out[safe_name] = (spectral_x_pop == val).astype(np.int8)

    logger.info(f"Built {out.shape[1]} features (model_type={model_type})")
    return out


def get_cat_features(model_type: str = "lgbm") -> list[str]:
    """返回需要告知模型的类别特征列名"""
    if model_type == "lgbm":
        return ["spectral_type", "galaxy_population", "spectral_x_population"]
    elif model_type == "cat":
        return ["spectral_type", "galaxy_population", "spectral_x_population"]
    return []
