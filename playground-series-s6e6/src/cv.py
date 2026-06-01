import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

from src.config import N_FOLDS, SEED, TARGET, CLASSES, CLASS2IDX, OOF_DIR
from src.utils import get_logger

logger = get_logger()


def get_folds(df: pd.DataFrame) -> np.ndarray:
    """返回与 df 等长的 fold 数组 (0..N_FOLDS-1)，并缓存到 artifacts/oof/folds.npy"""
    folds_path = OOF_DIR / "folds.npy"
    if folds_path.exists():
        folds = np.load(folds_path)
        if len(folds) == len(df):
            logger.info(f"Loaded existing folds from {folds_path}")
            return folds

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(df), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df[TARGET])):
        folds[val_idx] = fold_idx

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(folds_path, folds)
    logger.info(f"Generated and saved {N_FOLDS}-fold splits to {folds_path}")
    return folds


def eval_oof(y_true: np.ndarray, oof_probs: np.ndarray, label: str = "") -> float:
    """评估 OOF 概率，返回 balanced_accuracy，同时打印 per-class recall"""
    y_pred = np.argmax(oof_probs, axis=1)
    ba = balanced_accuracy_score(y_true, y_pred)
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))))
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"[OOF{' ' + label if label else ''}] balanced_accuracy = {ba:.6f}")
    for i, cls in enumerate(CLASSES):
        logger.info(f"  {cls:8s} recall = {recalls[i]:.4f}")
    logger.info(f"  Confusion matrix:\n{cm}")
    return ba


def encode_target(series: pd.Series) -> np.ndarray:
    """将字符串标签转为 0/1/2 整数"""
    return series.map(CLASS2IDX).values.astype(np.int32)
