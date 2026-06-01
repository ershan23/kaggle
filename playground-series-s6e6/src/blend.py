"""
Model blending: search optimal weights and generate final submission.
Run: python -m src.blend
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

from src.config import (
    TRAIN_CSV, TEST_CSV, TARGET, CLASSES, NUM_CLASSES,
    SUBMISSIONS_DIR, OOF_DIR,
)
from src.cv import encode_target
from src.utils import get_logger, load_array

logger = get_logger()

MODELS = ["lgbm", "cat", "xgb"]


def blend_probs(probs_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average of probability arrays"""
    blended = np.zeros_like(probs_list[0])
    for probs, w in zip(probs_list, weights):
        blended += probs * w
    return blended


def objective(weights, probs_list, y_true):
    """Negative balanced_accuracy for blended predictions"""
    # Normalize weights to sum to 1
    w = np.abs(weights)
    w = w / w.sum()
    blended = blend_probs(probs_list, w)
    y_pred = np.argmax(blended, axis=1)
    return -balanced_accuracy_score(y_true, y_pred)


def find_blend_weights(probs_list: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Search optimal blending weights"""
    best_result = None
    best_score = -1.0

    starts = [
        np.array([1/3, 1/3, 1/3]),
        np.array([0.2, 0.4, 0.4]),
        np.array([0.3, 0.5, 0.2]),
        np.array([0.1, 0.6, 0.3]),
        np.array([0.25, 0.5, 0.25]),
    ]

    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            args=(probs_list, y_true),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7},
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_result = result

    weights = np.abs(best_result.x)
    weights = weights / weights.sum()
    return weights


def main():
    logger.info("=== Model Blending ===")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    y = encode_target(train[TARGET])

    # Load calibrated OOF and test predictions
    oof_list = []
    test_list = []
    for model_name in MODELS:
        oof_list.append(load_array(f"{model_name}_oof_cal", subdir="oof"))
        test_list.append(load_array(f"{model_name}_test_cal", subdir="preds"))

    # Report individual model scores (calibrated)
    logger.info("\n--- Individual model scores (calibrated) ---")
    for model_name, oof_probs in zip(MODELS, oof_list):
        ba = balanced_accuracy_score(y, np.argmax(oof_probs, axis=1))
        logger.info(f"  {model_name}: balanced_accuracy = {ba:.6f}")

    # Find optimal blend weights
    logger.info("\n--- Searching blend weights ---")
    weights = find_blend_weights(oof_list, y)
    logger.info(f"Optimal weights: {dict(zip(MODELS, [f'{w:.4f}' for w in weights]))}")

    # Evaluate blend on OOF
    blended_oof = blend_probs(oof_list, weights)
    blend_ba = balanced_accuracy_score(y, np.argmax(blended_oof, axis=1))
    logger.info(f"\n[BLEND OOF] balanced_accuracy = {blend_ba:.6f}")

    # Per-class recall
    from sklearn.metrics import recall_score
    y_pred = np.argmax(blended_oof, axis=1)
    recalls = recall_score(y, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    for i, cls in enumerate(CLASSES):
        logger.info(f"  {cls:8s} recall = {recalls[i]:.4f}")

    # Generate blended test predictions
    blended_test = blend_probs(test_list, weights)
    test_preds = np.argmax(blended_test, axis=1)

    # Write final submission
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({
        "id": test["id"],
        "class": [CLASSES[i] for i in test_preds],
    })
    sub.to_csv(SUBMISSIONS_DIR / "blend_final.csv", index=False)
    logger.info(f"\nFinal submission saved: submissions/blend_final.csv")
    logger.info(f"Shape: {sub.shape}, classes: {sub['class'].value_counts().to_dict()}")

    # Also save blend with equal weights as comparison
    equal_blend = blend_probs(oof_list, np.array([1/3, 1/3, 1/3]))
    equal_ba = balanced_accuracy_score(y, np.argmax(equal_blend, axis=1))
    logger.info(f"\n[Equal-weight blend] balanced_accuracy = {equal_ba:.6f}")

    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL RESULT: balanced_accuracy = {blend_ba:.6f}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
