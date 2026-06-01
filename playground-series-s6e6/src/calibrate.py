"""
OOF probability calibration for balanced_accuracy optimization.
Searches per-class weights to maximize balanced_accuracy on OOF predictions.
Run: python -m src.calibrate
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

from src.config import TRAIN_CSV, TARGET, CLASSES, NUM_CLASSES, OOF_DIR, PREDS_DIR
from src.cv import encode_target
from src.utils import get_logger, load_array, save_array

logger = get_logger()


def calibrate_probs(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply per-class weights and re-normalize"""
    adjusted = probs * weights[np.newaxis, :]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def objective(weights, probs, y_true):
    """Negative balanced_accuracy (to minimize)"""
    adjusted = calibrate_probs(probs, weights)
    y_pred = np.argmax(adjusted, axis=1)
    return -balanced_accuracy_score(y_true, y_pred)


def find_best_weights(probs: np.ndarray, y_true: np.ndarray, name: str) -> np.ndarray:
    """Search for optimal per-class weights"""
    best_result = None
    best_score = -1.0

    # Multi-start optimization
    starts = [
        np.array([1.0, 1.0, 1.0]),
        np.array([0.8, 1.0, 1.2]),
        np.array([0.7, 1.1, 1.3]),
        np.array([1.2, 0.9, 1.0]),
        np.array([0.6, 1.0, 1.5]),
    ]

    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            args=(probs, y_true),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7},
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_result = result

    weights = best_result.x
    # Normalize weights so mean = 1
    weights = weights / weights.mean()

    ba_before = balanced_accuracy_score(y_true, np.argmax(probs, axis=1))
    ba_after = balanced_accuracy_score(y_true, np.argmax(calibrate_probs(probs, weights), axis=1))

    logger.info(f"[{name}] Calibration weights: {dict(zip(CLASSES, [f'{w:.4f}' for w in weights]))}")
    logger.info(f"[{name}] BA before={ba_before:.6f} -> after={ba_after:.6f} (delta={ba_after-ba_before:+.6f})")

    return weights


def main():
    logger.info("=== Probability Calibration ===")

    train = pd.read_csv(TRAIN_CSV)
    y = encode_target(train[TARGET])

    models = ["lgbm", "cat", "xgb"]
    all_weights = {}

    for model_name in models:
        oof_probs = load_array(f"{model_name}_oof", subdir="oof")
        weights = find_best_weights(oof_probs, y, model_name)
        all_weights[model_name] = weights

        # Save calibrated OOF
        cal_oof = calibrate_probs(oof_probs, weights)
        save_array(cal_oof, f"{model_name}_oof_cal", subdir="oof")

        # Calibrate test predictions
        test_probs = load_array(f"{model_name}_test", subdir="preds")
        cal_test = calibrate_probs(test_probs, weights)
        save_array(cal_test, f"{model_name}_test_cal", subdir="preds")

    # Save weights
    np.save(OOF_DIR / "calibration_weights.npy", all_weights)
    logger.info("\nCalibration complete! Saved calibrated predictions.")


if __name__ == "__main__":
    main()
