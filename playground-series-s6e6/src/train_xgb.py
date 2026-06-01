"""
XGBoost 5-fold training for Stellar Classification
Run: python -m src.train_xgb
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    TRAIN_CSV, TEST_CSV, TARGET, NUM_CLASSES, CLASSES, CLASS2IDX,
    N_FOLDS, SEED, OOF_DIR, PREDS_DIR, MODELS_DIR, SUBMISSIONS_DIR,
)
from src.features import build_features
from src.cv import get_folds, eval_oof, encode_target
from src.utils import get_logger, save_array, Timer

logger = get_logger()

PARAMS = dict(
    objective="multi:softprob",
    num_class=NUM_CLASSES,
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    seed=SEED,
)

NUM_ROUNDS = 5000
EARLY_STOPPING = 200


def main():
    logger.info("=== XGBoost Training ===")

    with Timer("Load data"):
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)

    with Timer("Build features"):
        X_train = build_features(train, model_type="xgb")
        X_test = build_features(test, model_type="xgb")

    y = encode_target(train[TARGET])
    folds = get_folds(train)

    # Compute balanced sample weights
    sample_weights = compute_sample_weight("balanced", y)

    oof_probs = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    test_probs = np.zeros((len(test), NUM_CLASSES), dtype=np.float64)
    fold_scores = []

    for fold_idx in range(N_FOLDS):
        logger.info(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        trn_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_trn, y_trn = X_train[trn_mask], y[trn_mask]
        X_val, y_val = X_train[val_mask], y[val_mask]
        w_trn = sample_weights[trn_mask]

        dtrain = xgb.DMatrix(X_trn, label=y_trn, weight=w_trn)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=200,
        )

        # OOF predictions
        val_probs = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof_probs[val_mask] = val_probs

        # Test predictions (average across folds)
        test_probs += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS

        # Per-fold score
        from sklearn.metrics import balanced_accuracy_score
        fold_ba = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
        fold_scores.append(fold_ba)
        logger.info(f"Fold {fold_idx + 1} balanced_accuracy = {fold_ba:.6f} (best_iter={model.best_iteration})")

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODELS_DIR / f"xgb_fold{fold_idx}.json"))

    # Overall OOF evaluation
    logger.info("\n=== Overall ===")
    overall_ba = eval_oof(y, oof_probs, label="XGBoost")
    logger.info(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")
    logger.info(f"Fold std: {np.std(fold_scores):.6f}")

    # Save artifacts
    save_array(oof_probs, "xgb_oof", subdir="oof")
    save_array(test_probs, "xgb_test", subdir="preds")
    logger.info("Saved OOF and test predictions")

    # Generate submission
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "class": [CLASSES[i] for i in np.argmax(test_probs, axis=1)]})
    sub.to_csv(SUBMISSIONS_DIR / "xgb.csv", index=False)
    logger.info(f"Submission saved: submissions/xgb.csv")

    logger.info(f"\n[OOF] balanced_accuracy = {overall_ba:.6f}")


if __name__ == "__main__":
    main()
