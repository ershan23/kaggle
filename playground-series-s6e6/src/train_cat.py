"""
CatBoost 5-fold training for Stellar Classification
Run: python -m src.train_cat
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    TRAIN_CSV, TEST_CSV, TARGET, NUM_CLASSES, CLASSES, CLASS2IDX,
    N_FOLDS, SEED, OOF_DIR, PREDS_DIR, MODELS_DIR, SUBMISSIONS_DIR,
)
from src.features import build_features, get_cat_features
from src.cv import get_folds, eval_oof, encode_target
from src.utils import get_logger, save_array, Timer

logger = get_logger()

PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    learning_rate=0.08,
    depth=7,
    l2_leaf_reg=5,
    iterations=2000,
    early_stopping_rounds=150,
    auto_class_weights="Balanced",
    random_seed=SEED,
    task_type="CPU",
    verbose=200,
)


def main():
    logger.info("=== CatBoost Training ===")

    with Timer("Load data"):
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)

    with Timer("Build features"):
        X_train = build_features(train, model_type="cat")
        X_test = build_features(test, model_type="cat")

    y = encode_target(train[TARGET])
    folds = get_folds(train)
    cat_features = get_cat_features("cat")

    oof_probs = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    test_probs = np.zeros((len(test), NUM_CLASSES), dtype=np.float64)
    fold_scores = []

    for fold_idx in range(N_FOLDS):
        logger.info(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        trn_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_trn, y_trn = X_train[trn_mask], y[trn_mask]
        X_val, y_val = X_train[val_mask], y[val_mask]

        train_pool = Pool(X_trn, label=y_trn, cat_features=cat_features)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_features)

        model = CatBoostClassifier(**PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # OOF predictions
        val_probs = model.predict_proba(X_val)
        oof_probs[val_mask] = val_probs

        # Test predictions (average across folds)
        test_probs += model.predict_proba(X_test) / N_FOLDS

        # Per-fold score
        from sklearn.metrics import balanced_accuracy_score
        fold_ba = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
        fold_scores.append(fold_ba)
        logger.info(f"Fold {fold_idx + 1} balanced_accuracy = {fold_ba:.6f} (best_iter={model.best_iteration_})")

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODELS_DIR / f"catboost_fold{fold_idx}.cbm"))

    # Overall OOF evaluation
    logger.info("\n=== Overall ===")
    overall_ba = eval_oof(y, oof_probs, label="CatBoost")
    logger.info(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")
    logger.info(f"Fold std: {np.std(fold_scores):.6f}")

    # Save artifacts
    save_array(oof_probs, "cat_oof", subdir="oof")
    save_array(test_probs, "cat_test", subdir="preds")
    logger.info("Saved OOF and test predictions")

    # Generate submission
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "class": [CLASSES[i] for i in np.argmax(test_probs, axis=1)]})
    sub.to_csv(SUBMISSIONS_DIR / "cat.csv", index=False)
    logger.info(f"Submission saved: submissions/cat.csv")

    logger.info(f"\n[OOF] balanced_accuracy = {overall_ba:.6f}")


if __name__ == "__main__":
    main()
