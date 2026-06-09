"""Model factory, tuning spaces, and ensemble helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_distributions: dict


def make_pipeline(estimator) -> Pipeline:
    """Create the shared preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def model_specs(random_state: int = 42, scale_pos_weight: float = 1.0) -> dict[str, ModelSpec]:
    """Return supported model specifications and tuning spaces."""
    specs: dict[str, ModelSpec] = {
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=120,
                max_depth=24,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            param_distributions={
                "model__n_estimators": [80, 120, 180, 240],
                "model__max_depth": [12, 18, 24, 30, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
            },
        )
    }

    if XGBClassifier is not None:
        specs["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                n_estimators=160,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=random_state,
            ),
            param_distributions={
                "model__n_estimators": [100, 160, 240, 320],
                "model__max_depth": [3, 5, 7, 9],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "model__subsample": [0.75, 0.85, 1.0],
                "model__colsample_bytree": [0.75, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5, 8],
            },
        )

    if LGBMClassifier is not None:
        specs["lightgbm"] = ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                n_estimators=220,
                learning_rate=0.06,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
                verbose=-1,
            ),
            param_distributions={
                "model__n_estimators": [120, 180, 240, 320],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "model__num_leaves": [31, 63, 95, 127],
                "model__max_depth": [-1, 8, 12, 16],
                "model__min_child_samples": [10, 20, 40, 80],
                "model__subsample": [0.75, 0.85, 1.0],
                "model__colsample_bytree": [0.75, 0.85, 1.0],
            },
        )

    return specs


def make_soft_voting_ensemble(named_pipelines: list[tuple[str, Pipeline]]) -> VotingClassifier:
    """Create a probability-averaging ensemble."""
    return VotingClassifier(estimators=named_pipelines, voting="soft", n_jobs=-1)


def make_stacking_ensemble(
    named_pipelines: list[tuple[str, Pipeline]],
    cv: int = 3,
    random_state: int = 42,
) -> StackingClassifier:
    """Create a stacking ensemble with logistic regression as meta-learner."""
    final_estimator = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    return StackingClassifier(
        estimators=named_pipelines,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        cv=cv,
        n_jobs=-1,
    )
