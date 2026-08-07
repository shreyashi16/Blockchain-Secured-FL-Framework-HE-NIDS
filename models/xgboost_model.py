"""
models/xgboost_model.py
========================
Thin, well-typed wrapper around ``xgboost.XGBClassifier`` used identically
by every federated client (and, for evaluation purposes, by the server's
global ensemble). Handles:

  * automatic binary vs multi-class detection
  * class-imbalance handling via ``scale_pos_weight`` (binary) or
    per-sample class weights (multi-class) - WITHOUT touching the
    already-completed preprocessing pipeline
  * warm-start continuation across communication rounds (local "epochs" are
    additional boosting rounds appended each time ``fit`` is called)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

import config


class XGBoostModel:
    """Wraps an ``xgb.XGBClassifier`` with FL-friendly warm-start training."""

    def __init__(
        self,
        num_classes: int,
        feature_names: Optional[List[str]] = None,
        random_state: int = config.RANDOM_SEED,
    ) -> None:
        self.num_classes = num_classes
        self.feature_names = feature_names
        self.random_state = random_state
        self._xgb_cfg = config.XGB_CONFIG
        self._booster_estimator: Optional[xgb.XGBClassifier] = None
        self._total_boosting_rounds = 0

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _build_estimator(self, n_estimators: int) -> xgb.XGBClassifier:
        common_kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=self._xgb_cfg.learning_rate,
            max_depth=self._xgb_cfg.max_depth,
            min_child_weight=self._xgb_cfg.min_child_weight,
            subsample=self._xgb_cfg.subsample,
            colsample_bytree=self._xgb_cfg.colsample_bytree,
            reg_alpha=self._xgb_cfg.reg_alpha,
            reg_lambda=self._xgb_cfg.reg_lambda,
            n_jobs=self._xgb_cfg.n_jobs,
            tree_method=self._xgb_cfg.tree_method,
            random_state=self.random_state,
            use_label_encoder=False,
        )
        if self.num_classes <= 2:
            return xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", **common_kwargs
            )
        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=self.num_classes,
            **common_kwargs,
        )

    @staticmethod
    def detect_num_classes(y: np.ndarray) -> int:
        return int(len(np.unique(y)))

    def _compute_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """scale_pos_weight-equivalent for binary, balanced sample weights for multi-class."""
        if not self._xgb_cfg.use_class_balancing:
            return None
        return compute_sample_weight(class_weight="balanced", y=y)

    @staticmethod
    def compute_scale_pos_weight(y: np.ndarray) -> float:
        """Ratio of negative to positive samples - only meaningful for binary tasks."""
        y = np.asarray(y)
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        return n_neg / n_pos if n_pos > 0 else 1.0

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int,
        warm_start: bool = True,
    ) -> "XGBoostModel":
        """
        Train (or continue training) for ``n_estimators`` additional boosting
        rounds. When ``warm_start`` is True and a booster already exists from
        a previous communication round, training continues from it (this is
        what turns "local epochs" into genuine incremental federated
        training rather than retraining from scratch every round).
        """
        sample_weight = self._compute_weights(y)
        target_total = self._total_boosting_rounds + n_estimators if warm_start else n_estimators

        estimator = self._build_estimator(n_estimators=target_total)
        xgb_model = None
        if warm_start and self._booster_estimator is not None:
            xgb_model = self._booster_estimator.get_booster()

        estimator.fit(X, y, sample_weight=sample_weight, xgb_model=xgb_model)

        self._booster_estimator = estimator
        self._total_boosting_rounds = target_total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._booster_estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._booster_estimator.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Normalized (sums to 1) importance vector, aligned to ``self.feature_names`` order."""
        self._check_fitted()
        booster = self._booster_estimator.get_booster()
        score = booster.get_score(importance_type=importance_type)

        n_features = len(self.feature_names) if self.feature_names else self._booster_estimator.n_features_in_
        importance = np.zeros(n_features, dtype=np.float64)
        for feat_key, value in score.items():
            # xgboost keys are like "f0", "f1", ...
            idx = int(feat_key[1:])
            if idx < n_features:
                importance[idx] = value

        total = importance.sum()
        return importance / total if total > 0 else importance

    def _check_fitted(self) -> None:
        if self._booster_estimator is None:
            raise RuntimeError("XGBoostModel.fit() must be called before inference.")

    @property
    def booster_estimator(self) -> xgb.XGBClassifier:
        self._check_fitted()
        return self._booster_estimator

    @property
    def total_boosting_rounds(self) -> int:
        return self._total_boosting_rounds

    def save(self, path: str) -> None:
        self._check_fitted()
        self._booster_estimator.save_model(path)

    def load(self, path: str, n_estimators: int) -> "XGBoostModel":
        estimator = self._build_estimator(n_estimators=n_estimators)
        estimator.load_model(path)
        self._booster_estimator = estimator
        self._total_boosting_rounds = n_estimators
        return self
