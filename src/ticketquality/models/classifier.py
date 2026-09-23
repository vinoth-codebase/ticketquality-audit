"""Premature-closure risk classifier.

The estimator is a gradient-boosted tree — LightGBM when the native
library loads, otherwise scikit-learn's histogram gradient booster, which
shares the same algorithmic family and keeps the repository usable on
minimal environments.

On top of the booster sits **post-hoc probability calibration** (isotonic
or Platt/sigmoid) and a *recall-first* decision threshold.

Why recall-first? Missing a genuine premature closure means an unhappy
customer discovers the problem before QA does. A false alarm merely costs
a peer review. The cost asymmetry is roughly ten to one, so the threshold
optimiser selects the most precise operating point that still clears the
configured recall floor.

Why post-hoc calibration rather than wrapping the booster in
``CalibratedClassifierCV``? Because SHAP explanations must be exact and
attributable to *one* model. Calibration is a monotonic remap of the
booster's output, so keeping it as a separate final layer leaves the tree
ensemble intact and fully explainable — the SHAP values explain the
log-odds the booster produces, which the calibrator then re-scales.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, recall_score

from ..config import AuditConfig, get_config
from ..features.transformers import (
    DEFAULT_CATEGORICAL,
    DEFAULT_NUMERIC,
    TicketFeaturePipeline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReopenRiskClassifier",
    "TicketAuditModel",
    "optimize_threshold",
    "risk_level",
    "LIGHTGBM_AVAILABLE",
]


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


LIGHTGBM_AVAILABLE = _lightgbm_available()


# --------------------------------------------------------------------------- #
# Threshold optimisation
# --------------------------------------------------------------------------- #
def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_floor: float = 0.80,
    grid_size: int = 99,
) -> Tuple[float, Dict[str, float]]:
    """Choose the operating threshold that maximises precision subject to
    a recall floor.

    Returns ``(threshold, stats)`` where ``stats`` carries the achieved
    precision, recall and F1 at that threshold, plus a boolean
    ``recall_floor_met`` flag.
    """

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have identical shape")

    if len(np.unique(y_true)) < 2:
        threshold = 0.5
        preds = (y_prob >= threshold).astype(int)
        precision = float(y_true[preds == 1].mean()) if preds.sum() > 0 else 0.0
        positive_recall = (
            float(preds[y_true == 1].mean()) if (y_true == 1).any() else 0.0
        )
        return threshold, {
            "precision": precision,
            "recall": positive_recall,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "recall_floor_met": False,
        }

    # Sweep the *actual* precision-recall operating points rather than a
    # fixed linear grid. Calibrated probabilities are often sharply
    # bimodal, so a linear grid can step straight over the useful region.
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]

    best_threshold = 0.5
    best_stats: Dict[str, float] = {}
    fallback_threshold = 0.5
    fallback_stats: Dict[str, float] = {}

    if len(thresholds):
        fallback_index = int(
            np.argmax(
                [
                    f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                    for t in thresholds
                ]
            )
        )
        fallback_threshold = float(thresholds[fallback_index])

        eligible = np.where(recall_curve >= recall_floor)[0]
        if eligible.size:
            # Highest precision among operating points that clear the floor.
            best_index = eligible[np.argmax(precision_curve[eligible])]
            best_threshold = float(thresholds[best_index])
            preds = (y_prob >= best_threshold).astype(int)
            best_stats = {
                "precision": float(precision_curve[best_index]),
                "recall": float(recall_curve[best_index]),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "recall_floor_met": True,
            }

        preds = (y_prob >= fallback_threshold).astype(int)
        fallback_recall = (
            float(preds[y_true == 1].mean()) if (y_true == 1).any() else 0.0
        )
        fallback_stats = {
            "precision": float(y_true[preds == 1].mean()) if preds.sum() > 0 else 0.0,
            "recall": fallback_recall,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "recall_floor_met": bool(fallback_recall >= recall_floor),
        }

    if best_stats:
        return best_threshold, best_stats
    # Unreachable for binary targets: a recall floor <= 1 is always met by
    # lowering the threshold. Retained as a defensive fallback.
    return fallback_threshold, fallback_stats  # pragma: no cover


def risk_level(
    probability: float, low_upper: float = 0.35, elevated_upper: float = 0.60
) -> str:
    """Map a reopen probability to a management-friendly risk band."""

    if probability < low_upper:
        return "LOW"
    if probability < elevated_upper:
        return "ELEVATED"
    return "CRITICAL"


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #
@dataclass
class ReopenRiskClassifier:
    """Gradient-boosted binary classifier with calibration and a
    recall-first threshold."""

    params: Dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    calibrate: bool = True
    calibration_method: str = "isotonic"
    random_state: int = 42
    recall_floor: float = 0.80
    recall_margin: float = 0.0
    threshold_grid_size: int = 99

    # learned attributes
    estimator_: Any = None
    calibrator_: Any = None
    threshold_: float = 0.5
    threshold_stats_: Dict[str, float] = field(default_factory=dict)
    backend_: str = ""
    feature_names_: List[str] = field(default_factory=list)
    n_features_in_: int = 0

    # ------------------------------------------------------------------ #
    def _resolve_backend(self) -> str:
        if self.backend in {"lightgbm", "lgbm"}:
            if not LIGHTGBM_AVAILABLE:
                raise RuntimeError(
                    "LightGBM requested but its native library is unavailable"
                )
            return "lightgbm"
        if self.backend == "sklearn":
            return "sklearn"
        return "lightgbm" if LIGHTGBM_AVAILABLE else "sklearn"

    def _build_base_estimator(self) -> Any:
        backend = self._resolve_backend()
        self.backend_ = backend
        params = dict(self.params or {})

        if backend == "lightgbm":
            from lightgbm import LGBMClassifier

            lgbm_params = dict(params.get("lightgbm", {})) or {}
            if not lgbm_params:
                lgbm_params = {
                    k: v for k, v in params.items() if k not in {"sklearn_fallback", "calibration", "backend"}
                }
            lgbm_params.setdefault("random_state", self.random_state)
            return LGBMClassifier(**lgbm_params)

        from sklearn.ensemble import HistGradientBoostingClassifier

        sk_params = dict(params.get("sklearn_fallback", {})) or {}
        sk_params.setdefault("random_state", self.random_state)
        return HistGradientBoostingClassifier(**sk_params)

    # ------------------------------------------------------------------ #
    def _raw_probability(self, X: np.ndarray) -> np.ndarray:
        """Booster probability *before* calibration."""

        if self.estimator_ is None:
            raise RuntimeError("classifier is not fitted")
        proba = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _fit_calibrator(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.calibrate or len(np.unique(y)) < 2:
            self.calibrator_ = None
            return
        raw = self._raw_probability(X)
        try:
            if self.calibration_method == "sigmoid":
                calibrator = LogisticRegression(C=1e6, solver="lbfgs")
                calibrator.fit(raw.reshape(-1, 1), y)
            else:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(raw, y)
            self.calibrator_ = calibrator
        except Exception as exc:  # pragma: no cover - numerical edge cases
            logger.warning("calibration failed (%s); using raw probabilities", exc)
            self.calibrator_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ReopenRiskClassifier":
        """Fit the booster, calibrate it, then tune the threshold."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.n_features_in_ = X.shape[1]
        self.feature_names_ = (
            list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]
        )

        estimator = self._build_base_estimator()
        estimator.fit(X, y)
        self.estimator_ = estimator

        self._fit_calibrator(X, y)

        probabilities = self.predict_proba(X)[:, 1]
        target_recall = min(1.0, self.recall_floor + self.recall_margin)
        self.threshold_, self.threshold_stats_ = optimize_threshold(
            y, probabilities, target_recall, self.threshold_grid_size
        )
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated class probabilities, shape ``(n, 2)``."""

        X = np.asarray(X, dtype=float)
        raw = self._raw_probability(X)
        if self.calibrator_ is not None:
            if self.calibration_method == "sigmoid":
                calibrated = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
            else:
                calibrated = self.calibrator_.predict(raw)
            calibrated = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        else:
            calibrated = raw
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Hard labels using the tuned (or supplied) threshold."""

        proba = self.predict_proba(X)[:, 1]
        cut = self.threshold_ if threshold is None else threshold
        return (proba >= cut).astype(int)

    def predict_risk(self, X: np.ndarray, **bands) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return np.array([risk_level(p, **bands) for p in proba])

    # ------------------------------------------------------------------ #
    def feature_importances(self) -> pd.DataFrame:
        """Gain-based importances for the booster."""

        if self.estimator_ is None:
            raise RuntimeError("classifier is not fitted")

        names = self.feature_names_ or [f"f{i}" for i in range(self.n_features_in_)]
        importances = np.zeros(len(names))
        for attr in ("feature_importances_",):
            if hasattr(self.estimator_, attr):
                values = np.asarray(getattr(self.estimator_, attr), dtype=float)
                if values.shape[0] == len(names):
                    importances = values
                    break

        frame = pd.DataFrame({"feature": names, "importance": importances})
        total = frame["importance"].sum()
        frame["importance_pct"] = (
            (frame["importance"] / total * 100).round(3) if total > 0 else 0.0
        )
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def underlying_booster(self) -> Any:
        """Return the raw tree booster used for SHAP attribution."""

        return self.estimator_

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _clone(self) -> "ReopenRiskClassifier":
        """A fresh, unfitted classifier with identical configuration."""

        return ReopenRiskClassifier(
            params=self.params,
            backend=self.backend,
            calibrate=self.calibrate,
            calibration_method=self.calibration_method,
            random_state=self.random_state,
            recall_floor=self.recall_floor,
            recall_margin=self.recall_margin,
            threshold_grid_size=self.threshold_grid_size,
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ReopenRiskClassifier":
        return joblib.load(Path(path))


# --------------------------------------------------------------------------- #
# End-to-end model: raw ticket frame -> risk
# --------------------------------------------------------------------------- #
@dataclass
class TicketAuditModel:
    """Feature pipeline + classifier, operating on raw ticket frames."""

    config: Optional[AuditConfig] = None
    classifier: ReopenRiskClassifier = field(default_factory=ReopenRiskClassifier)
    pipeline: TicketFeaturePipeline = field(default_factory=TicketFeaturePipeline)
    target_column: str = "is_premature_closure"
    trained_at: Optional[str] = None
    metrics_: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = get_config()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Optional[AuditConfig] = None) -> "TicketAuditModel":
        cfg = config or get_config()
        params = cfg.model_params or {}
        calibration = params.get("calibration", {}) or {}
        classifier = ReopenRiskClassifier(
            params=params,
            backend=params.get("backend", "auto"),
            calibrate=bool(calibration.get("method")),
            calibration_method=calibration.get("method", "isotonic"),
            random_state=cfg.project.random_seed,
            recall_floor=cfg.model.recall_floor,
            recall_margin=cfg.model.recall_margin,
            threshold_grid_size=cfg.model.threshold_grid_size,
        )
        numeric = cfg.features.numeric_features or list(DEFAULT_NUMERIC)
        categorical = cfg.features.categorical_features or list(DEFAULT_CATEGORICAL)
        pipeline = TicketFeaturePipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            min_active_minutes=cfg.features.min_active_minutes,
            caller_min_history=cfg.features.caller_min_history,
            random_state=cfg.project.random_seed,
        )
        return cls(
            config=cfg,
            classifier=classifier,
            pipeline=pipeline,
            target_column=cfg.model.target,
        )

    # ------------------------------------------------------------------ #
    def train(
        self,
        frame: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, float]:
        """Fit the whole stack and return training-set metrics.

        The decision threshold is chosen from **out-of-fold** predictions
        rather than in-sample ones. Optimising on the data the booster just
        memorised yields a threshold that collapses under deployment — the
        classic failure mode where a model looks perfect in training and
        misses half the real cases.
        """

        target = target_column or self.target_column
        if target not in frame.columns:
            raise KeyError(f"target column {target!r} missing from training frame")

        self.target_column = target
        y = frame[target].to_numpy(dtype=int)
        X = self.pipeline.fit_transform(frame, target_column=target)
        self.classifier.fit(X, y, feature_names=self.pipeline.feature_names)

        # --- out-of-fold probabilities for honest threshold selection ---
        oof_prob = self._out_of_fold_probabilities(frame, y, target)
        if oof_prob is not None:
            target_recall = min(
                1.0, self.classifier.recall_floor + self.classifier.recall_margin
            )
            threshold, stats = optimize_threshold(
                y, oof_prob, target_recall, self.classifier.threshold_grid_size
            )
            self.classifier.threshold_ = threshold
            self.classifier.threshold_stats_ = stats

        proba = self.classifier.predict_proba(X)[:, 1]
        preds = self.classifier.predict(X)
        self.metrics_ = {
            "train_rows": float(len(frame)),
            "train_positive_rate": float(y.mean()),
            "train_f1": float(f1_score(y, preds, zero_division=0)),
            "train_recall": float(recall_score(y, preds, zero_division=0)),
            "threshold": float(self.classifier.threshold_),
        }
        if self.classifier.threshold_stats_:
            self.metrics_.update(
                {
                    f"threshold_{k}": float(v)
                    for k, v in self.classifier.threshold_stats_.items()
                    if isinstance(v, (int, float))
                }
            )
        self.trained_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return self.metrics_

    def _out_of_fold_probabilities(
        self, frame: pd.DataFrame, y: np.ndarray, target: str, n_splits: int = 5
    ) -> Optional[np.ndarray]:
        """Cross-validated probabilities, or ``None`` when data is too small.

        The *whole* stack — feature engineer included — is refit inside each
        fold. Fitting the pipeline once on all rows would leak the
        per-caller reopen prior (a target-derived statistic) into the
        validation fold and flatter the threshold estimate.
        """

        if len(y) < n_splits * 4 or len(np.unique(y)) < 2:
            return None
        try:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.classifier.random_state
            )
            oof = np.zeros(len(y), dtype=float)
            for train_index, valid_index in splitter.split(frame, y):
                fold_pipeline = TicketFeaturePipeline(
                    numeric_features=list(self.pipeline.numeric_features),
                    categorical_features=list(self.pipeline.categorical_features),
                    min_active_minutes=self.pipeline.min_active_minutes,
                    caller_min_history=self.pipeline.caller_min_history,
                    random_state=self.pipeline.random_state,
                )
                X_train = fold_pipeline.fit_transform(
                    frame.iloc[train_index], target_column=target
                )
                X_valid = fold_pipeline.transform(frame.iloc[valid_index])
                fold_model = self.classifier._clone()
                fold_model.fit(X_train, y[train_index])
                oof[valid_index] = fold_model.predict_proba(X_valid)[:, 1]
            return oof
        except Exception as exc:  # pragma: no cover - numerical edge cases
            logger.warning("out-of-fold threshold selection failed (%s)", exc)
            return None

    # ------------------------------------------------------------------ #
    def risk_bands(self) -> Tuple[float, float]:
        """Resolve the ``(low_upper, elevated_upper)`` band boundaries.

        When the config leaves them unset they are derived from the tuned
        decision threshold, so the "flagged for review" line and the start
        of the ELEVATED band never disagree.
        """

        threshold = float(self.classifier.threshold_)
        low = self.config.dashboard.risk_low_upper
        elevated = self.config.dashboard.risk_elevated_upper
        if low is None:
            low = threshold
        if elevated is None:
            elevated = max(threshold * 2.5, low + 1e-6)
        return float(low), float(elevated)

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted probabilities, risk bands and hard flags."""

        X = self.pipeline.transform(frame)
        proba = self.classifier.predict_proba(X)[:, 1]
        low, elevated = self.risk_bands()
        result = frame.copy()
        result["reopen_probability"] = np.round(proba, 6)
        result["risk_level"] = [
            risk_level(p, low, elevated) for p in proba
        ]
        result["flagged_for_review"] = (proba >= self.classifier.threshold_).astype(int)
        return result

    def design_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(frame)

    # ------------------------------------------------------------------ #
    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else self.config.path("model_path")
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: Path) -> "TicketAuditModel":
        return joblib.load(Path(path))
