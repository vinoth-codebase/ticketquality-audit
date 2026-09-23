"""Scikit-learn preprocessing pipeline for the reopen-risk model.

Two things make ticket data awkward to feed a gradient-boosted tree:

* the duration columns are heavily right-skewed (a handful of incidents
  stay open for weeks), and
* the categorical columns are moderately high-cardinality (assignment
  groups, categories).

:func:`build_preprocessing_pipeline` therefore assembles a
:class:`~sklearn.compose.ColumnTransformer` with a quantile branch for the
skewed durations, a scaling branch for the bounded counters, and a
one-hot branch for the categoricals. :class:`TicketFeaturePipeline` wraps
that together with the :class:`~ticketquality.features.engineer.FeatureEngineer`
so a single ``fit``/``transform`` call goes from raw ticket rows to a
model-ready matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
)

from .engineer import ENGINEERED_FEATURES, FeatureEngineer

__all__ = [
    "TicketFeaturePipeline",
    "build_preprocessing_pipeline",
    "select_feature_columns",
    "FORBIDDEN_FEATURES",
    "SKEWED_DURATION_COLUMNS",
]

# Columns whose distribution is long-tailed and benefit from quantile
# normalisation rather than z-scoring.
SKEWED_DURATION_COLUMNS: List[str] = [
    "resolution_minutes",
    "handle_time",
    "active_minutes",
    "resolution_hours",
    "active_hours",
    "assignment_group_dwell",
    "reassignment_rate",
    "modification_density",
    "speed_per_handoff",
]

DEFAULT_NUMERIC = [
    "reassignment_count",
    "sys_mod_count",
    "interaction_count",
    "caller_reopen_rate",
    "priority_mismatch_index",
    "handoff_intensity",
    "instant_close_flag",
    "combined_failure_mode",
]

DEFAULT_CATEGORICAL = [
    "priority",
    "impact",
    "urgency",
    "category",
    "assignment_group",
    "contact_type",
    "shift",
    "made_sla",
]

# Columns that must never reach the model: the target itself, its direct
# antecedents, and free identifiers. Enforced in ``TicketFeaturePipeline``
# so a config edit cannot silently reintroduce target leakage.
FORBIDDEN_FEATURES = frozenset(
    {
        "is_premature_closure",
        "reopen_count",
        "target",
        "label",
        "ticket_id",
        "caller_id",
        "opened_at",
        "resolved_at",
        "incident_state",
    }
)


def _drop_forbidden(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c not in FORBIDDEN_FEATURES]


def select_feature_columns(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> tuple[List[str], List[str]]:
    """Intersect requested feature columns with what ``frame`` actually has."""

    numeric_present = [
        c for c in _drop_forbidden(numeric) if c in frame.columns
    ]
    categorical_present = [
        c for c in _drop_forbidden(categorical) if c in frame.columns
    ]
    return numeric_present, categorical_present


def build_preprocessing_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    skewed_features: Optional[Sequence[str]] = None,
    n_quantiles: int = 500,
    random_state: int = 42,
) -> ColumnTransformer:
    """Assemble the column transformer.

    Numeric features are split into a quantile branch (skewed durations)
    and a standard-scaling branch (bounded counters). Categorical features
    are one-hot encoded with ``handle_unknown="ignore"`` so an unseen
    assignment group at inference does not crash the pipeline.
    """

    skewed = [c for c in (skewed_features or SKEWED_DURATION_COLUMNS) if c in numeric_features]
    bounded = [c for c in numeric_features if c not in skewed]

    transformers = []
    if skewed:
        transformers.append(
            (
                "skewed",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "quantile",
                            QuantileTransformer(
                                n_quantiles=min(n_quantiles, 1000),
                                output_distribution="normal",
                                subsample=100_000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
                skewed,
            )
        )
    if bounded:
        transformers.append(
            (
                "bounded",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                bounded,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                min_frequency=None,
                            ),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class TicketFeaturePipeline:
    """End-to-end raw-ticket -> model-matrix pipeline.

    Composition:
        FeatureEngineer (operational velocity) -> ColumnTransformer (scale/encode)
    """

    numeric_features: Sequence[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC))
    categorical_features: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_CATEGORICAL)
    )
    min_active_minutes: float = 1.0
    caller_min_history: int = 2
    random_state: int = 42

    def __post_init__(self) -> None:
        self.engineer = FeatureEngineer(
            min_active_minutes=self.min_active_minutes,
            caller_min_history=self.caller_min_history,
        )
        self.transformer: Optional[ColumnTransformer] = None
        self.numeric_used_: List[str] = []
        self.categorical_used_: List[str] = []

    # ------------------------------------------------------------------ #
    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.engineer.transform(frame)

    def fit(
        self,
        frame: pd.DataFrame,
        target_column: str = "is_premature_closure",
    ) -> "TicketFeaturePipeline":
        """Fit the engineer and the column transformer."""

        self.engineer.fit(frame, target_column=target_column)
        engineered = self._engineer(frame)

        numeric = list(self.numeric_features) + [
            c for c in ENGINEERED_FEATURES if c not in self.numeric_features
        ]
        numeric_present, categorical_present = select_feature_columns(
            engineered, numeric, self.categorical_features
        )
        self.numeric_used_ = numeric_present
        self.categorical_used_ = categorical_present

        self.transformer = build_preprocessing_pipeline(
            numeric_present,
            categorical_present,
            random_state=self.random_state,
        )
        self.transformer.fit(engineered)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the model-ready design matrix."""

        if self.transformer is None:
            raise RuntimeError("TicketFeaturePipeline must be fitted before transform()")
        engineered = self._engineer(frame)
        matrix = self.transformer.transform(engineered)
        return np.asarray(matrix, dtype=float)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        target_column: str = "is_premature_closure",
    ) -> np.ndarray:
        self.fit(frame, target_column=target_column)
        return self.transform(frame)

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> List[str]:
        """Human-readable names for every column of the design matrix."""

        if self.transformer is None:
            return []
        names = list(self.transformer.get_feature_names_out())
        return [str(n) for n in names]

    @property
    def raw_feature_names(self) -> List[str]:
        """Names of the columns *before* the column transformer."""

        return list(self.numeric_used_) + list(self.categorical_used_)
