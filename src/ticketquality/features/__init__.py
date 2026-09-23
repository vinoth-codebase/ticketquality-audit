"""Operational-velocity feature engineering."""

from .engineer import (
    ENGINEERED_FEATURES,
    FeatureEngineer,
    compute_combined_failure_mode,
    compute_modification_density,
    compute_priority_mismatch_index,
    compute_reassignment_rate,
)
from .transformers import (
    DEFAULT_CATEGORICAL,
    DEFAULT_NUMERIC,
    FORBIDDEN_FEATURES,
    SKEWED_DURATION_COLUMNS,
    TicketFeaturePipeline,
    build_preprocessing_pipeline,
    select_feature_columns,
)

__all__ = [
    "ENGINEERED_FEATURES",
    "FeatureEngineer",
    "compute_combined_failure_mode",
    "compute_modification_density",
    "compute_priority_mismatch_index",
    "compute_reassignment_rate",
    "DEFAULT_CATEGORICAL",
    "DEFAULT_NUMERIC",
    "FORBIDDEN_FEATURES",
    "SKEWED_DURATION_COLUMNS",
    "TicketFeaturePipeline",
    "build_preprocessing_pipeline",
    "select_feature_columns",
]
