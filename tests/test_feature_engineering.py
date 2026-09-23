"""Tests for operational-velocity feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticketquality.features.engineer import (
    ENGINEERED_FEATURES,
    FeatureEngineer,
    compute_combined_failure_mode,
    compute_modification_density,
    compute_priority_mismatch_index,
    compute_reassignment_rate,
)
from ticketquality.features.transformers import (
    DEFAULT_CATEGORICAL,
    DEFAULT_NUMERIC,
    FORBIDDEN_FEATURES,
    SKEWED_DURATION_COLUMNS,
    TicketFeaturePipeline,
    build_preprocessing_pipeline,
    select_feature_columns,
)

# --------------------------------------------------------------------------- #
# Standalone metric functions
# --------------------------------------------------------------------------- #
def test_reassignment_rate_is_per_active_hour():
    rate = compute_reassignment_rate(
        pd.Series([4.0]), pd.Series([120.0])  # 4 handoffs in 2 hours
    )
    assert rate.iloc[0] == pytest.approx(2.0)


def test_reassignment_rate_guards_against_zero_duration():
    rate = compute_reassignment_rate(pd.Series([3.0]), pd.Series([0.0]))
    assert np.isfinite(rate.iloc[0])
    assert rate.iloc[0] == pytest.approx(3.0 * 60.0)


def test_modification_density_is_per_lifecycle_hour():
    density = compute_modification_density(pd.Series([6.0]), pd.Series([180.0]))
    assert density.iloc[0] == pytest.approx(2.0)


def test_priority_mismatch_index_magnitude():
    index = compute_priority_mismatch_index(
        pd.Series(["Low", "High", "Medium", "HIGH"]),
        pd.Series(["High", "High", "Medium", "low"]),
    )
    assert index.tolist() == [2.0, 0.0, 0.0, 2.0]


def test_priority_mismatch_handles_unknown_values():
    index = compute_priority_mismatch_index(pd.Series(["weird"]), pd.Series(["Medium"]))
    assert index.iloc[0] == 0.0


def test_combined_failure_mode_is_positive_for_quick_close_with_handoffs():
    value = compute_combined_failure_mode(pd.Series([5.0]), pd.Series([5.0]))
    assert value.iloc[0] > 0

    slow = compute_combined_failure_mode(pd.Series([900.0]), pd.Series([0.0]))
    assert slow.iloc[0] == 0.0


# --------------------------------------------------------------------------- #
# FeatureEngineer
# --------------------------------------------------------------------------- #
def test_engineer_fit_learns_caller_history(ticket_frame):
    engineer = FeatureEngineer(caller_min_history=2).fit(ticket_frame)
    assert engineer.is_fitted_
    assert 0.0 <= engineer.global_reopen_rate_ <= 1.0
    assert isinstance(engineer.caller_reopen_rates_, dict)


def test_engineer_transform_produces_all_engineered_columns(ticket_frame):
    engineer = FeatureEngineer().fit(ticket_frame)
    transformed = engineer.transform(ticket_frame)
    for column in ENGINEERED_FEATURES:
        assert column in transformed.columns, column
    assert transformed[ENGINEERED_FEATURES].notna().all().all()


def test_engineer_fit_requires_target_column(ticket_frame):
    with pytest.raises(KeyError):
        FeatureEngineer().fit(ticket_frame.drop(columns=["is_premature_closure"]))


def test_engineer_transform_without_fit_still_works(small_frame):
    transformed = FeatureEngineer().transform(small_frame)
    assert "reassignment_rate" in transformed.columns
    assert transformed["caller_reopen_rate"].notna().all()


def test_engineer_dwell_is_converted_to_hours(ticket_frame):
    engineer = FeatureEngineer().fit(ticket_frame)
    transformed = engineer.transform(ticket_frame)
    # Synthetic dwell is stored in minutes; hours must be smaller.
    assert (
        transformed["assignment_group_dwell"] <= transformed["resolution_hours"] * 1.5
    ).all()


def test_engineer_unknown_caller_falls_back_to_global_rate(ticket_frame):
    engineer = FeatureEngineer().fit(ticket_frame)
    unseen = ticket_frame.copy()
    unseen["caller_id"] = "BRAND-NEW-CALLER"
    transformed = engineer.transform(unseen)
    assert (transformed["caller_reopen_rate"] == engineer.global_reopen_rate_).all()


def test_engineer_handles_frame_without_caller_column(ticket_frame):
    engineer = FeatureEngineer().fit(ticket_frame)
    without = ticket_frame.drop(columns=["caller_id"])
    transformed = engineer.transform(without)
    assert "caller_reopen_rate" in transformed.columns


def test_engineer_fit_transform_matches_fit_then_transform(ticket_frame):
    a = FeatureEngineer().fit_transform(ticket_frame)
    engineer = FeatureEngineer().fit(ticket_frame)
    b = engineer.transform(ticket_frame)
    pd.testing.assert_series_equal(a["reassignment_rate"], b["reassignment_rate"])


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #
def test_select_feature_columns_intersects_with_frame(ticket_frame):
    numeric, categorical = select_feature_columns(
        ticket_frame, ["reassignment_count", "not_a_column"], ["priority", "nope"]
    )
    assert numeric == ["reassignment_count"]
    assert categorical == ["priority"]


def test_select_feature_columns_drops_leakage_columns(ticket_frame):
    numeric, categorical = select_feature_columns(
        ticket_frame,
        ["reassignment_count", "reopen_count", "is_premature_closure"],
        ["priority"],
    )
    assert "reopen_count" not in numeric
    assert "is_premature_closure" not in numeric
    for forbidden in FORBIDDEN_FEATURES:
        assert forbidden not in numeric


def test_build_preprocessing_pipeline_shapes(ticket_frame):
    numeric = ["resolution_minutes", "reassignment_count", "sys_mod_count"]
    categorical = ["priority", "category"]
    pipeline = build_preprocessing_pipeline(numeric, categorical)
    matrix = pipeline.fit_transform(ticket_frame)
    assert matrix.shape[0] == len(ticket_frame)
    assert matrix.shape[1] > len(numeric)


def test_pipeline_handles_unseen_category_at_transform_time(ticket_frame):
    pipeline = TicketFeaturePipeline()
    pipeline.fit(ticket_frame)
    unseen = ticket_frame.copy()
    unseen["assignment_group"] = "A GROUP THAT NEVER EXISTED"
    unseen["category"] = "NOVEL CATEGORY"
    matrix = pipeline.transform(unseen)
    assert matrix.shape[0] == len(unseen)
    assert np.isfinite(matrix).all()


def test_pipeline_fit_transform_returns_design_matrix(ticket_frame):
    pipeline = TicketFeaturePipeline()
    matrix = pipeline.fit_transform(ticket_frame)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == len(ticket_frame)
    assert matrix.shape[1] == len(pipeline.feature_names)


def test_pipeline_transform_before_fit_raises(ticket_frame):
    with pytest.raises(RuntimeError):
        TicketFeaturePipeline().transform(ticket_frame)


def test_pipeline_feature_names_before_fit_is_empty():
    assert TicketFeaturePipeline().feature_names == []


def test_pipeline_raw_feature_names_after_fit(ticket_frame):
    pipeline = TicketFeaturePipeline().fit(ticket_frame)
    assert pipeline.raw_feature_names
    assert set(pipeline.raw_feature_names).isdisjoint(FORBIDDEN_FEATURES)


def test_pipeline_never_includes_forbidden_features(ticket_frame):
    pipeline = TicketFeaturePipeline().fit(ticket_frame)
    for name in pipeline.raw_feature_names:
        assert name not in FORBIDDEN_FEATURES


def test_pipeline_accepts_custom_feature_lists(ticket_frame):
    pipeline = TicketFeaturePipeline(
        numeric_features=["reassignment_count", "sys_mod_count"],
        categorical_features=["priority"],
    ).fit(ticket_frame)
    assert "reassignment_count" in pipeline.raw_feature_names
    assert "priority" in pipeline.raw_feature_names


def test_default_feature_lists_are_non_empty():
    assert DEFAULT_NUMERIC
    assert DEFAULT_CATEGORICAL
    assert SKEWED_DURATION_COLUMNS
