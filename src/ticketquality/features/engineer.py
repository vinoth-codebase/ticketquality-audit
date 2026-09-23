"""Operational velocity feature engineering.

Support quality is not a function of *how long* a ticket took, but of how
the resolution was *conducted*. A ticket closed in four minutes after
three reassignments is a very different object from one closed in four
minutes with a single clean touch — yet both look identical to a naive
"resolution time" model.

The :class:`FeatureEngineer` derives the operational-velocity metrics that
separate those cases:

``reassignment_rate``
    Handoffs per active resolution hour. High values mean the ticket
    ping-ponged between queues.
``modification_density``
    System update actions (``sys_mod_count``) per lifecycle hour — a proxy
    for churn and last-minute edits.
``caller_reopen_rate``
    A caller's historical tendency to contest closures. Fitted on the
    training split only, so it is a genuine prior rather than a leak.
``assignment_group_dwell``
    Hours spent parked in intermediate triage queues before resolution.
``priority_mismatch_index``
    Divergence between the recorded impact and urgency ranks — the classic
    sign of a mis-triaged ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "FeatureEngineer",
    "ENGINEERED_FEATURES",
    "compute_reassignment_rate",
    "compute_modification_density",
    "compute_priority_mismatch_index",
    "compute_combined_failure_mode",
]

# The full set of derived columns produced by :meth:`FeatureEngineer.transform`.
ENGINEERED_FEATURES: List[str] = [
    "reassignment_rate",
    "modification_density",
    "caller_reopen_rate",
    "assignment_group_dwell",
    "priority_mismatch_index",
    "resolution_hours",
    "active_hours",
    "handoff_intensity",
    "instant_close_flag",
    "speed_per_handoff",
    "combined_failure_mode",
]

_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
_URGENCY_RANK = {"low": 1, "medium": 2, "high": 3}


# --------------------------------------------------------------------------- #
# Standalone metric functions (easy to unit-test in isolation)
# --------------------------------------------------------------------------- #
def compute_reassignment_rate(
    reassignment_count: pd.Series, active_minutes: pd.Series, min_active_minutes: float = 1.0
) -> pd.Series:
    """Handoffs per active resolution **hour**.

    ``active_minutes`` is floored at ``min_active_minutes`` so instant
    closures do not produce infinities.
    """

    active_hours = np.maximum(active_minutes.astype(float), min_active_minutes) / 60.0
    return reassignment_count.astype(float) / active_hours


def compute_modification_density(
    sys_mod_count: pd.Series, resolution_minutes: pd.Series, min_active_minutes: float = 1.0
) -> pd.Series:
    """System update actions per lifecycle **hour**."""

    hours = np.maximum(resolution_minutes.astype(float), min_active_minutes) / 60.0
    return sys_mod_count.astype(float) / hours


def compute_priority_mismatch_index(
    impact: pd.Series, urgency: pd.Series
) -> pd.Series:
    """Absolute rank gap between impact and urgency (0, 1 or 2)."""

    impact_rank = impact.astype("string").str.strip().str.lower().map(_IMPACT_RANK)
    urgency_rank = urgency.astype("string").str.strip().str.lower().map(_URGENCY_RANK)
    diff = (impact_rank.fillna(2) - urgency_rank.fillna(2)).abs()
    return diff.astype(float)


def compute_combined_failure_mode(
    resolution_minutes: pd.Series,
    reassignment_count: pd.Series,
    reference_minutes: float = 120.0,
) -> pd.Series:
    """Interaction term: quick close *with* many handoffs.

    This is the single strongest reopen signal in real support data, and
    giving the model the interaction explicitly (rather than hoping a tree
    discovers it) sharpens the learned boundary.
    """

    quick = np.clip(
        (reference_minutes - resolution_minutes.astype(float)) / reference_minutes,
        -1.0,
        1.0,
    )
    handoffs = np.clip(reassignment_count.astype(float) / 3.0, 0.0, 2.0)
    return quick * handoffs


# --------------------------------------------------------------------------- #
# Fitted transformer
# --------------------------------------------------------------------------- #
@dataclass
class FeatureEngineer:
    """Derive operational-velocity features from cleaned ticket frames.

    Follows a fit/transform contract so the caller-history prior is learned
    on training data and merely *applied* at inference time.
    """

    min_active_minutes: float = 1.0
    caller_min_history: int = 2
    reference_minutes: float = 120.0

    # learned state
    caller_reopen_rates_: Dict[str, float] = field(default_factory=dict)
    global_reopen_rate_: float = 0.0
    is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    def fit(
        self,
        frame: pd.DataFrame,
        target_column: str = "is_premature_closure",
        caller_column: str = "caller_id",
    ) -> "FeatureEngineer":
        """Learn per-caller reopen priors from a labelled training frame."""

        if target_column not in frame.columns:
            raise KeyError(
                f"target column {target_column!r} required to fit caller history"
            )

        target = pd.to_numeric(frame[target_column], errors="coerce").fillna(0)
        self.global_reopen_rate_ = float(target.mean())

        if caller_column in frame.columns:
            grouped = target.groupby(frame[caller_column].astype("string"))
            counts = grouped.size()
            means = grouped.mean()
            # Only trust callers with enough history.
            eligible = counts[counts >= self.caller_min_history].index
            self.caller_reopen_rates_ = {
                str(caller): float(means.loc[caller]) for caller in eligible
            }
        else:  # pragma: no cover - defensive
            self.caller_reopen_rates_ = {}

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    def transform(
        self,
        frame: pd.DataFrame,
        caller_column: str = "caller_id",
    ) -> pd.DataFrame:
        """Return ``frame`` with all engineered columns attached."""

        if not self.is_fitted_:
            # Transform-only use (e.g. single-ticket inference) still works:
            # fall back to a neutral prior.
            self.global_reopen_rate_ = self.global_reopen_rate_ or 0.25

        work = frame.copy()

        reassignment = pd.to_numeric(
            work.get("reassignment_count", pd.Series(0, index=work.index)),
            errors="coerce",
        ).fillna(0)
        sys_mod = pd.to_numeric(
            work.get("sys_mod_count", pd.Series(0, index=work.index)), errors="coerce"
        ).fillna(0)
        resolution = pd.to_numeric(
            work.get("resolution_minutes", pd.Series(60.0, index=work.index)),
            errors="coerce",
        ).fillna(60.0)
        active = pd.to_numeric(
            work.get("active_minutes", resolution * 0.45), errors="coerce"
        ).fillna(30.0)
        interaction = pd.to_numeric(
            work.get("interaction_count", pd.Series(1, index=work.index)), errors="coerce"
        ).fillna(1)

        # --- core velocity metrics -------------------------------------
        work["reassignment_rate"] = compute_reassignment_rate(
            reassignment, active, self.min_active_minutes
        )
        work["modification_density"] = compute_modification_density(
            sys_mod, resolution, self.min_active_minutes
        )

        # --- dwell time (hours) -----------------------------------------
        if "assignment_group_dwell" in work.columns:
            dwell = pd.to_numeric(work["assignment_group_dwell"], errors="coerce")
            work["assignment_group_dwell"] = (
                dwell.fillna(resolution * 0.25).clip(lower=0.0) / 60.0
            )
        else:
            work["assignment_group_dwell"] = (resolution * 0.25 / 60.0).clip(lower=0.0)

        # --- priority mismatch -------------------------------------------
        if "impact" in work.columns and "urgency" in work.columns:
            work["priority_mismatch_index"] = compute_priority_mismatch_index(
                work["impact"], work["urgency"]
            )
        else:
            work["priority_mismatch_index"] = 0.0

        # --- caller history ----------------------------------------------
        if caller_column in work.columns:
            callers = work[caller_column].astype("string")
            work["caller_reopen_rate"] = (
                callers.map(self.caller_reopen_rates_)
                .astype(float)
                .fillna(self.global_reopen_rate_)
            )
        else:
            work["caller_reopen_rate"] = self.global_reopen_rate_

        # --- convenience / interaction terms -----------------------------
        work["resolution_hours"] = resolution / 60.0
        work["active_hours"] = np.maximum(active, self.min_active_minutes) / 60.0
        work["handoff_intensity"] = reassignment / np.maximum(interaction, 1.0)
        work["instant_close_flag"] = (resolution < 30.0).astype(int)
        work["speed_per_handoff"] = work["resolution_hours"] / (reassignment + 1.0)
        work["combined_failure_mode"] = compute_combined_failure_mode(
            resolution, reassignment, self.reference_minutes
        )

        # round for readability
        for column in ENGINEERED_FEATURES:
            work[column] = pd.to_numeric(work[column], errors="coerce").astype(float).round(6)

        return work

    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        frame: pd.DataFrame,
        target_column: str = "is_premature_closure",
        caller_column: str = "caller_id",
    ) -> pd.DataFrame:
        """Fit on ``frame`` and return its transformed copy."""

        self.fit(frame, target_column=target_column, caller_column=caller_column)
        return self.transform(frame, caller_column=caller_column)
