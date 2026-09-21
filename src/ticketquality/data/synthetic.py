"""Synthetic ITSM ticket generator.

The project's primary data source is the UCI *Incident Management Process
Enriched Event Log*. That download is roughly 40 MB and requires network
access, so the repository ships a fully self-contained synthetic generator
that reproduces the statistical *shape* of a real support desk: skewed
resolution durations, a minority of premature closures, and — crucially —
the correlated failure modes QA leads actually look for.

The generator is not uniform noise. Reopen probability is produced by a
logistic mechanism over latent operational drivers, which means:

* short active resolution time **combined with** a high reassignment count
  raises reopen probability dramatically (the headline failure mode);
* tickets whose priority diverges from impact/urgency, or that were
  modified frequently relative to their lifetime, are likelier to bounce;
* caller identity carries a persistent "contests closures" tendency.

Everything is seeded, so identical arguments produce identical frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_CATEGORIES: Sequence[str] = (
    "Hardware",
    "Software",
    "Network",
    "Database",
    "Access Management",
    "Email",
    "Security",
)
DEFAULT_GROUPS: Sequence[str] = (
    "Service Desk L1",
    "Service Desk L2",
    "Infrastructure",
    "Applications",
    "Network Ops",
    "Database Admin",
)
DEFAULT_CONTACT_TYPES: Sequence[str] = ("Self Service", "Phone", "Email", "Chat")
DEFAULT_SHIFTS: Sequence[str] = ("Morning", "Afternoon", "Night")


@dataclass
class SyntheticTicketGenerator:
    """Generate a reproducible frame of synthetic support tickets.

    Parameters
    ----------
    n_samples:
        Number of tickets to produce.
    seed:
        Seed for :class:`numpy.random.Generator`.
    reopen_intercept:
        Baseline log-odds of premature closure for a "neutral" ticket.
    """

    n_samples: int = 1000
    seed: int = 42
    reopen_intercept: float = -8.5

    categories: Sequence[str] = field(default_factory=lambda: DEFAULT_CATEGORIES)
    groups: Sequence[str] = field(default_factory=lambda: DEFAULT_GROUPS)
    contact_types: Sequence[str] = field(default_factory=lambda: DEFAULT_CONTACT_TYPES)
    shifts: Sequence[str] = field(default_factory=lambda: DEFAULT_SHIFTS)

    # Logistic weights for the latent reopen mechanism.
    w_reassignments: float = 2.11
    w_short_resolution: float = 3.91
    w_mismatch: float = 1.87
    w_mod_density: float = 1.36
    w_caller_bias: float = 3.06
    w_interaction: float = 0.61
    w_offshift: float = 0.85
    # Irreducible noise + a latent, *unobserved* driver. Without these the
    # target would be a deterministic function of the observed features and
    # the classifier would recover it perfectly (AUC 1.0), which is neither
    # realistic nor a useful demonstration.
    noise_scale: float = 0.80
    w_latent: float = 0.50

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _choice(self, options: Sequence[str], size: int) -> np.ndarray:
        return self._rng.choice(list(options), size=size)

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        """Return a ``pandas.DataFrame`` of synthetic tickets."""

        n = self.n_samples
        rng = self._rng

        # --- categorical attributes --------------------------------------
        priority = rng.choice(
            ["Low", "Moderate", "High", "Critical"], size=n, p=[0.30, 0.42, 0.20, 0.08]
        )
        impact = rng.choice(
            ["Low", "Medium", "High"], size=n, p=[0.45, 0.38, 0.17]
        )
        urgency = rng.choice(
            ["Low", "Medium", "High"], size=n, p=[0.42, 0.40, 0.18]
        )
        category = self._choice(self.categories, n)
        assignment_group = self._choice(self.groups, n)
        contact_type = self._choice(self.contact_types, n)
        shift = self._choice(self.shifts, n)

        # --- caller identity & historical tendency -----------------------
        n_callers = max(8, n // 12)
        caller_ids = rng.integers(1, n_callers + 1, size=n)
        caller_bias_map = rng.normal(0.0, 0.75, size=n_callers + 1)
        caller_bias = caller_bias_map[caller_ids]

        # --- operational counters ----------------------------------------
        # Reassignments: heavy right skew, some tickets bounce a lot.
        reassignment_count = rng.poisson(0.9, size=n)
        spike = rng.random(n) < 0.12
        reassignment_count = reassignment_count + (spike * rng.integers(2, 6, size=n))
        reassignment_count = reassignment_count.astype(int)

        # System modifications correlate with activity.
        sys_mod_count = rng.poisson(2.4, size=n) + reassignment_count
        sys_mod_count = sys_mod_count.astype(int)

        interaction_count = rng.poisson(3.1, size=n) + 1

        # --- durations (minutes) ------------------------------------------
        # Base resolution time depends on priority (Critical solved faster).
        priority_speed = {
            "Low": 1.55,
            "Moderate": 1.15,
            "High": 0.85,
            "Critical": 0.60,
        }
        speed_factor = np.array([priority_speed[p] for p in priority])
        base = rng.lognormal(mean=6.2, sigma=0.85, size=n) * speed_factor
        # Reassignments add handling overhead.
        resolution_minutes = base + reassignment_count * rng.uniform(
            12.0, 70.0, size=n
        )
        resolution_minutes = np.clip(resolution_minutes, 2.0, 60 * 24 * 30)

        handle_time = resolution_minutes * rng.uniform(0.35, 0.75, size=n)
        active_minutes = np.clip(
            resolution_minutes * rng.uniform(0.25, 0.60, size=n), 1.0, None
        )
        assignment_group_dwell = np.clip(
            resolution_minutes * rng.uniform(0.10, 0.45, size=n), 0.5, None
        )

        # --- derived / engineered signals ---------------------------------
        min_active = 1.0
        reassignment_rate = reassignment_count / np.maximum(active_minutes, min_active)
        modification_density = sys_mod_count / np.maximum(
            resolution_minutes, min_active
        )
        # caller reopen rate: a historical tendency, jittered around the bias
        caller_reopen_rate = np.clip(
            0.5 + 0.28 * np.tanh(caller_bias) + rng.normal(0, 0.05, size=n), 0.0, 1.0
        )

        impact_rank = np.array([{"Low": 0, "Medium": 1, "High": 2}[i] for i in impact])
        urgency_rank = np.array([{"Low": 0, "Medium": 1, "High": 2}[u] for u in urgency])
        priority_mismatch_index = np.abs(impact_rank - urgency_rank).astype(float)

        # --- latent reopen mechanism --------------------------------------
        # "Short resolution" indicator: relative to a 240-minute reference,
        # a quick close (under ~1/2 the median) is the premature-closure tell.
        median_res = float(np.median(resolution_minutes))
        short_resolution = np.clip(
            (0.5 * median_res - resolution_minutes) / (0.5 * median_res), -1.0, 1.0
        )
        # Combined failure mode: quick close *and* many handoffs.
        combined = short_resolution * np.clip(reassignment_count / 3.0, 0.0, 2.0)

        offshift = np.isin(shift, ["Night"]).astype(float)

        # Latent, unobserved driver: a customer-frustration signal that is
        # *not* exposed as a feature. It makes the target genuinely
        # unpredictable from the observed columns alone.
        latent = rng.normal(0.0, 1.0, size=n)

        logit = (
            self.reopen_intercept
            + self.w_reassignments * np.clip(reassignment_count, 0, 6)
            + self.w_short_resolution * short_resolution
            + self.w_mismatch * priority_mismatch_index
            + self.w_mod_density * np.clip(modification_density * 60.0, 0, 4)
            + self.w_caller_bias * caller_bias
            + self.w_interaction * interaction_count
            + self.w_offshift * offshift
            + 3.06 * combined
            + self.w_latent * latent
            + rng.normal(0.0, self.noise_scale, size=n)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        reopen_count = rng.binomial(1, prob, size=n)
        # A slice of reopened tickets reopen more than once.
        reopen_count = reopen_count * rng.integers(1, 3, size=n)
        reopen_count = reopen_count.astype(int)

        # --- timestamps ----------------------------------------------------
        opened_at = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 60 * 24 * 180, size=n), unit="m"
        )
        resolved_at = opened_at + pd.to_timedelta(resolution_minutes, unit="m")

        made_sla = (
            resolution_minutes
            < np.array(
                [{"Low": 2880, "Moderate": 1440, "High": 480, "Critical": 240}[p] for p in priority]
            )
        )

        frame = pd.DataFrame(
            {
                "ticket_id": [f"INC{100000 + i:06d}" for i in range(n)],
                "opened_at": opened_at,
                "resolved_at": resolved_at,
                "incident_state": "Closed",
                "priority": priority,
                "impact": impact,
                "urgency": urgency,
                "category": category,
                "assignment_group": assignment_group,
                "contact_type": contact_type,
                "shift": shift,
                "caller_id": [f"USR{cid:05d}" for cid in caller_ids],
                "made_sla": made_sla,
                "reassignment_count": reassignment_count,
                "reopen_count": reopen_count,
                "sys_mod_count": sys_mod_count,
                "interaction_count": interaction_count,
                "handle_time": np.round(handle_time, 2),
                "resolution_minutes": np.round(resolution_minutes, 2),
                "active_minutes": np.round(active_minutes, 2),
                "assignment_group_dwell": np.round(assignment_group_dwell, 2),
                "reassignment_rate": np.round(reassignment_rate, 6),
                "modification_density": np.round(modification_density, 6),
                "caller_reopen_rate": np.round(caller_reopen_rate, 4),
                "priority_mismatch_index": priority_mismatch_index,
                # ground-truth target
                "is_premature_closure": (reopen_count >= 1).astype(int),
            }
        )
        return frame


def generate_synthetic_tickets(
    n_samples: int = 1000, seed: int = 42, **kwargs
) -> pd.DataFrame:
    """Convenience wrapper around :class:`SyntheticTicketGenerator`."""

    return SyntheticTicketGenerator(n_samples=n_samples, seed=seed, **kwargs).generate()


def write_synthetic_dataset(
    destination,
    n_samples: int = 1000,
    seed: int = 42,
) -> "pd.DataFrame":
    """Generate tickets and persist them to ``destination`` (CSV)."""

    from pathlib import Path

    frame = generate_synthetic_tickets(n_samples=n_samples, seed=seed)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return frame


__all__ = [
    "SyntheticTicketGenerator",
    "generate_synthetic_tickets",
    "write_synthetic_dataset",
    "DEFAULT_CATEGORIES",
    "DEFAULT_GROUPS",
]
