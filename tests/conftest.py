"""Shared pytest fixtures.

The heavy fixtures (dataset, trained model) are session-scoped: training a
gradient-boosted model is the slow part of this suite, and re-fitting it
per test would push the run past any reasonable budget. The model is
trained on a deliberately small frame with a reduced booster so the whole
suite stays fast while still exercising the real code paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticketquality.config import load_config
from ticketquality.data.preprocessor import TicketPreprocessor
from ticketquality.data.synthetic import generate_synthetic_tickets


@pytest.fixture(scope="session")
def base_config():
    """The project's real configuration."""

    return load_config()


@pytest.fixture(scope="session")
def fast_config(base_config):
    """A copy of the config with a small, fast booster."""

    config = base_config.model_copy(deep=True)
    config.model_params = {
        "backend": "auto",
        "lightgbm": {
            "objective": "binary",
            "n_estimators": 60,
            "learning_rate": 0.08,
            "num_leaves": 12,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "verbosity": -1,
        },
        "sklearn_fallback": {
            "max_iter": 80,
            "learning_rate": 0.08,
            "max_leaf_nodes": 12,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
        "calibration": {"method": "sigmoid", "cv": 3},
    }
    config.model.recall_margin = 0.05
    return config


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """A small synthetic ticket frame (unprocessed)."""

    return generate_synthetic_tickets(400, seed=11)


@pytest.fixture(scope="session")
def ticket_frame(raw_frame) -> pd.DataFrame:
    """A cleaned synthetic ticket frame."""

    return TicketPreprocessor().clean(raw_frame)


@pytest.fixture(scope="session")
def small_frame() -> pd.DataFrame:
    """A very small frame for cheap unit tests."""

    return generate_synthetic_tickets(60, seed=3)


@pytest.fixture(scope="session")
def trained_model(fast_config, ticket_frame):
    """A fitted :class:`TicketAuditModel` shared across the suite."""

    from ticketquality.models.classifier import TicketAuditModel

    model = TicketAuditModel.from_config(fast_config)
    model.train(ticket_frame)
    return model


@pytest.fixture()
def event_log() -> pd.DataFrame:
    """A miniature multi-state incident event log."""

    return pd.DataFrame(
        {
            "number": ["INC1", "INC1", "INC1", "INC2", "INC2", "INC3", "INC4"],
            "incident_state": [
                "New",
                "In Progress",
                "Closed",
                "New",
                "Resolved",
                "Closed",
                "In Progress",
            ],
            "sys_updated_at": [
                "2025-01-01 09:00",
                "2025-01-01 10:00",
                "2025-01-01 11:00",
                "2025-01-02 09:00",
                "2025-01-02 12:00",
                "2025-01-03 09:00",
                "2025-01-04 09:00",
            ],
            "reopen_count": [0, 0, 1, 0, 0, 2, 0],
            "reassignment_count": [0, 2, 2, 0, 1, 3, 1],
            "sys_mod_count": [1, 3, 3, 0, 2, 5, 1],
        }
    )


@pytest.fixture(scope="session")
def confusion_pairs():
    """A handful of (y_true, y_prob) pairs for metric unit tests."""

    rng = np.random.default_rng(0)
    y_true = np.array([0] * 80 + [1] * 20)
    y_prob = np.concatenate(
        [rng.uniform(0.0, 0.5, size=80), rng.uniform(0.4, 1.0, size=20)]
    )
    return y_true, y_prob
