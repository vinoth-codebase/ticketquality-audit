"""Configuration loading for TicketQuality-Audit.

The project ships two YAML documents:

``config/config.yaml``
    Master configuration covering paths, dataset locations, feature
    definitions and model/evaluation knobs.

``config/model_params.yaml``
    Estimator hyper-parameters for the LightGBM backend and the
    scikit-learn fallback.

Both are validated through Pydantic models so a malformed config fails
loudly at load time rather than midway through a training run.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Locating the project root
# --------------------------------------------------------------------------- #

# src/ticketquality/config.py -> src/ticketquality -> src -> <project root>
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"

ENV_CONFIG_PATH = "TICKETQUALITY_CONFIG"
ENV_PARAMS_PATH = "TICKETQUALITY_PARAMS"


class ProjectMeta(BaseModel):
    """Identity block for the project."""

    name: str = "ticketquality-audit"
    version: str = "1.0.0"
    random_seed: int = 42


class PathConfig(BaseModel):
    """Filesystem locations used across the pipeline."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    model_path: str = "artifacts/reopen_risk_model.joblib"
    metrics_path: str = "artifacts/model_metrics.json"
    examples_dir: str = "examples"

    def resolve(self, relative: str, root: Optional[Path] = None) -> Path:
        """Resolve ``relative`` against the project root (or ``root``)."""
        base = root or PROJECT_ROOT
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        return (base / candidate).resolve()


class DataConfig(BaseModel):
    """Ingestion parameters for the UCI ITSM dataset and synthetic fallback."""

    uci_dataset_id: int = 498
    uci_dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/498/"
        "incident+management+process+enriched+event+log.zip"
    )
    archive_name: str = "incident_management_process_enriched_event_log.zip"
    event_log_file: str = "Incident_event_log.csv"
    incident_file: str = "Incident.csv"
    terminal_states: List[str] = Field(default_factory=lambda: ["Resolved", "Closed"])
    triage_states: List[str] = Field(
        default_factory=lambda: ["New", "Assigned", "In Progress"]
    )
    synthetic_samples: int = 1000

    @field_validator("terminal_states", "triage_states")
    @classmethod
    def _non_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("state list must not be empty")
        return value


class FeatureConfig(BaseModel):
    """Feature-engineering knobs."""

    min_active_minutes: float = 1.0
    caller_min_history: int = 2
    high_cardinality_threshold: int = 10
    numeric_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Training / evaluation settings for the classifier."""

    target: str = "is_premature_closure"
    test_size: float = 0.25
    validation_size: float = 0.2
    recall_floor: float = 0.80
    # Head-room added to the recall floor when *selecting* the operating
    # threshold on out-of-fold predictions. The floor is a deployment
    # guarantee, but it is estimated on a finite sample; targeting a small
    # margin above it absorbs that sampling noise so the hold-out recall
    # reliably clears the floor instead of straddling it.
    recall_margin: float = 0.10
    threshold_grid_size: int = 99
    params_file: str = "config/model_params.yaml"

    @field_validator("test_size", "validation_size")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("split fraction must lie strictly in (0, 1)")
        return value

    @field_validator("recall_floor")
    @classmethod
    def _valid_recall(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("recall_floor must lie in (0, 1]")
        return value

    @field_validator("recall_margin")
    @classmethod
    def _valid_margin(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("recall_margin must lie in [0, 1)")
        return value


class DashboardConfig(BaseModel):
    """Presentation thresholds for the Streamlit UI."""

    page_title: str = "TicketQuality-Audit"
    # Absolute band boundaries. Leave unset (null) to derive them from the
    # model's tuned operating threshold, which keeps the "flagged for
    # review" line and the ELEVATED band consistent with each other.
    risk_low_upper: Optional[float] = None
    risk_elevated_upper: Optional[float] = None


class AuditConfig(BaseModel):
    """Root configuration object."""

    project: ProjectMeta = Field(default_factory=ProjectMeta)
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    # Populated separately from model_params.yaml
    model_params: Dict[str, Any] = Field(default_factory=dict)

    def path(self, key: str) -> Path:
        """Return an absolute path for a named entry of ``paths``."""
        try:
            relative = getattr(self.paths, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown path key: {key}") from exc
        return self.paths.resolve(relative)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def load_config(
    config_path: Optional[os.PathLike | str] = None,
    params_path: Optional[os.PathLike | str] = None,
) -> AuditConfig:
    """Load and validate the project configuration.

    Parameters
    ----------
    config_path:
        Override for ``config/config.yaml``. Falls back to the
        ``TICKETQUALITY_CONFIG`` environment variable, then the packaged
        default.
    params_path:
        Override for ``config/model_params.yaml``.
    """

    resolved_config = Path(
        config_path or os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH
    )
    resolved_params = Path(
        params_path or os.environ.get(ENV_PARAMS_PATH) or DEFAULT_PARAMS_PATH
    )

    raw = _read_yaml(resolved_config)
    config = AuditConfig(**raw)

    if resolved_params.exists():
        config.model_params = _read_yaml(resolved_params)
    elif DEFAULT_PARAMS_PATH.exists() and resolved_params != DEFAULT_PARAMS_PATH:
        config.model_params = _read_yaml(DEFAULT_PARAMS_PATH)

    return config


@lru_cache(maxsize=1)
def get_config() -> AuditConfig:
    """Cached accessor for the default configuration."""
    return load_config()


def reset_config_cache() -> None:
    """Clear the :func:`get_config` cache (primarily for tests)."""
    get_config.cache_clear()


__all__ = [
    "AuditConfig",
    "DataConfig",
    "DashboardConfig",
    "FeatureConfig",
    "ModelConfig",
    "PathConfig",
    "ProjectMeta",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PARAMS_PATH",
    "get_config",
    "load_config",
    "reset_config_cache",
]
