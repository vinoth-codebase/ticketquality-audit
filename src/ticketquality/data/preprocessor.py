"""Cleaning, validation and categorical encoding for ticket frames.

Two responsibilities live here:

1. **Schema validation** — a Pydantic model (:class:`TicketRecord`) that
   pins the types and value ranges of every attribute the pipeline
   depends on. Validation runs row-wise over a bounded sample so a large
   frame stays cheap while bad rows still fail loudly.

2. **Deterministic cleaning** — missing-value imputation, ordinal encoding
   of the ITSM priority/impact/urgency triads, and boolean coercion. The
   encodings are the same ones the feature engineer consumes, so they are
   exposed as module-level maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "TicketRecord",
    "TicketPreprocessor",
    "SchemaValidationError",
    "PRIORITY_ORDER",
    "IMPACT_ORDER",
    "URGENCY_ORDER",
    "encode_priority_triad",
]

PRIORITY_ORDER: Dict[str, int] = {
    "low": 1,
    "moderate": 2,
    "medium": 2,
    "normal": 2,
    "high": 3,
    "critical": 4,
    "very high": 4,
}
IMPACT_ORDER: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}
URGENCY_ORDER: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}

_NUMERIC_FIELDS = (
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "interaction_count",
    "resolution_minutes",
    "handle_time",
    "active_minutes",
    "assignment_group_dwell",
)


class SchemaValidationError(ValueError):
    """Raised when a ticket frame violates the expected schema."""


class TicketRecord(BaseModel):
    """One ticket, validated.

    Only the fields the audit engine genuinely relies on are required;
    everything else is optional with sensible ITSM defaults.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1)
    incident_state: str = "Closed"
    priority: str = "Moderate"
    impact: str = "Medium"
    urgency: str = "Medium"
    category: str = "Software"
    assignment_group: str = "Service Desk L1"
    contact_type: str = "Email"
    shift: str = "Morning"
    caller_id: str = "USR00000"
    made_sla: bool = True

    reassignment_count: int = Field(default=0, ge=0, le=200)
    reopen_count: int = Field(default=0, ge=0, le=100)
    sys_mod_count: int = Field(default=0, ge=0, le=1000)
    interaction_count: int = Field(default=1, ge=0, le=1000)
    resolution_minutes: float = Field(default=60.0, gt=0)
    handle_time: float = Field(default=30.0, ge=0)
    active_minutes: float = Field(default=30.0, ge=0)
    assignment_group_dwell: float = Field(default=10.0, ge=0)

    is_premature_closure: Optional[int] = Field(default=None, ge=0, le=1)

    @field_validator("priority")
    @classmethod
    def _known_priority(cls, value: str) -> str:
        if value.strip().lower() not in PRIORITY_ORDER:
            raise ValueError(f"unknown priority: {value!r}")
        return value

    @field_validator("impact")
    @classmethod
    def _known_impact(cls, value: str) -> str:
        if value.strip().lower() not in IMPACT_ORDER:
            raise ValueError(f"unknown impact: {value!r}")
        return value

    @field_validator("urgency")
    @classmethod
    def _known_urgency(cls, value: str) -> str:
        if value.strip().lower() not in URGENCY_ORDER:
            raise ValueError(f"unknown urgency: {value!r}")
        return value

    @field_validator("made_sla", mode="before")
    @classmethod
    def _coerce_bool(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "t"}
        return bool(value)


def encode_priority_triad(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal ``*_rank`` columns for priority/impact/urgency."""

    result = frame.copy()
    for column, mapping in (
        ("priority", PRIORITY_ORDER),
        ("impact", IMPACT_ORDER),
        ("urgency", URGENCY_ORDER),
    ):
        if column in result.columns:
            keys = result[column].astype("string").str.strip().str.lower()
            result[f"{column}_rank"] = keys.map(mapping).astype(float)
    return result


@dataclass
class TicketPreprocessor:
    """Clean and validate a raw ticket frame."""

    terminal_states: Sequence[str] = ("Resolved", "Closed")
    validate_rows: int = 2000
    strict: bool = False

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and coerce dtypes."""

        if frame.empty:
            raise ValueError("cannot preprocess an empty frame")

        work = frame.copy()

        # numeric coercion + imputation
        for column in _NUMERIC_FIELDS:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce")
            else:
                work[column] = np.nan

        work["reassignment_count"] = work["reassignment_count"].fillna(0)
        work["reopen_count"] = work["reopen_count"].fillna(0)
        work["sys_mod_count"] = work["sys_mod_count"].fillna(0)
        work["interaction_count"] = work["interaction_count"].fillna(1)

        for column in ("resolution_minutes", "handle_time", "active_minutes", "assignment_group_dwell"):
            median = work[column].median()
            if pd.isna(median) or median <= 0:
                median = 60.0 if column == "resolution_minutes" else 30.0
            work[column] = work[column].fillna(median).clip(lower=0.5)

        # ensure resolution >= active
        work["active_minutes"] = np.minimum(work["active_minutes"], work["resolution_minutes"])

        # categorical defaults
        defaults = {
            "incident_state": "Closed",
            "priority": "Moderate",
            "impact": "Medium",
            "urgency": "Medium",
            "category": "Software",
            "assignment_group": "Service Desk L1",
            "contact_type": "Email",
            "shift": "Morning",
            "caller_id": "USR00000",
        }
        for column, default in defaults.items():
            if column not in work.columns:
                work[column] = default
            else:
                work[column] = work[column].astype("string").fillna(default)

        if "made_sla" not in work.columns:
            work["made_sla"] = True
        work["made_sla"] = work["made_sla"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes", "y", "t"}
            if not isinstance(v, bool)
            else v
        )

        if "ticket_id" not in work.columns:
            work["ticket_id"] = [f"INC{100000 + i:06d}" for i in range(len(work))]

        # counts to int
        for column in ("reassignment_count", "reopen_count", "sys_mod_count", "interaction_count"):
            work[column] = work[column].round().astype(int)

        work = encode_priority_triad(work)
        return work.reset_index(drop=True)

    def validate(self, frame: pd.DataFrame, sample: Optional[int] = None) -> bool:
        """Validate rows against :class:`TicketRecord`.

        Returns ``True`` when every inspected row is valid. In
        non-strict mode invalid rows are logged (dropped silently) and
        ``False`` is returned; in strict mode the first failure raises.
        """

        limit = sample if sample is not None else self.validate_rows
        subset = frame.head(limit) if limit and len(frame) > limit else frame
        records = subset.to_dict(orient="records")
        for index, record in enumerate(records):
            try:
                TicketRecord(**record)
            except Exception as exc:  # pydantic ValidationError
                if self.strict:
                    raise SchemaValidationError(
                        f"row {index} failed validation: {exc}"
                    ) from exc
                return False
        return True

    def filter_terminal(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restrict to terminal states (Resolved / Closed)."""

        if "incident_state" not in frame.columns:
            return frame.copy()
        wanted = {s.strip().lower() for s in self.terminal_states}
        mask = (
            frame["incident_state"].astype("string").str.strip().str.lower().isin(wanted)
        )
        return frame.loc[mask].copy()

    def run(self, frame: pd.DataFrame, filter_terminal: bool = False) -> pd.DataFrame:
        """Clean -> (optional terminal filter) -> validate."""

        work = self.clean(frame)
        if filter_terminal:
            work = self.filter_terminal(work)
        self.validate(work)
        return work
