"""Ingestion of the UCI *Incident Management Process Enriched Event Log*.

The dataset (UCI ML Repository id 498) is an event log: every row is a
state transition of a support incident. Ingesting it means collapsing the
event stream down to one row per terminal ticket and deriving the binary
ground truth the audit engine learns from:

.. math::

    y = \\begin{cases}
        1 & \\text{if } reopen\\_count \\ge 1 \\\\
        0 & \\text{if } reopen\\_count = 0
    \\end{cases}

Network access is optional. When the archive cannot be fetched the loader
raises :class:`DatasetUnavailableError`, and callers (the CLI, the
dashboard) fall back to :mod:`ticketquality.data.synthetic`.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AuditConfig, get_config

__all__ = [
    "DatasetUnavailableError",
    "UCIDatasetLoader",
    "build_reopen_target",
    "filter_terminal_states",
    "collapse_event_log",
    "load_itsm_dataset",
]


class DatasetUnavailableError(RuntimeError):
    """Raised when the UCI archive cannot be downloaded or extracted."""


# --------------------------------------------------------------------------- #
# Pure transform helpers (independently unit-testable)
# --------------------------------------------------------------------------- #
def filter_terminal_states(
    frame: pd.DataFrame,
    terminal_states: Sequence[str] = ("Resolved", "Closed"),
    state_column: str = "incident_state",
) -> pd.DataFrame:
    """Keep only rows whose incident state is terminal.

    Matching is case-insensitive and whitespace tolerant so that the raw
    UCI strings (``"Resolved"``/``"Closed"``) and any lower-cased variants
    both survive.
    """

    if state_column not in frame.columns:
        raise KeyError(f"state column {state_column!r} not present in frame")

    normalised = (
        frame[state_column].astype("string").str.strip().str.lower()
    )
    wanted = {s.strip().lower() for s in terminal_states}
    mask = normalised.isin(wanted)
    return frame.loc[mask].copy()


def build_reopen_target(
    frame: pd.DataFrame,
    reopen_column: str = "reopen_count",
    target_column: str = "is_premature_closure",
    threshold: int = 1,
) -> pd.DataFrame:
    """Attach the binary premature-closure target.

    ``y = 1`` when the ticket was reopened at least ``threshold`` times,
    otherwise ``0``. Missing reopen counts are treated as zero (never
    reopened).
    """

    if reopen_column not in frame.columns:
        raise KeyError(f"reopen column {reopen_column!r} not present in frame")

    result = frame.copy()
    counts = pd.to_numeric(result[reopen_column], errors="coerce").fillna(0)
    result[target_column] = (counts >= threshold).astype(int)
    return result


def _to_minutes(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def collapse_event_log(
    events: pd.DataFrame,
    id_column: str = "number",
    state_column: str = "incident_state",
    time_column: str = "sys_updated_at",
) -> pd.DataFrame:
    """Collapse a multi-row lifecycle event log to one row per ticket.

    The last event per ticket is treated as the terminal record; numeric
    counters take their maximum over the ticket's lifetime (a reopen is
    counted even if a later event resets a transient field).
    """

    if id_column not in events.columns:
        # The enriched log uses `number`; tolerate alternatives.
        for candidate in ("incident_id", "ticket_id", "id"):
            if candidate in events.columns:
                id_column = candidate
                break
        else:
            raise KeyError("no ticket identifier column found in event log")

    work = events.copy()
    if time_column in work.columns:
        work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
        work = work.sort_values(time_column)

    numeric_max = [
        c
        for c in ("reopen_count", "reassignment_count", "sys_mod_count")
        if c in work.columns
    ]
    aggregations: dict = {}
    for column in work.columns:
        if column == id_column:
            continue
        if column in numeric_max:
            aggregations[column] = "max"
        elif column == state_column:
            aggregations[column] = "last"
        elif column == time_column:
            aggregations[column] = "max"
        elif pd.api.types.is_numeric_dtype(work[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "last"

    collapsed = work.groupby(id_column, as_index=False).agg(aggregations)
    return collapsed


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
@dataclass
class UCIDatasetLoader:
    """Download, extract and shape the UCI ITSM incident event log."""

    config: AuditConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = get_config()
        self.raw_dir = self.config.path("raw_dir")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # -- download ------------------------------------------------------- #
    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.config.data.archive_name

    def download(
        self,
        force: bool = False,
        timeout: int = 60,
        url: Optional[str] = None,
    ) -> Path:
        """Fetch the archive to ``data/raw`` (cached unless ``force``)."""

        target = self.archive_path
        if target.exists() and not force and target.stat().st_size > 0:
            return target

        source = url or self.config.data.uci_dataset_url
        try:
            request = urllib.request.Request(
                source, headers={"User-Agent": "ticketquality-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DatasetUnavailableError(
                f"could not download UCI dataset from {source}: {exc}"
            ) from exc

        if not payload:
            raise DatasetUnavailableError("downloaded archive was empty")

        target.write_bytes(payload)
        return target

    # -- extract -------------------------------------------------------- #
    def extract(self, archive: Optional[Path] = None, force: bool = False) -> Path:
        """Extract the archive into ``data/raw/<stem>`` and return the dir."""

        archive = Path(archive) if archive else self.archive_path
        if not archive.exists():
            raise DatasetUnavailableError(f"archive not found: {archive}")

        destination = self.raw_dir / archive.stem
        if destination.exists() and not force and any(destination.iterdir()):
            return destination
        destination.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(destination)
        except zipfile.BadZipFile as exc:
            raise DatasetUnavailableError(f"corrupt archive: {archive}") from exc
        return destination

    # -- find the event log --------------------------------------------- #
    def locate_event_log(self, root: Optional[Path] = None) -> Path:
        """Locate the incident event-log CSV inside an extracted tree."""

        root = Path(root) if root else self.extract()
        wanted = self.config.data.event_log_file.lower()
        candidates: List[Path] = sorted(root.rglob("*.csv"))
        # exact filename match wins
        for path in candidates:
            if path.name.lower() == wanted:
                return path
        # then anything that looks like the event log
        for path in candidates:
            if "event" in path.name.lower() and "incident" in path.name.lower():
                return path
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        raise DatasetUnavailableError(f"no CSV files found under {root}")

    # -- full pipeline --------------------------------------------------- #
    def load(
        self,
        force_download: bool = False,
        sample: Optional[int] = None,
    ) -> pd.DataFrame:
        """Download -> extract -> collapse -> filter -> label.

        Returns one row per terminal ticket with the binary target column
        attached.
        """

        archive = self.download(force=force_download)
        root = self.extract(archive)
        event_log = self.locate_event_log(root)

        events = pd.read_csv(event_log, low_memory=False)
        if sample is not None and sample > 0 and len(events) > sample:
            events = events.sample(n=sample, random_state=self.config.project.random_seed)

        collapsed = collapse_event_log(events)
        terminal = filter_terminal_states(
            collapsed, terminal_states=self.config.data.terminal_states
        )
        labelled = build_reopen_target(terminal)
        return self._normalise(labelled)

    # -- normalisation --------------------------------------------------- #
    def _normalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce raw UCI columns into the canonical audit schema."""

        work = frame.copy()
        rename_map = {
            "number": "ticket_id",
            "opened_at": "opened_at",
            "resolved_at": "resolved_at",
            "caller_id": "caller_id",
            "u_symptom": "category",
            "category": "category",
            "assignment_group": "assignment_group",
            "contact_type": "contact_type",
        }
        work = work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns})

        if "ticket_id" not in work.columns and work.index.name:
            work = work.reset_index().rename(columns={work.index.name: "ticket_id"})

        # durations: UCI exposes `made_sla`, `sys_mod_count`, `reassignment_count`
        for column in (
            "reassignment_count",
            "reopen_count",
            "sys_mod_count",
            "interaction_count",
        ):
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
            else:
                work[column] = 0

        if "opened_at" in work.columns and "resolved_at" in work.columns:
            opened = pd.to_datetime(work["opened_at"], errors="coerce")
            resolved = pd.to_datetime(work["resolved_at"], errors="coerce")
            minutes = (resolved - opened).dt.total_seconds() / 60.0
            work["resolution_minutes"] = minutes.fillna(minutes.median()).clip(lower=1.0)
        else:
            work["resolution_minutes"] = 60.0

        work["handle_time"] = work.get("handle_time", work["resolution_minutes"] * 0.5)
        work["active_minutes"] = np.clip(work["resolution_minutes"] * 0.45, 1.0, None)
        work["assignment_group_dwell"] = np.clip(
            work["resolution_minutes"] * 0.25, 0.5, None
        )

        for column, default in (
            ("priority", "Moderate"),
            ("impact", "Medium"),
            ("urgency", "Medium"),
            ("category", "Software"),
            ("assignment_group", "Service Desk L1"),
            ("contact_type", "Email"),
            ("shift", "Morning"),
            ("caller_id", "USR00000"),
        ):
            if column not in work.columns:
                work[column] = default

        if "made_sla" not in work.columns:
            work["made_sla"] = True
        work["made_sla"] = work["made_sla"].astype(bool)

        work = build_reopen_target(work)
        return work.reset_index(drop=True)


def load_itsm_dataset(
    prefer_synthetic: bool = False,
    n_synthetic: Optional[int] = None,
    sample: Optional[int] = None,
    config: Optional[AuditConfig] = None,
) -> pd.DataFrame:
    """High-level entry point used by the CLI and dashboard.

    Tries the real UCI download first and silently degrades to the
    synthetic generator when the network is unavailable (unless the caller
    explicitly asks for synthetic data).
    """

    cfg = config or get_config()
    if not prefer_synthetic:
        try:
            return UCIDatasetLoader(cfg).load(sample=sample)
        except DatasetUnavailableError:
            pass

    from .synthetic import generate_synthetic_tickets

    count = n_synthetic or cfg.data.synthetic_samples
    return generate_synthetic_tickets(n_samples=count, seed=cfg.project.random_seed)
