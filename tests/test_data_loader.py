"""Tests for the data layer: loader, synthetic generator, preprocessor, config."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ticketquality.config import (
    AuditConfig,
    DashboardConfig,
    load_config,
    reset_config_cache,
)
from ticketquality.data.loader import (
    DatasetUnavailableError,
    UCIDatasetLoader,
    build_reopen_target,
    collapse_event_log,
    filter_terminal_states,
    load_itsm_dataset,
)
from ticketquality.data.preprocessor import (
    SchemaValidationError,
    TicketPreprocessor,
    TicketRecord,
    encode_priority_triad,
)
from ticketquality.data.synthetic import (
    SyntheticTicketGenerator,
    generate_synthetic_tickets,
    write_synthetic_dataset,
)

# --------------------------------------------------------------------------- #
# Loader: pure transforms
# --------------------------------------------------------------------------- #
def test_filter_terminal_states_keeps_resolved_and_closed():
    frame = pd.DataFrame(
        {
            "incident_state": ["New", "Resolved", "Closed", "In Progress", "closed"],
            "ticket_id": list("abcde"),
        }
    )
    kept = filter_terminal_states(frame)
    assert list(kept["ticket_id"]) == ["b", "c", "e"]


def test_filter_terminal_states_is_case_and_space_insensitive():
    frame = pd.DataFrame({"incident_state": ["  RESOLVED ", "Closed\n"]})
    assert len(filter_terminal_states(frame)) == 2


def test_filter_terminal_states_missing_column_raises():
    with pytest.raises(KeyError):
        filter_terminal_states(pd.DataFrame({"x": [1]}))


def test_build_reopen_target_threshold_and_missing_values():
    frame = pd.DataFrame({"reopen_count": [0, 1, 2, np.nan]})
    labelled = build_reopen_target(frame)
    assert list(labelled["is_premature_closure"]) == [0, 1, 1, 0]


def test_build_reopen_target_missing_column_raises():
    with pytest.raises(KeyError):
        build_reopen_target(pd.DataFrame({"x": [1]}))


def test_collapse_event_log_takes_max_counters_and_last_state(event_log):
    collapsed = collapse_event_log(event_log)
    assert set(collapsed["number"]) == {"INC1", "INC2", "INC3", "INC4"}
    inc1 = collapsed.set_index("number").loc["INC1"]
    assert inc1["incident_state"] == "Closed"
    assert inc1["reopen_count"] == 1
    assert inc1["reassignment_count"] == 2


def test_collapse_event_log_accepts_alternative_id_column():
    frame = pd.DataFrame(
        {
            "ticket_id": ["A", "A", "B"],
            "incident_state": ["New", "Closed", "Resolved"],
            "reopen_count": [0, 0, 1],
        }
    )
    collapsed = collapse_event_log(frame)
    assert len(collapsed) == 2


def test_collapse_event_log_without_identifier_raises():
    with pytest.raises(KeyError):
        collapse_event_log(pd.DataFrame({"foo": [1, 2]}))


# --------------------------------------------------------------------------- #
# Loader: download / extract
# --------------------------------------------------------------------------- #
def test_download_raises_when_network_unavailable(base_config, monkeypatch, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)

    import urllib.request

    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(DatasetUnavailableError):
        loader.download(force=True)


def test_download_uses_cache_when_present(base_config, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    loader.archive_path.parent.mkdir(parents=True, exist_ok=True)
    loader.archive_path.write_bytes(b"cached-bytes")
    assert loader.download() == loader.archive_path


def test_extract_and_locate_event_log(base_config, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    loader.archive_path.parent.mkdir(parents=True, exist_ok=True)

    payload = b"number,incident_state,reopen_count\nINC1,Closed,1\n"
    with zipfile.ZipFile(loader.archive_path, "w") as zf:
        zf.writestr("nested/Incident_event_log.csv", payload)

    extracted = loader.extract()
    assert extracted.exists()
    located = loader.locate_event_log(extracted)
    assert located.name == "Incident_event_log.csv"


def test_extract_missing_archive_raises(base_config, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    with pytest.raises(DatasetUnavailableError):
        loader.extract()


def test_extract_corrupt_archive_raises(base_config, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    loader.archive_path.parent.mkdir(parents=True, exist_ok=True)
    loader.archive_path.write_bytes(b"not-a-zip")
    with pytest.raises(DatasetUnavailableError):
        loader.extract(force=True)


def test_locate_event_log_no_csv_raises(base_config, tmp_path):
    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetUnavailableError):
        loader.locate_event_log(empty)


def test_loader_load_end_to_end(base_config, tmp_path, monkeypatch):
    """The full download -> extract -> collapse -> label path, offline."""

    config = base_config.model_copy(deep=True)
    config.paths.raw_dir = str(tmp_path / "raw")
    loader = UCIDatasetLoader(config)
    loader.archive_path.parent.mkdir(parents=True, exist_ok=True)

    events = pd.DataFrame(
        {
            "number": ["INC1", "INC1", "INC2", "INC3"],
            "incident_state": ["New", "Closed", "Resolved", "In Progress"],
            "sys_updated_at": [
                "2025-01-01 09:00",
                "2025-01-01 15:00",
                "2025-01-02 09:00",
                "2025-01-03 09:00",
            ],
            "reopen_count": [0, 2, 0, 0],
            "reassignment_count": [0, 3, 1, 0],
            "sys_mod_count": [1, 4, 2, 0],
            "opened_at": [
                "2025-01-01 09:00",
                "2025-01-01 09:00",
                "2025-01-02 09:00",
                "2025-01-03 09:00",
            ],
            "resolved_at": [
                "2025-01-01 15:00",
                "2025-01-01 15:00",
                "2025-01-02 11:00",
                "2025-01-03 12:00",
            ],
            "priority": ["High", "High", "Low", "Moderate"],
        }
    )
    with zipfile.ZipFile(loader.archive_path, "w") as zf:
        zf.writestr("Incident_event_log.csv", events.to_csv(index=False))

    frame = loader.load()
    # INC3 is still In Progress, so only INC1/INC2 survive.
    assert set(frame["ticket_id"]) == {"INC1", "INC2"}
    assert frame.set_index("ticket_id").loc["INC1", "is_premature_closure"] == 1
    assert "resolution_minutes" in frame.columns


def test_load_itsm_dataset_falls_back_to_synthetic(monkeypatch):
    def boom(*args, **kwargs):
        raise DatasetUnavailableError("offline")

    monkeypatch.setattr(UCIDatasetLoader, "load", boom)
    frame = load_itsm_dataset(n_synthetic=50)
    assert len(frame) == 50
    assert "is_premature_closure" in frame.columns


def test_load_itsm_dataset_prefers_synthetic_when_asked():
    frame = load_itsm_dataset(prefer_synthetic=True, n_synthetic=25)
    assert len(frame) == 25


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
def test_generator_is_deterministic():
    a = generate_synthetic_tickets(120, seed=5)
    b = generate_synthetic_tickets(120, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_generator_different_seeds_differ():
    a = generate_synthetic_tickets(120, seed=5)
    b = generate_synthetic_tickets(120, seed=6)
    assert not a["reopen_count"].equals(b["reopen_count"])


def test_generator_schema_and_target_consistency():
    frame = generate_synthetic_tickets(300, seed=1)
    assert (frame["is_premature_closure"] == (frame["reopen_count"] >= 1).astype(int)).all()
    expected = {
        "ticket_id",
        "priority",
        "impact",
        "urgency",
        "category",
        "assignment_group",
        "caller_id",
        "reassignment_count",
        "resolution_minutes",
        "is_premature_closure",
    }
    assert expected.issubset(frame.columns)


def test_generator_reassignment_drives_reopen_risk():
    """The headline correlated failure mode must be present in the data."""

    frame = generate_synthetic_tickets(4000, seed=17)
    high = frame.loc[frame["reassignment_count"] >= 3, "is_premature_closure"].mean()
    low = frame.loc[frame["reassignment_count"] == 0, "is_premature_closure"].mean()
    assert high > low * 1.5
    assert 0.05 < frame["is_premature_closure"].mean() < 0.6


def test_generator_quick_close_with_handoffs_is_riskier():
    frame = generate_synthetic_tickets(4000, seed=23)
    quick = frame["resolution_minutes"] < frame["resolution_minutes"].median()
    many_handoffs = frame["reassignment_count"] >= 2
    combined = frame.loc[quick & many_handoffs, "is_premature_closure"].mean()
    neither = frame.loc[~quick & ~many_handoffs, "is_premature_closure"].mean()
    assert combined > neither


def test_generator_rejects_bad_sample_count():
    with pytest.raises(ValueError):
        SyntheticTicketGenerator(n_samples=0).generate()


def test_generator_accepts_custom_weights():
    gen = SyntheticTicketGenerator(n_samples=50, seed=2, reopen_intercept=-3.0)
    frame = gen.generate()
    assert len(frame) == 50


def test_write_synthetic_dataset_roundtrip(tmp_path):
    destination = tmp_path / "nested" / "tickets.csv"
    frame = write_synthetic_dataset(destination, n_samples=40, seed=9)
    assert destination.exists()
    reloaded = pd.read_csv(destination)
    assert len(reloaded) == len(frame)


# --------------------------------------------------------------------------- #
# Preprocessor / schema
# --------------------------------------------------------------------------- #
def test_preprocessor_cleans_missing_values(raw_frame):
    dirty = raw_frame.copy()
    dirty.loc[0:5, "resolution_minutes"] = np.nan
    dirty.loc[0:5, "priority"] = None
    cleaned = TicketPreprocessor().clean(dirty)
    assert cleaned["resolution_minutes"].notna().all()
    assert cleaned["priority"].notna().all()


def test_preprocessor_adds_missing_columns():
    frame = pd.DataFrame({"ticket_id": ["A", "B"], "resolution_minutes": [10, 20]})
    cleaned = TicketPreprocessor().clean(frame)
    for column in ("priority", "impact", "urgency", "category", "assignment_group"):
        assert column in cleaned.columns


def test_preprocessor_coerces_booleans():
    frame = pd.DataFrame(
        {
            "ticket_id": ["A", "B", "C"],
            "made_sla": ["true", "False", "1"],
            "resolution_minutes": [10, 20, 30],
        }
    )
    cleaned = TicketPreprocessor().clean(frame)
    assert cleaned["made_sla"].tolist() == [True, False, True]


def test_preprocessor_empty_frame_raises():
    with pytest.raises(ValueError):
        TicketPreprocessor().clean(pd.DataFrame())


def test_preprocessor_filters_terminal_states():
    frame = pd.DataFrame(
        {
            "ticket_id": list("abc"),
            "incident_state": ["New", "Closed", "Resolved"],
            "resolution_minutes": [1, 2, 3],
        }
    )
    pre = TicketPreprocessor()
    cleaned = pre.clean(frame)
    assert len(pre.filter_terminal(cleaned)) == 2


def test_preprocessor_filter_without_state_column_is_noop():
    frame = pd.DataFrame({"ticket_id": ["a"], "resolution_minutes": [1.0]})
    cleaned = TicketPreprocessor().clean(frame)
    assert len(TicketPreprocessor().filter_terminal(cleaned)) == 1


def test_preprocessor_run_pipeline(ticket_frame):
    result = TicketPreprocessor().run(ticket_frame, filter_terminal=True)
    assert not result.empty
    assert "priority_rank" in result.columns


def test_preprocessor_validate_returns_true_on_good_rows(ticket_frame):
    assert TicketPreprocessor().validate(ticket_frame) is True


def test_preprocessor_validate_detects_bad_rows():
    frame = pd.DataFrame({"ticket_id": ["A"], "priority": ["nonsense"], "resolution_minutes": [5.0]})
    assert TicketPreprocessor().validate(frame) is False


def test_preprocessor_strict_mode_raises():
    frame = pd.DataFrame({"ticket_id": ["A"], "priority": ["nonsense"], "resolution_minutes": [5.0]})
    with pytest.raises(SchemaValidationError):
        TicketPreprocessor(strict=True).validate(frame)


def test_ticket_record_rejects_negative_counts():
    with pytest.raises(ValidationError):
        TicketRecord(ticket_id="A", reassignment_count=-4)


def test_ticket_record_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        TicketRecord(ticket_id="A", priority="Urgent-ish")


def test_ticket_record_coerces_boolean_strings():
    record = TicketRecord(ticket_id="A", made_sla="yes")
    assert record.made_sla is True


def test_encode_priority_triad():
    frame = pd.DataFrame(
        {
            "priority": ["Low", "Critical"],
            "impact": ["Low", "High"],
            "urgency": ["Medium", "High"],
        }
    )
    encoded = encode_priority_triad(frame)
    assert encoded["priority_rank"].tolist() == [1.0, 4.0]
    assert encoded["impact_rank"].tolist() == [1.0, 3.0]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_load_config_defaults():
    config = load_config()
    assert isinstance(config, AuditConfig)
    assert config.model.recall_floor == 0.80
    assert "Resolved" in config.data.terminal_states
    assert config.model_params  # model_params.yaml was picked up


def test_load_config_rejects_bad_split(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  test_size: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_load_config_rejects_empty_state_list(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  terminal_states: []\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_non_mapping_root(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_path_resolution_is_absolute():
    config = load_config()
    assert config.path("model_path").is_absolute()
    with pytest.raises(KeyError):
        config.path("does_not_exist")


def test_dashboard_bands_default_to_none():
    config = load_config()
    assert config.dashboard.risk_low_upper is None
    assert config.dashboard.risk_elevated_upper is None
    assert DashboardConfig().risk_low_upper is None


def test_reset_config_cache():
    reset_config_cache()
    assert load_config().project.name == "ticketquality-audit"
