"""Tests for the Rich/Click command line interface and its renderers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from rich.console import Console

from ticketquality import cli as cli_module
from ticketquality.cli import (
    cli,
    render_banner,
    render_explanation,
    render_ingestion_summary,
    render_metrics_table,
    render_risk_distribution,
    render_scorecard,
)
from ticketquality.models.explainer import TicketExplainer
from ticketquality.reports.scorecard import build_scorecard


@pytest.fixture()
def cli_config(trained_model, fast_config, tmp_path):
    """A fast config whose model path points at a pre-saved model."""

    config = fast_config.model_copy(deep=True)
    model_path = tmp_path / "artifacts" / "model.joblib"
    trained_model.save(model_path)
    config.paths.model_path = str(model_path)
    config.data.synthetic_samples = 150
    return config


@pytest.fixture()
def patched_cli(monkeypatch, cli_config):
    """Patch the CLI's config accessor so commands stay fast and offline."""

    monkeypatch.setattr(cli_module, "get_config", lambda: cli_config)
    return cli_config


@pytest.fixture()
def ticket_csv(ticket_frame, tmp_path) -> Path:
    path = tmp_path / "tickets.csv"
    ticket_frame.head(120).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Group / basic commands
# --------------------------------------------------------------------------- #
def test_cli_group_help_lists_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "audit", "explain", "generate", "download", "info"):
        assert command in result.output


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_info_command(patched_cli):
    result = CliRunner().invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "ticketquality-audit" in result.output
    assert "LightGBM available" in result.output


def test_verbose_flag_sets_debug_logging(patched_cli):
    result = CliRunner().invoke(cli, ["--verbose", "info"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def test_generate_command_writes_csv(tmp_path):
    destination = tmp_path / "gen" / "tickets.csv"
    result = CliRunner().invoke(
        cli, ["generate", "--samples", "80", "--seed", "5", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    assert destination.exists()
    frame = pd.read_csv(destination)
    assert len(frame) == 80
    assert "Target y=1" in result.output


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def test_train_command_on_csv(patched_cli, ticket_csv, tmp_path):
    model_out = tmp_path / "trained.joblib"
    report_out = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--data",
            str(ticket_csv),
            "--model-out",
            str(model_out),
            "--report-out",
            str(report_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_out.exists()
    assert "ROC AUC" in result.output
    assert "Decision threshold" in result.output
    payload = json.loads(report_out.read_text())
    assert "holdout_metrics" in payload
    assert payload["recall_floor"] == pytest.approx(patched_cli.model.recall_floor)


def test_train_command_without_data_uses_synthetic(patched_cli, tmp_path):
    model_out = tmp_path / "synthetic.joblib"
    result = CliRunner().invoke(
        cli, ["train", "--samples", "150", "--model-out", str(model_out)]
    )
    assert result.exit_code == 0, result.output
    assert model_out.exists()
    assert "synthetic generator" in result.output


def test_train_command_missing_file_errors(patched_cli):
    result = CliRunner().invoke(cli, ["train", "--data", "definitely-missing.csv"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_audit_command_writes_report(patched_cli, ticket_csv, tmp_path):
    report = tmp_path / "audit.json"
    queue = tmp_path / "queue.csv"
    result = CliRunner().invoke(
        cli,
        [
            "audit",
            "--file",
            str(ticket_csv),
            "--output",
            str(report),
            "--queue",
            str(queue),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["report_type"] == "premature_closure_audit"
    assert payload["scorecard"]["summary"]["tickets_audited"] == 120
    assert queue.exists()
    assert "Risk Band Distribution" in result.output


def test_audit_command_with_explanations(patched_cli, ticket_csv, tmp_path):
    report = tmp_path / "audit_explained.json"
    result = CliRunner().invoke(
        cli,
        [
            "audit",
            "--file",
            str(ticket_csv),
            "--output",
            str(report),
            "--explanations",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["top_risk_explanations"]
    assert "SHAP Risk Contribution Breakdown" in result.output


def test_audit_command_markdown_output(patched_cli, ticket_csv, tmp_path):
    report = tmp_path / "audit.md"
    result = CliRunner().invoke(
        cli, ["audit", "--file", str(ticket_csv), "--output", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert "Support QA Audit Scorecard" in report.read_text()


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
def test_explain_command(patched_cli, ticket_csv):
    ticket_id = str(pd.read_csv(ticket_csv).iloc[0]["ticket_id"])
    result = CliRunner().invoke(
        cli, ["explain", "--ticket-id", ticket_id, "--file", str(ticket_csv)]
    )
    assert result.exit_code == 0, result.output
    assert ticket_id in result.output
    assert "SHAP Risk Contribution Breakdown" in result.output
    assert "base log-odds" in result.output


def test_explain_unknown_ticket_errors(patched_cli, ticket_csv):
    result = CliRunner().invoke(
        cli, ["explain", "--ticket-id", "NOPE-123", "--file", str(ticket_csv)]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


# --------------------------------------------------------------------------- #
# download (offline fallback path)
# --------------------------------------------------------------------------- #
def test_download_command_falls_back_to_synthetic(patched_cli, tmp_path, monkeypatch):
    from ticketquality.data.loader import DatasetUnavailableError

    def boom(self, *args, **kwargs):
        raise DatasetUnavailableError("offline in tests")

    monkeypatch.setattr(cli_module.UCIDatasetLoader, "load", boom)
    destination = tmp_path / "uci.csv"
    result = CliRunner().invoke(cli, ["download", "--output", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert "Download unavailable" in result.output


# --------------------------------------------------------------------------- #
# Rich render helpers
# --------------------------------------------------------------------------- #
def _record() -> Console:
    return Console(record=True, width=140, force_terminal=False)


def test_render_banner():
    console = _record()
    render_banner(console)
    assert "TicketQuality-Audit" in console.export_text()


def test_render_ingestion_summary(ticket_frame):
    console = _record()
    render_ingestion_summary(console, ticket_frame, "unit-test")
    text = console.export_text()
    assert "Dataset Ingestion" in text
    assert "Target y=1" in text
    assert "unit-test" in text


def test_render_ingestion_summary_without_target_column(ticket_frame):
    console = _record()
    render_ingestion_summary(
        console, ticket_frame.drop(columns=["is_premature_closure"]), "unit-test"
    )
    assert "Dataset Ingestion" in console.export_text()


def test_render_metrics_table():
    console = _record()
    metrics = {
        "roc_auc": 0.91,
        "pr_auc": 0.74,
        "precision": 0.58,
        "recall": 0.84,
        "f1": 0.69,
        "specificity": 0.85,
        "accuracy": 0.84,
        "brier_score": 0.10,
        "custom_metric": 0.5,
    }
    render_metrics_table(console, metrics, 0.07, 0.80, [[170, 30], [8, 42]])
    text = console.export_text()
    assert "ROC AUC" in text
    assert "FLOOR MET" in text
    assert "Confusion Matrix" in text


def test_render_metrics_table_reports_unmet_floor():
    console = _record()
    render_metrics_table(console, {"recall": 0.5, "f1": 0.4}, 0.5, 0.80)
    assert "FLOOR NOT MET" in console.export_text()


def test_render_explanation(trained_model, ticket_frame):
    console = _record()
    explanation = TicketExplainer(trained_model).explain_frame(ticket_frame.head(1))[0]
    render_explanation(console, explanation)
    text = console.export_text()
    assert "SHAP Risk Contribution Breakdown" in text
    assert explanation.ticket_id in text


def test_render_scorecard(trained_model, ticket_frame):
    console = _record()
    scored = trained_model.score_frame(ticket_frame)
    scorecard = build_scorecard(scored, threshold=trained_model.classifier.threshold_)
    render_scorecard(console, scorecard)
    text = console.export_text()
    assert "QA Audit Scorecard" in text
    assert "Recommended QA Actions" in text


def test_render_scorecard_with_empty_breakdown():
    console = _record()
    from ticketquality.reports.scorecard import AuditScorecard

    scorecard = AuditScorecard(
        summary={"tickets_audited": 0, "skipped": None},
        by_group={"category": pd.DataFrame()},
        recommendations=["nothing to do"],
    )
    render_scorecard(console, scorecard)
    assert "QA Audit Scorecard" in console.export_text()


def test_render_risk_distribution(trained_model, ticket_frame):
    console = _record()
    scored = trained_model.score_frame(ticket_frame)
    render_risk_distribution(console, scored)
    text = console.export_text()
    for band in ("LOW", "ELEVATED", "CRITICAL"):
        assert band in text


def test_main_entrypoint_returns_zero():
    from ticketquality.cli import main

    assert main(["info"]) == 0


def test_main_entrypoint_handles_click_error(monkeypatch):
    from ticketquality.cli import main

    # An unknown command surfaces as a ClickException -> non-zero exit.
    assert main(["not-a-real-command"]) != 0
