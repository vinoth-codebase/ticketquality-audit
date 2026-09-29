"""Unified Rich/Click command line interface.

Commands
--------
``ticketquality train``    train + validate the reopen-risk classifier
``ticketquality audit``    batch inference over a CSV, emit a scorecard
``ticketquality explain``  ASCII SHAP breakdown for one ticket
``ticketquality generate`` produce a synthetic audit dataset
``ticketquality download`` fetch the UCI ITSM dataset
``ticketquality info``     show resolved configuration and environment

The Rich rendering helpers at the bottom of this module are also used by
``scripts/generate_screenshots.py`` so the documented screenshots are
byte-for-byte what the CLI actually prints.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import get_config
from .data.loader import DatasetUnavailableError, UCIDatasetLoader
from .data.preprocessor import TicketPreprocessor
from .data.synthetic import generate_synthetic_tickets, write_synthetic_dataset
from .models.classifier import LIGHTGBM_AVAILABLE, TicketAuditModel, risk_level
from .models.evaluation import calibration_table, evaluate_classifier
from .models.explainer import SHAP_AVAILABLE, TicketExplainer
from .reports.exporter import (
    build_audit_report,
    export_json,
    export_markdown,
    export_triage_queue,
    save_audit_report,
)
from .reports.scorecard import build_scorecard

logger = logging.getLogger(__name__)

__all__ = [
    "cli",
    "main",
    "console",
    "render_ingestion_summary",
    "render_metrics_table",
    "render_explanation",
    "render_scorecard",
    "render_risk_distribution",
]

console = Console()

BANNER = "TicketQuality-Audit :: Premature Closure & Re-Open Risk Forecaster"


# --------------------------------------------------------------------------- #
# Rich render helpers (shared with the screenshot generator)
# --------------------------------------------------------------------------- #
def render_banner(console_: Console) -> None:
    console_.print(
        Panel(
            Text(BANNER, style="bold white"),
            border_style="cyan",
            padding=(0, 1),
        )
    )


def render_ingestion_summary(
    console_: Console,
    frame: pd.DataFrame,
    source: str,
    target_column: str = "is_premature_closure",
) -> None:
    """Render the dataset ingestion + class-balance panel."""

    table = Table(title="Dataset Ingestion", border_style="cyan", title_style="bold cyan")
    table.add_column("Attribute", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Source", source)
    table.add_row("Terminal tickets", f"{len(frame):,}")
    table.add_row("Columns", str(len(frame.columns)))
    if "incident_state" in frame.columns:
        states = frame["incident_state"].value_counts()
        table.add_row("States retained", ", ".join(f"{k} ({v:,})" for k, v in states.items()))

    if target_column in frame.columns:
        positives = int(frame[target_column].sum())
        negatives = int(len(frame) - positives)
        rate = positives / len(frame) if len(frame) else 0.0
        table.add_row("Target y=1 (reopen_count>=1)", f"{positives:,}")
        table.add_row("Target y=0 (reopen_count=0)", f"{negatives:,}")
        table.add_row("Positive rate", f"{rate:.1%}")
        imbalance = negatives / positives if positives else float("inf")
        table.add_row("Imbalance ratio (neg:pos)", f"{imbalance:.2f}:1")

    if "reassignment_count" in frame.columns:
        table.add_row(
            "Mean reassignments",
            f"{frame['reassignment_count'].mean():.2f}",
        )
    if "resolution_minutes" in frame.columns:
        table.add_row(
            "Median resolution (min)",
            f"{frame['resolution_minutes'].median():.1f}",
        )

    console_.print(table)


def render_metrics_table(
    console_: Console,
    metrics: Dict[str, float],
    threshold: float,
    recall_floor: float,
    confusion: Optional[List[List[int]]] = None,
    title: str = "Model Evaluation",
) -> None:
    """Render the evaluation-metrics table."""

    table = Table(title=title, border_style="green", title_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Reading", style="dim")

    readings = {
        "roc_auc": "global ranking power",
        "pr_auc": "rare-class ranking power",
        "f1": "balanced precision/recall",
        "precision": "share of flags that are real",
        "recall": "share of true reopens caught",
        "specificity": "share of clean tickets left alone",
        "accuracy": "overall hit rate",
        "brier_score": "calibration error (lower is better)",
    }
    order = ["roc_auc", "pr_auc", "precision", "recall", "f1", "specificity", "accuracy", "brier_score"]

    for key in order:
        if key in metrics:
            table.add_row(
                key.replace("_", " ").upper(),
                f"{metrics[key]:.4f}",
                readings.get(key, ""),
            )
    for key, value in metrics.items():
        if key not in order and key not in {"false_negative_count", "false_positive_count", "true_positive_count", "true_negative_count"}:
            table.add_row(key.replace("_", " ").title(), f"{value:.4f}", "")

    console_.print(table)

    achieved = metrics.get("recall", 0.0)
    met = achieved >= recall_floor
    style = "bold green" if met else "bold yellow"
    console_.print(
        Text(
            f"Decision threshold = {threshold:.4f}  |  recall floor = {recall_floor:.2f}  |  "
            f"achieved recall = {achieved:.4f}  ({'FLOOR MET' if met else 'FLOOR NOT MET'})",
            style=style,
        )
    )

    if confusion:
        cm_table = Table(title="Confusion Matrix", border_style="green", title_style="bold green")
        cm_table.add_column("", style="bold")
        cm_table.add_column("pred: clean", justify="right")
        cm_table.add_column("pred: reopen", justify="right")
        cm_table.add_row("actual: clean", str(confusion[0][0]), str(confusion[0][1]))
        cm_table.add_row("actual: reopen", str(confusion[1][0]), str(confusion[1][1]))
        console_.print(cm_table)


def render_explanation(console_: Console, explanation: Any) -> None:
    """Render a SHAP risk-driver breakdown for one ticket."""

    header = Text.assemble(
        ("Ticket ", "bold white"),
        (str(explanation.ticket_id), "bold cyan"),
        ("  |  reopen probability ", "white"),
        (f"{explanation.reopen_probability:.1%}", "bold red"),
        ("  |  risk ", "white"),
        (str(explanation.risk_level), "bold red"),
    )
    console_.print(Panel(header, border_style="red", padding=(0, 1)))

    table = Table(
        title="SHAP Risk Contribution Breakdown",
        border_style="magenta",
        title_style="bold magenta",
    )
    table.add_column("Driver", style="bold")
    table.add_column("Observed", justify="right")
    table.add_column("SHAP", justify="right")
    table.add_column("Contribution", justify="right")
    table.add_column("Direction", justify="center")
    table.add_column("Bar", style="magenta")

    for driver in explanation.drivers:
        magnitude = abs(driver.contribution_pct)
        bar = "#" * max(1, int(round(magnitude / 4)))
        direction_style = "red" if driver.direction == "increases" else "green"
        arrow = "+ raises risk" if driver.direction == "increases" else "- lowers risk"
        table.add_row(
            driver.label,
            _format_value(driver.value),
            f"{driver.shap_value:+.4f}",
            f"{driver.contribution_pct:+.1f}%",
            Text(arrow, style=direction_style),
            bar,
        )

    console_.print(table)

    base = getattr(explanation, "base_value", 0.0)
    raw = getattr(explanation, "raw_log_odds", 0.0)
    console_.print(
        Text(
            f"base log-odds {base:+.4f}  ->  ticket log-odds {raw:+.4f}  ->  "
            f"probability {explanation.reopen_probability:.4f}",
            style="dim",
        )
    )


def _format_value(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}"
    return str(value)


def render_scorecard(console_: Console, scorecard: Any, title: str = "QA Audit Scorecard") -> None:
    """Render the batch-audit scorecard."""

    summary = scorecard.summary
    table = Table(title=title, border_style="yellow", title_style="bold yellow")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for key, value in summary.items():
        if value is None:
            continue
        table.add_row(key.replace("_", " ").title(), str(value))
    console_.print(table)

    for group_name, frame in scorecard.by_group.items():
        if frame is None or frame.empty:
            continue
        group_table = Table(
            title=f"By {group_name.replace('_', ' ').title()}",
            border_style="yellow",
            title_style="bold yellow",
        )
        columns = list(frame.columns)
        for column in columns:
            group_table.add_column(
                column.replace("_", " ").title(),
                justify="right" if column != group_name else "left",
            )
        for _, row in frame.head(8).iterrows():
            group_table.add_row(*[str(row[column]) for column in columns])
        console_.print(group_table)

    if scorecard.recommendations:
        body = "\n".join(f"• {note}" for note in scorecard.recommendations)
        console_.print(
            Panel(body, title="Recommended QA Actions", border_style="cyan", padding=(0, 1))
        )


def render_risk_distribution(console_: Console, scored: pd.DataFrame) -> None:
    """Render the LOW / ELEVATED / CRITICAL band distribution."""

    counts = scored["risk_level"].value_counts() if "risk_level" in scored else pd.Series(dtype=int)
    table = Table(title="Risk Band Distribution", border_style="red", title_style="bold red")
    table.add_column("Band", style="bold")
    table.add_column("Tickets", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Bar", style="red")
    total = int(len(scored)) or 1
    for band in ("LOW", "ELEVATED", "CRITICAL"):
        count = int(counts.get(band, 0))
        share = count / total
        table.add_row(band, f"{count:,}", f"{share:.1%}", "#" * int(round(share * 40)))
    console_.print(table)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise click.ClickException(f"data file not found: {path}")
    return pd.read_csv(path)


@click.group(help=BANNER)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="ticketquality-audit", prog_name="ticketquality")
def cli(verbose: bool) -> None:
    """TicketQuality-Audit command line interface."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s :: %(message)s",
    )


@cli.command()
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV of tickets to train on. Omit to use the synthetic generator.",
)
@click.option("--samples", type=int, default=None, help="Synthetic sample count when --data is omitted.")
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--test-size", type=float, default=None, help="Hold-out fraction for evaluation.")
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def train(
    data_path: Optional[Path],
    samples: Optional[int],
    model_out: Optional[Path],
    test_size: Optional[float],
    report_out: Optional[Path],
) -> None:
    """Train and validate the premature-closure classifier."""

    config = get_config()
    render_banner(console)

    if data_path is not None:
        frame = _load_frame(data_path)
        source = str(data_path)
    else:
        count = samples or config.data.synthetic_samples
        frame = generate_synthetic_tickets(count, seed=config.project.random_seed)
        source = f"synthetic generator ({count:,} tickets)"

    preprocessor = TicketPreprocessor(terminal_states=config.data.terminal_states)
    frame = preprocessor.clean(frame)
    render_ingestion_summary(console, frame, source, target_column=config.model.target)

    from sklearn.model_selection import train_test_split

    fraction = test_size if test_size is not None else config.model.test_size
    stratify = frame[config.model.target] if frame[config.model.target].nunique() > 1 else None
    train_frame, test_frame = train_test_split(
        frame,
        test_size=fraction,
        random_state=config.project.random_seed,
        stratify=stratify,
    )

    model = TicketAuditModel.from_config(config)
    with console.status("[cyan]Training gradient-boosted classifier…[/cyan]"):
        train_metrics = model.train(train_frame)

    console.print(
        Text(
            f"Backend: {model.classifier.backend_}  |  "
            f"LightGBM available: {LIGHTGBM_AVAILABLE}  |  "
            f"train rows: {train_metrics['train_rows']:.0f}",
            style="dim",
        )
    )

    X_test = model.pipeline.transform(test_frame)
    report = evaluate_classifier(
        model.classifier, X_test, test_frame[config.model.target].to_numpy()
    )
    render_metrics_table(
        console,
        report.metrics,
        report.threshold,
        config.model.recall_floor,
        report.confusion,
        title="Hold-out Evaluation",
    )

    calibration = calibration_table(
        test_frame[config.model.target].to_numpy(),
        model.classifier.predict_proba(X_test)[:, 1],
    )
    console.print(
        Panel(
            calibration.to_string(index=False),
            title="Calibration (predicted vs observed)",
            border_style="blue",
            padding=(0, 1),
        )
    )

    saved = model.save(model_out or config.path("model_path"))
    console.print(Text(f"Model saved to {saved}", style="bold green"))

    if report_out:
        export_json(
            {
                "holdout_metrics": report.to_dict(),
                "train_metrics": train_metrics,
                "backend": model.classifier.backend_,
                "threshold": report.threshold,
                "recall_floor": config.model.recall_floor,
            },
            report_out,
        )
        console.print(Text(f"Metrics written to {report_out}", style="bold green"))


@cli.command()
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="CSV of closed tickets to audit.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Report destination (.json or .md).",
)
@click.option(
    "--queue",
    "queue_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional CSV destination for the ranked triage queue.",
)
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--explanations", is_flag=True, help="Include SHAP explanations for the top risks.")
def audit(
    file_path: Path,
    output_path: Optional[Path],
    queue_path: Optional[Path],
    model_path: Optional[Path],
    explanations: bool,
) -> None:
    """Run batch inference and emit an audit scorecard."""

    config = get_config()
    render_banner(console)

    frame = _load_frame(file_path)
    frame = TicketPreprocessor(terminal_states=config.data.terminal_states).clean(frame)
    render_ingestion_summary(console, frame, str(file_path), target_column=config.model.target)

    resolved_model_path = model_path or config.path("model_path")
    if resolved_model_path.exists():
        model = TicketAuditModel.load(resolved_model_path)
        console.print(Text(f"Loaded model from {resolved_model_path}", style="dim"))
    else:
        console.print(
            Text(
                "No trained model found; training a fresh one on synthetic data.",
                style="yellow",
            )
        )
        model = TicketAuditModel.from_config(config)
        model.train(generate_synthetic_tickets(config.data.synthetic_samples, seed=config.project.random_seed))

    with console.status("[cyan]Scoring tickets…[/cyan]"):
        scored = model.score_frame(frame)

    render_risk_distribution(console, scored)

    top_explanations = None
    if explanations:
        explainer = TicketExplainer(model)
        top_explanations = explainer.explain_frame(
            scored.sort_values("reopen_probability", ascending=False).head(5)
        )
        for explanation in top_explanations[:2]:
            render_explanation(console, explanation)

    scorecard = build_scorecard(scored, threshold=model.classifier.threshold_)
    render_scorecard(console, scorecard)

    payload = build_audit_report(
        scored,
        model_metrics=model.metrics_,
        explanations=top_explanations,
        threshold=model.classifier.threshold_,
        model_backend=getattr(model.classifier, "backend_", None),
        source=str(file_path),
    )

    destination = output_path or Path("audit_report.json")
    save_audit_report(payload, destination)
    console.print(Text(f"Audit report written to {destination}", style="bold green"))

    if queue_path:
        export_triage_queue(scored, queue_path)
        console.print(Text(f"Triage queue written to {queue_path}", style="bold green"))


@cli.command()
@click.option("--ticket-id", "ticket_id", type=str, required=True, help="Ticket identifier to explain.")
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV containing the ticket. Defaults to synthetic data.",
)
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def explain(ticket_id: str, file_path: Optional[Path], model_path: Optional[Path]) -> None:
    """Print an ASCII SHAP contribution breakdown for a single ticket."""

    config = get_config()
    render_banner(console)

    if not SHAP_AVAILABLE:
        raise click.ClickException("the `shap` package is required for explanations")

    if file_path is not None:
        frame = _load_frame(file_path)
    else:
        frame = generate_synthetic_tickets(config.data.synthetic_samples, seed=config.project.random_seed)
    frame = TicketPreprocessor(terminal_states=config.data.terminal_states).clean(frame)

    resolved_model_path = model_path or config.path("model_path")
    if resolved_model_path.exists():
        model = TicketAuditModel.load(resolved_model_path)
    else:
        model = TicketAuditModel.from_config(config)
        with console.status("[cyan]Training model for explanation…[/cyan]"):
            model.train(frame)

    explainer = TicketExplainer(model)
    try:
        explanation = explainer.explain_ticket(frame, ticket_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    render_explanation(console, explanation)


@cli.command()
@click.option("--samples", type=int, default=1000, help="Number of synthetic tickets to generate.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/synthetic_tickets.csv"),
)
def generate(samples: int, seed: int, output_path: Path) -> None:
    """Generate a synthetic audit dataset."""

    render_banner(console)
    frame = write_synthetic_dataset(output_path, n_samples=samples, seed=seed)
    render_ingestion_summary(console, frame, f"synthetic generator ({samples:,} tickets)")
    console.print(Text(f"Wrote {len(frame):,} tickets to {output_path}", style="bold green"))


@cli.command()
@click.option("--force", is_flag=True, help="Re-download even if cached.")
@click.option("--sample", type=int, default=None, help="Subsample the event log before collapsing.")
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/uci_incidents.csv"),
)
def download(force: bool, sample: Optional[int], output_path: Path) -> None:
    """Download and prepare the UCI ITSM incident dataset."""

    config = get_config()
    render_banner(console)
    loader = UCIDatasetLoader(config)
    try:
        with console.status("[cyan]Downloading UCI Incident Management dataset…[/cyan]"):
            frame = loader.load(force_download=force, sample=sample)
    except DatasetUnavailableError as exc:
        console.print(Text(f"Download unavailable: {exc}", style="bold red"))
        console.print(
            Text(
                "Falling back to the synthetic generator so the pipeline still runs.",
                style="yellow",
            )
        )
        frame = generate_synthetic_tickets(config.data.synthetic_samples, seed=config.project.random_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    render_ingestion_summary(console, frame, str(output_path))
    console.print(Text(f"Wrote {len(frame):,} tickets to {output_path}", style="bold green"))


@cli.command()
def info() -> None:
    """Show resolved configuration and environment details."""

    config = get_config()
    render_banner(console)
    table = Table(title="Environment", border_style="cyan", title_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Project", f"{config.project.name} v{config.project.version}")
    table.add_row("Random seed", str(config.project.random_seed))
    table.add_row("Model backend preference", str(config.model_params.get("backend", "auto")))
    table.add_row("LightGBM available", str(LIGHTGBM_AVAILABLE))
    table.add_row("SHAP available", str(SHAP_AVAILABLE))
    table.add_row("Recall floor", f"{config.model.recall_floor:.2f}")
    table.add_row("Model path", str(config.path("model_path")))
    table.add_row("Synthetic samples", str(config.data.synthetic_samples))
    table.add_row("Terminal states", ", ".join(config.data.terminal_states))
    console.print(table)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by the console script."""

    try:
        cli.main(args=argv, standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Abort:  # pragma: no cover
        console.print("[red]Aborted[/red]")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
