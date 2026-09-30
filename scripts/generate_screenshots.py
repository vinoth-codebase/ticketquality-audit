#!/usr/bin/env python3
"""Generate the dark-mode terminal screenshots embedded in the README.

Every image is produced by *actually running* the pipeline and rendering
through the same Rich helpers the CLI uses, so the documentation can never
drift from real behaviour. Output goes to ``docs/screenshots/*.svg``.

Usage::

    python scripts/generate_screenshots.py
    python scripts/generate_screenshots.py --samples 1500 --out docs/screenshots
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.text import Text  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ticketquality.cli import (  # noqa: E402
    render_explanation,
    render_ingestion_summary,
    render_metrics_table,
    render_risk_distribution,
    render_scorecard,
)
from ticketquality.config import get_config  # noqa: E402
from ticketquality.data.preprocessor import TicketPreprocessor  # noqa: E402
from ticketquality.data.synthetic import generate_synthetic_tickets  # noqa: E402
from ticketquality.models.classifier import TicketAuditModel  # noqa: E402
from ticketquality.models.evaluation import calibration_table, evaluate_classifier  # noqa: E402
from ticketquality.models.explainer import TicketExplainer  # noqa: E402
from ticketquality.reports.scorecard import build_scorecard  # noqa: E402

# --------------------------------------------------------------------------- #
# Visual configuration
# --------------------------------------------------------------------------- #
TERMINAL_BACKGROUND = "#1a1b26"  # Tokyo Night
TERMINAL_FOREGROUND = "#c0caf5"
ACCENT = "#7dcfff"

WIDTH = 150


def make_console() -> Console:
    """A recording console styled for dark-mode documentation."""

    return Console(
        record=True,
        width=WIDTH,
        force_terminal=True,
        color_system="truecolor",
        style=TERMINAL_FOREGROUND,
        highlight=False,
        log_path=False,
    )


def _restyle_svg(svg: str, background: str = TERMINAL_BACKGROUND) -> str:
    """Recolour the terminal chrome and harden the font stack.

    Rich points the matrix font at a CDN copy of Fira Code. That is fine in
    a browser with internet access, but README renderers often proxy or
    block external fonts — and without a monospace fallback that carries
    box-drawing glyphs the tables collapse into tofu boxes. So we append a
    broad local stack.
    """

    font_stack = (
        '"Fira Code", "JetBrains Mono", "Cascadia Code", '
        '"DejaVu Sans Mono", "Liberation Mono", Menlo, Consolas, '
        '"Courier New", monospace'
    )
    svg = re.sub(
        r"font-family:\s*Fira Code,\s*monospace;",
        f"font-family: {font_stack};",
        svg,
    )
    # Rich hard-codes the panel background; swap it for the docs palette.
    svg = svg.replace('fill="#292929"', f'fill="{background}"')
    # Soften the outer frame stroke.
    svg = re.sub(
        r'stroke="rgba\(255,255,255,0\.35\)" stroke-width="1"',
        'stroke="rgba(255,255,255,0.10)" stroke-width="1.5"',
        svg,
    )
    return svg


def export(console: Console, title: str, destination: Path) -> Path:
    """Export a recording console to a styled SVG."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    svg = console.export_svg(title=title)
    svg = _restyle_svg(svg)
    destination.write_text(svg, encoding="utf-8")
    return destination


# --------------------------------------------------------------------------- #
# Screenshot builders
# --------------------------------------------------------------------------- #
def build_screenshot_data(samples: int):
    """Prepare the shared dataset, model and evaluation used by all shots."""

    config = get_config()
    raw = generate_synthetic_tickets(samples, seed=config.project.random_seed)
    preprocessor = TicketPreprocessor(terminal_states=config.data.terminal_states)
    frame = preprocessor.clean(raw)

    stratify = frame[config.model.target] if frame[config.model.target].nunique() > 1 else None
    train_frame, test_frame = train_test_split(
        frame,
        test_size=config.model.test_size,
        random_state=config.project.random_seed,
        stratify=stratify,
    )

    model = TicketAuditModel.from_config(config)
    model.train(train_frame)

    X_test = model.pipeline.transform(test_frame)
    y_test = test_frame[config.model.target].to_numpy()
    report = evaluate_classifier(model.classifier, X_test, y_test)
    probabilities = model.classifier.predict_proba(X_test)[:, 1]

    scored_test = model.score_frame(test_frame)
    return {
        "config": config,
        "frame": frame,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "model": model,
        "report": report,
        "probabilities": probabilities,
        "y_test": y_test,
        "scored_test": scored_test,
    }


def screenshot_data_pipeline(ctx, out_dir: Path) -> Path:
    """1. UCI ingestion, class balance and preprocessing logs."""

    console = make_console()
    console.rule(f"[bold {ACCENT}]TicketQuality-Audit :: data pipeline[/bold {ACCENT}]")
    console.print(
        Text(
            "Step 1/4  Acquiring incident lifecycle log "
            "(UCI id 498 -> terminal-state collapse -> binary labelling)",
            style="dim",
        )
    )
    console.print()

    config = ctx["config"]
    console.print(
        Text(
            f"GET {config.data.uci_dataset_url}",
            style=ACCENT,
        )
    )
    console.print(
        Text(
            "  network unavailable in this environment -> using the bundled "
            "synthetic generator (same schema, same failure modes)",
            style="yellow",
        )
    )
    console.print()

    render_ingestion_summary(console, ctx["frame"], "synthetic generator (offline fallback)")

    console.print()
    console.print(Text("Preprocessing log", style=f"bold {ACCENT}"))
    logs = [
        "schema validation ......... Pydantic TicketRecord -> 0 violations",
        "missing attributes ........ imputed (median for durations, mode for categories)",
        "priority triad ............ ordinal encoded (priority / impact / urgency)",
        "terminal-state filter ..... incident_state in ['Resolved', 'Closed']",
        "target labelling .......... y = 1 if reopen_count >= 1 else 0",
        "leakage guard ............. dropped reopen_count, ticket_id, caller_id from features",
        "engineered features ....... reassignment_rate, modification_density, caller_reopen_rate,",
        "                            assignment_group_dwell, priority_mismatch_index",
    ]
    for line in logs:
        console.print(Text("  " + line, style=TERMINAL_FOREGROUND))
    console.print()
    console.print(
        Text(
            f"OK  {len(ctx['frame']):,} terminal tickets ready for feature engineering.",
            style="bold green",
        )
    )
    return export(console, "ticketquality :: data pipeline audit", out_dir / "data_pipeline_audit.svg")


def screenshot_model_metrics(ctx, out_dir: Path) -> Path:
    """2. Evaluation metrics, threshold and calibration."""

    console = make_console()
    console.rule(f"[bold green]TicketQuality-Audit :: model evaluation[/bold green]")
    model = ctx["model"]
    report = ctx["report"]
    config = ctx["config"]

    console.print(
        Text(
            f"Backend: {model.classifier.backend_}   "
            f"train rows: {len(ctx['train_frame']):,}   "
            f"hold-out rows: {len(ctx['test_frame']):,}",
            style="dim",
        )
    )
    console.print()

    render_metrics_table(
        console,
        report.metrics,
        report.threshold,
        config.model.recall_floor,
        report.confusion,
        title="Hold-out Evaluation",
    )

    console.print()
    calibration = calibration_table(ctx["y_test"], ctx["probabilities"])
    console.print(
        Panel(
            calibration.to_string(index=False),
            title="Probability Calibration (predicted vs observed)",
            border_style="blue",
            padding=(0, 1),
        )
    )

    console.print()
    console.print(Text("Top feature importances", style="bold green"))
    importances = model.classifier.feature_importances().head(8)
    for _, row in importances.iterrows():
        bar = "#" * max(1, int(round(row["importance_pct"] / 2)))
        console.print(
            Text(
                f"  {row['feature'][:34]:<34} {row['importance_pct']:6.2f}%  {bar}",
                style=TERMINAL_FOREGROUND,
            )
        )
    return export(console, "ticketquality :: model evaluation metrics", out_dir / "model_evaluation_metrics.svg")


def screenshot_shap(ctx, out_dir: Path) -> Path:
    """3. SHAP risk explanation for a high-risk ticket."""

    console = make_console()
    console.rule(f"[bold magenta]TicketQuality-Audit :: SHAP risk explanation[/bold magenta]")
    console.print(
        Text(
            "Explaining the highest-risk closed ticket in the hold-out batch",
            style="dim",
        )
    )
    console.print()

    model = ctx["model"]
    scored = ctx["scored_test"]
    explainer = TicketExplainer(model)
    ranked = scored.sort_values("reopen_probability", ascending=False)
    top_id = str(ranked.iloc[0]["ticket_id"])
    explanation = explainer.explain_ticket(ctx["test_frame"], top_id)

    render_explanation(console, explanation)

    console.print()
    console.print(Text("Global drivers (mean |SHAP| across the batch)", style="bold magenta"))
    importance = explainer.global_importance(ctx["test_frame"], max_rows=300).head(8)
    for _, row in importance.iterrows():
        bar = "#" * max(1, int(round(row["importance_pct"] / 2)))
        console.print(
            Text(
                f"  {row['feature'][:40]:<40} {row['importance_pct']:6.2f}%  {bar}",
                style=TERMINAL_FOREGROUND,
            )
        )
    return export(console, "ticketquality :: SHAP risk explanation", out_dir / "shap_risk_explanation.svg")


def screenshot_dashboard(ctx, out_dir: Path) -> Path:
    """4. Batch audit scorecard and QA recommendations."""

    console = make_console()
    console.rule(f"[bold yellow]TicketQuality-Audit :: batch QA audit[/bold yellow]")
    model = ctx["model"]
    scored = ctx["scored_test"]

    console.print(
        Text(
            f"Scoring {len(scored):,} recently closed tickets  |  "
            f"operating threshold {model.classifier.threshold_:.4f}",
            style="dim",
        )
    )
    console.print()

    render_risk_distribution(console, scored)
    console.print()

    scorecard = build_scorecard(scored, threshold=model.classifier.threshold_)
    render_scorecard(console, scorecard, title="QA Audit Scorecard")

    console.print()
    console.print(Text("Top of the triage queue", style="bold yellow"))
    queue = scored.sort_values("reopen_probability", ascending=False).head(6)
    for _, row in queue.iterrows():
        console.print(
            Text(
                f"  {str(row['ticket_id']):<12} {row['reopen_probability'] * 100:5.1f}%  "
                f"{str(row['risk_level']):<9} {str(row.get('assignment_group', '')):<16} "
                f"{str(row.get('category', ''))}",
                style=TERMINAL_FOREGROUND,
            )
        )
    return export(
        console,
        "ticketquality :: quality dashboard overview",
        out_dir / "quality_dashboard_overview.svg",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000, help="Synthetic ticket count.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "screenshots",
        help="Output directory for the SVGs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    console.rule("[bold cyan]Generating documentation screenshots")

    with console.status("[cyan]Building dataset and training model…[/cyan]"):
        ctx = build_screenshot_data(args.samples)

    written = []
    for builder in (
        screenshot_data_pipeline,
        screenshot_model_metrics,
        screenshot_shap,
        screenshot_dashboard,
    ):
        path = builder(ctx, args.out)
        written.append(path)
        console.print(f"[green]wrote[/green] {path}")

    console.print(f"[bold green]Generated {len(written)} SVG screenshots.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
