#!/usr/bin/env python3
"""Download and prepare the UCI Incident Management dataset.

Usage::

    python scripts/download_uci_dataset.py --output data/uci_incidents.csv
    python scripts/download_uci_dataset.py --force --sample 20000

Falls back to the synthetic generator when the network is unavailable, so
the script never leaves the repository in a broken state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when the package is not installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rich.console import Console  # noqa: E402

from ticketquality.config import get_config  # noqa: E402
from ticketquality.data.loader import DatasetUnavailableError, UCIDatasetLoader  # noqa: E402
from ticketquality.data.synthetic import generate_synthetic_tickets  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uci_incidents.csv"),
        help="Where to write the prepared ticket frame.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many raw events before collapsing to tickets.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Exit non-zero instead of falling back to synthetic data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()
    loader = UCIDatasetLoader(config)

    console.rule("[bold cyan]UCI ITSM Incident Management — ingestion")
    console.print(f"Dataset URL: [dim]{config.data.uci_dataset_url}[/dim]")
    console.print(f"Raw cache:   [dim]{loader.raw_dir}[/dim]")

    try:
        with console.status("[cyan]Downloading and extracting…[/cyan]"):
            frame = loader.load(force_download=args.force, sample=args.sample)
        console.print(f"[green]Ingested {len(frame):,} terminal tickets from UCI.[/green]")
    except DatasetUnavailableError as exc:
        console.print(f"[yellow]UCI download unavailable:[/yellow] {exc}")
        if args.no_fallback:
            console.print("[red]--no-fallback set; exiting.[/red]")
            return 2
        console.print("[yellow]Falling back to the synthetic generator.[/yellow]")
        frame = generate_synthetic_tickets(
            config.data.synthetic_samples, seed=config.project.random_seed
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    target = config.model.target
    if target in frame.columns:
        positives = int(frame[target].sum())
        console.print(
            f"Target balance: y=1 [bold]{positives:,}[/bold] / "
            f"y=0 [bold]{len(frame) - positives:,}[/bold] "
            f"({positives / len(frame):.1%} positive)"
        )
    console.print(f"[bold green]Wrote {len(frame):,} tickets to {args.output}[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
