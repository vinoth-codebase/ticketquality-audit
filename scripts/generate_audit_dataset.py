#!/usr/bin/env python3
"""Generate the synthetic audit dataset used for offline demos and tests.

Usage::

    python scripts/generate_audit_dataset.py --samples 1000
    python scripts/generate_audit_dataset.py --samples 5000 --seed 7 --output data/big.csv

The generator is *not* uniform noise: reopen probability is driven by a
logistic mechanism over operational drivers (reassignment counts, short
active resolution, priority mismatch, caller history, a quick-close x
handoff interaction) plus a latent unobserved factor, so the resulting
data has the correlated failure modes a real support desk exhibits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from ticketquality.data.synthetic import (  # noqa: E402
    SyntheticTicketGenerator,
    write_synthetic_dataset,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000, help="Number of tickets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_tickets.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the summary table without writing a CSV.",
    )
    return parser.parse_args()


def describe(frame, console_: Console) -> None:
    table = Table(title="Synthetic Audit Dataset", border_style="cyan", title_style="bold cyan")
    table.add_column("Attribute", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Tickets", f"{len(frame):,}")
    table.add_row("Columns", str(len(frame.columns)))

    positives = int(frame["is_premature_closure"].sum())
    table.add_row("Premature closures (y=1)", f"{positives:,}")
    table.add_row("Clean closures (y=0)", f"{len(frame) - positives:,}")
    table.add_row("Positive rate", f"{positives / len(frame):.1%}")
    table.add_row("Mean reassignments", f"{frame['reassignment_count'].mean():.2f}")
    table.add_row("Mean reopen count", f"{frame['reopen_count'].mean():.3f}")
    table.add_row("Median resolution (min)", f"{frame['resolution_minutes'].median():.1f}")
    table.add_row(
        "Reopen rate | 3+ reassignments",
        f"{frame.loc[frame['reassignment_count'] >= 3, 'is_premature_closure'].mean():.1%}",
    )
    table.add_row(
        "Reopen rate | <3 reassignments",
        f"{frame.loc[frame['reassignment_count'] < 3, 'is_premature_closure'].mean():.1%}",
    )
    console_.print(table)


def main() -> int:
    args = parse_args()
    console.rule("[bold cyan]Synthetic audit dataset")

    frame = SyntheticTicketGenerator(n_samples=args.samples, seed=args.seed).generate()
    describe(frame, console)

    if not args.summary_only:
        write_synthetic_dataset(args.output, n_samples=args.samples, seed=args.seed)
        console.print(
            f"[bold green]Wrote {len(frame):,} tickets to {args.output}[/bold green]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
