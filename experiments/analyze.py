"""CLI to aggregate results and write the analysis report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .analysis import condition_names, write_condition_csv
from .report import write_report

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results and report.")
    parser.add_argument("--results", required=True, help="results directory for an experiment")
    parser.add_argument("--report", default=None, help="output markdown report path")
    parser.add_argument("--plots", default=None, help="output plots directory")
    args = parser.parse_args(argv)

    exp_dir = Path(args.results)
    for name in condition_names(exp_dir):
        path = write_condition_csv(exp_dir, name)
        print(f"wrote {path}")

    report_path = args.report or (exp_dir / "report.md")
    plot_dir = args.plots or (exp_dir / "plots")
    write_report(exp_dir, report_path, plot_dir=plot_dir)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
