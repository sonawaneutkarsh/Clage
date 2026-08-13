"""CLI for the Clage viewer.

    python -m visual replay    --recording <file> [--export-tick N --export-out p]
    python -m visual tui       --recording <file> [--fps N] [--export-tick N --export-out file]
    python -m visual analytics --results <dir>    [--export <dir>] [--metrics a,b]
    python -m visual network   --recording <file> --organism <id> [--export <file>]

Interactive subcommands open matplotlib windows; `tui` renders a boxed terminal
replay; any `--export` renders headless output.
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def _set_backend(headless: bool) -> None:
    import matplotlib

    if headless:
        matplotlib.use("Agg")


def _cmd_replay(args: argparse.Namespace) -> None:
    _set_backend(args.export_tick is not None)
    from .data import load_recording
    from .world_view import WorldViewer, export_tick

    recording = load_recording(args.recording)
    if args.export_tick is not None:
        export_tick(recording, args.export_tick, args.export_out)
        print(f"wrote {args.export_out}")
        return
    WorldViewer(recording).show()


def _cmd_tui(args: argparse.Namespace) -> None:
    from .data import load_recording
    from .terminal_view import export_frame, play

    recording = load_recording(args.recording)
    if args.export_tick is not None:
        export_frame(recording, args.export_tick, args.export_out or "frame.txt")
        print(f"wrote {args.export_out or 'frame.txt'}")
        return
    play(recording, fps=args.fps)


def _cmd_analytics(args: argparse.Namespace) -> None:
    _set_backend(args.export is not None)
    from .analytics import analytics_figure, export_analytics
    from .data import METRICS, condition_names

    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]

    if args.export:
        written = export_analytics(args.results, args.export, metrics)
        for path in written:
            print(f"wrote {path}")
        return

    fig = analytics_figure(args.results, metrics)
    fig.suptitle(f"conditions: {', '.join(condition_names(args.results))}")
    import matplotlib.pyplot as plt

    plt.show()


def _cmd_network(args: argparse.Namespace) -> None:
    _set_backend(args.export is not None)
    from .data import load_recording
    from .network_view import genome_figure

    recording = load_recording(args.recording)
    genomes = {g["id"]: g for g in recording["genomes"]}
    if args.organism not in genomes:
        raise SystemExit(f"organism {args.organism} is not in this recording's genomes")
    genome = genomes[args.organism]
    fig = genome_figure(genome, title=f"genome {args.organism} (fitness {genome['fitness']:.3f})")

    if args.export:
        fig.savefig(args.export, dpi=110)
        print(f"wrote {args.export}")
        return
    import matplotlib.pyplot as plt

    plt.show()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m visual")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="replay a recorded generation")
    replay.add_argument("--recording", required=True)
    replay.add_argument("--export-tick", type=int, default=None,
                        help="export this tick as a PNG instead of opening a window")
    replay.add_argument("--export-out", default="tick.png")
    replay.set_defaults(func=_cmd_replay)

    tui = sub.add_parser("tui", help="boxed terminal replay of a recorded generation")
    tui.add_argument("--recording", required=True)
    tui.add_argument("--fps", type=float, default=10.0)
    tui.add_argument("--export-tick", type=int, default=None,
                     help="export this tick as a text file instead of live playback")
    tui.add_argument("--export-out", default="frame.txt")
    tui.set_defaults(func=_cmd_tui)

    analytics = sub.add_parser("analytics", help="condition-comparison charts")
    analytics.add_argument("--results", required=True, help="results directory of an experiment")
    analytics.add_argument("--metrics", default=None, help="comma-separated metric names")
    analytics.add_argument("--export", default=None,
                           help="directory for headless PNG export (else interactive)")
    analytics.set_defaults(func=_cmd_analytics)

    network = sub.add_parser("network", help="inspect a genome's network topology")
    network.add_argument("--recording", required=True)
    network.add_argument("--organism", type=int, required=True, help="genome id in the recording")
    network.add_argument("--export", default=None, help="PNG path for headless export")
    network.set_defaults(func=_cmd_network)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
