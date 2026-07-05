from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.paths import ensure_project_layout, project_root
from .core.contracts import AnalysisRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttflux",
        description="TTFlux V4 - pipeline d'analyse vidéo mesurable",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("smoke", help="Vérifie le socle projet et affiche un résumé JSON.")
    sub.add_parser("init-layout", help="Crée les dossiers runtime locaux manquants.")

    return parser


def cmd_smoke() -> int:
    root = project_root()
    ensure_project_layout(root)
    run = AnalysisRun.new(root=root, input_video=None, status="smoke_ok")
    print(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_init_layout() -> int:
    root = project_root()
    ensure_project_layout(root)
    print(f"TTFlux V4 layout OK: {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        return cmd_smoke()
    if args.command == "init-layout":
        return cmd_init_layout()

    parser.print_help()
    return 0
