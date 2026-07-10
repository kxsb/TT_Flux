from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

from ttflux import __version__
from ttflux.core.paths import (
    ANNOTATIONS_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    VIDEOS_DIR,
    ensure_project_layout,
)
from ttflux.video.catalog import write_catalog


def status(value: bool) -> str:
    return "OK" if value else "MANQUANT"


def command_doctor(_: argparse.Namespace) -> int:
    ensure_project_layout()

    print()
    print("TTFlux doctor")
    print("=" * 72)
    print(f"Version       : {__version__}")
    print(f"Python        : {platform.python_version()}")
    print(f"Exécutable    : {sys.executable}")
    print(f"Plateforme    : {platform.platform()}")
    print(f"Project root  : {PROJECT_ROOT}")
    print()
    print(f"ffmpeg        : {status(shutil.which('ffmpeg') is not None)}")
    print(f"ffprobe       : {status(shutil.which('ffprobe') is not None)}")
    print()
    print(f"data/videos   : {status(VIDEOS_DIR.is_dir())}")
    print(f"annotations   : {status(ANNOTATIONS_DIR.is_dir())}")
    print(f"runs          : {status(RUNS_DIR.is_dir())}")

    try:
        import fastapi
        import uvicorn

        print()
        print(f"FastAPI       : {fastapi.__version__}")
        print(f"Uvicorn       : {uvicorn.__version__}")
    except ImportError as exc:
        print()
        print(f"Dépendance manquante : {exc}")
        return 1

    required_ok = all(
        (
            shutil.which("ffmpeg") is not None,
            shutil.which("ffprobe") is not None,
            VIDEOS_DIR.is_dir(),
            ANNOTATIONS_DIR.is_dir(),
            RUNS_DIR.is_dir(),
        )
    )

    print()
    print("Résultat       :", "PRÊT" if required_ok else "À CORRIGER")
    return 0 if required_ok else 1


def command_index(_: argparse.Namespace) -> int:
    videos = write_catalog()

    print()
    print(f"Catalogue écrit : {PROJECT_ROOT / 'data' / 'video_catalog.json'}")
    print(f"Vidéos trouvées : {len(videos)}")

    for video in videos:
        resolution = "?"
        if video.get("width") and video.get("height"):
            resolution = f"{video['width']}x{video['height']}"

        fps = video.get("fps") or "?"
        duration = video.get("duration_s") or "?"

        print(
            f"- {video['relative_path']} | "
            f"{resolution} | {fps} fps | {duration} s"
        )

    return 0


def command_serve(args: argparse.Namespace) -> int:
    ensure_project_layout()

    import uvicorn

    print()
    print(f"TTFlux : http://{args.host}:{args.port}")
    print("Arrêt : Ctrl+C")
    print()

    uvicorn.run(
        "ttflux.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttflux",
        description="Pipeline local d'analyse vidéo de tennis de table",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Vérifie l'environnement local",
    )
    doctor_parser.set_defaults(handler=command_doctor)

    index_parser = subparsers.add_parser(
        "index",
        help="Indexe les vidéos présentes dans data/videos",
    )
    index_parser.set_defaults(handler=command_index)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Lance l'interface locale",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8787,
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
    )
    serve_parser.set_defaults(handler=command_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        exit_code = args.handler(args)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        exit_code = 1

    raise SystemExit(exit_code)