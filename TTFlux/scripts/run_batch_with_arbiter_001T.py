from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], root: Path) -> None:
    print()
    print("[001T] RUN:", " ".join(cmd))

    env = os.environ.copy()
    src_path = str((root / "src").resolve())

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = src_path + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = src_path

    subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--batch-runner", default="scripts/run_batch_001E.py")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--respect-ignore", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    config_dir = Path(args.config_dir)
    batch_runner = Path(args.batch_runner)

    py = sys.executable

    print("[001T] root         :", root)
    print("[001T] run_dir      :", run_dir)
    print("[001T] config_dir   :", config_dir)
    print("[001T] batch_runner :", batch_runner)
    print("[001T] python       :", py)

    if not args.skip_batch:
        run_cmd([py, str(batch_runner)], root)
    else:
        print("[001T] skip batch runner")

    # 001F raw manifest
    raw_manifest_csv = run_dir / "review_manifest_001T_raw.csv"
    raw_manifest_html = run_dir / "review_manifest_001T_raw.html"

    run_cmd(
        [
            py,
            "tools/build_review_manifest_001F.py",
            "--run-dir",
            str(run_dir),
            "--out-csv",
            str(raw_manifest_csv),
            "--out-html",
            str(raw_manifest_html),
        ],
        root,
    )

    # 001F2 repaired manifest
    repaired_csv = run_dir / "review_manifest_001T_repaired.csv"
    repaired_html = run_dir / "review_manifest_001T_repaired.html"

    run_cmd(
        [
            py,
            "tools/repair_review_manifest_001F2.py",
            "--in-csv",
            str(raw_manifest_csv),
            "--out-csv",
            str(repaired_csv),
            "--out-html",
            str(repaired_html),
        ],
        root,
    )

    # 001G trajectory
    trajectory_csv = run_dir / "review_manifest_001T_trajectory.csv"
    trajectory_html = run_dir / "review_manifest_001T_trajectory.html"

    run_cmd(
        [
            py,
            "tools/score_review_manifest_001G.py",
            "--run-dir",
            str(run_dir),
            "--manifest",
            str(repaired_csv),
            "--out-csv",
            str(trajectory_csv),
            "--out-html",
            str(trajectory_html),
        ],
        root,
    )

    # 001L appearance
    appearance_csv = run_dir / "appearance_features_001T.csv"
    appearance_html = run_dir / "appearance_features_001T.html"
    appearance_json = run_dir / "appearance_summary_001T.json"

    run_cmd(
        [
            py,
            "tools/extract_appearance_features_001L.py",
            "--run-dir",
            str(run_dir),
            "--config-dir",
            str(config_dir),
            "--goldset",
            str(trajectory_csv),
            "--out-csv",
            str(appearance_csv),
            "--out-html",
            str(appearance_html),
            "--out-json",
            str(appearance_json),
        ],
        root,
    )

    # 001N center blob
    center_csv = run_dir / "center_blob_features_001T.csv"
    center_html = run_dir / "center_blob_features_001T.html"
    center_json = run_dir / "center_blob_summary_001T.json"

    run_cmd(
        [
            py,
            "tools/extract_center_blob_features_001N.py",
            "--run-dir",
            str(run_dir),
            "--config-dir",
            str(config_dir),
            "--features",
            str(appearance_csv),
            "--out-csv",
            str(center_csv),
            "--out-html",
            str(center_html),
            "--out-json",
            str(center_json),
            "--patch-size",
            "33",
            "--max-points",
            "24",
        ],
        root,
    )

    # 001O micro blob
    micro_csv = run_dir / "micro_blob_features_001T.csv"
    micro_html = run_dir / "micro_blob_features_001T.html"
    micro_json = run_dir / "micro_blob_summary_001T.json"

    run_cmd(
        [
            py,
            "tools/extract_micro_blob_features_001O.py",
            "--run-dir",
            str(run_dir),
            "--config-dir",
            str(config_dir),
            "--features",
            str(center_csv),
            "--out-csv",
            str(micro_csv),
            "--out-html",
            str(micro_html),
            "--out-json",
            str(micro_json),
            "--patch-size",
            "41",
            "--max-points",
            "24",
        ],
        root,
    )

    # 001S arbiter
    arbiter_dir = run_dir / "arbiter_001T"

    arbiter_cmd = [
        py,
        "tools/apply_arbiter_001S.py",
        "--features",
        str(micro_csv),
        "--out-dir",
        str(arbiter_dir),
    ]

    if args.respect_ignore:
        arbiter_cmd.append("--respect-ignore")

    run_cmd(arbiter_cmd, root)

    print()
    print("[001T] DONE")
    print("[001T] raw manifest :", raw_manifest_html)
    print("[001T] trajectory   :", trajectory_html)
    print("[001T] appearance   :", appearance_html)
    print("[001T] center blob  :", center_html)
    print("[001T] micro blob   :", micro_html)
    print("[001T] arbiter      :", arbiter_dir / "arbiter_001S.html")


if __name__ == "__main__":
    main()
