from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003E"

RULE_NAME = "multi_object_table_player_safe_v1"
RULE_TEXT = (
    "inside_player_motion_mask_ratio_003C >= 0.516666 "
    "AND point_distance_to_table_px_median <= 216.719 "
    "AND max_accel >= 17.672"
)

REQUIRED_COLS = [
    "inside_player_motion_mask_ratio_003C",
    "point_distance_to_table_px_median",
    "max_accel",
]

EXPECTED_HITS = {"R0010", "R0009", "R0020", "R0014", "R0008", "R0021"}
CTX003B4_HITS = {"R0010", "R0009", "R0020", "R0008", "R0021"}

ID_ALIASES = ["ID", "review_id", "id", "record_id", "segment_review_id"]
TARGET_ALIASES = ["target", "target_class", "class", "label", "review_class"]
CTX_ALIASES = ["ctx003B4", "contextual_hit_003B4", "contextual_shadow_hit_003B4"]


def first_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def norm_target(x) -> str:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"humanreject", "human_rejected"}:
        return "human_reject"
    if s in {"humanpartial", "partial"}:
        return "human_partial"
    if s in {"autoreject", "auto_rejected"}:
        return "auto_reject"
    if s in {"autoreview", "review"}:
        return "auto_review"
    if s in {"keep", "ok", "valid"}:
        return "keep"
    return s


def boolish(x) -> bool:
    return str(x or "").strip().lower() in {"1", "true", "yes", "hit", "reject"}


def discover_metric_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "multi_object_audit_rows_003D.csv",
        run_dir / "multi_object_audit_003D.csv",
        run_dir / "003D_multi_object_audit.csv",
        run_dir / "multi_object_audit_segments_003D.csv",
    ]
    candidates += sorted(run_dir.rglob("*003D*.csv"))
    candidates += sorted(run_dir.rglob("*multi*object*audit*.csv"))

    seen = set()
    for p in candidates:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if all(c in df.columns for c in REQUIRED_COLS):
            return p

    return None


def discover_file(run_dir: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return hits[0]
    return None


def apply_metric_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    id_col = first_col(df, ID_ALIASES)
    target_col = first_col(df, TARGET_ALIASES)
    ctx_col = first_col(df, CTX_ALIASES)

    if not id_col:
        raise RuntimeError("Colonne ID introuvable dans le CSV 003D.")
    if not target_col:
        raise RuntimeError("Colonne target introuvable dans le CSV 003D.")

    a = pd.to_numeric(df["inside_player_motion_mask_ratio_003C"], errors="coerce")
    b = pd.to_numeric(df["point_distance_to_table_px_median"], errors="coerce")
    c = pd.to_numeric(df["max_accel"], errors="coerce")

    hit = (a >= 0.516666) & (b <= 216.719) & (c >= 17.672)
    hit = hit.fillna(False)

    out = df.copy()
    out["review_id_003E"] = out[id_col].astype(str)
    out["target_class_003E"] = out[target_col].map(norm_target)

    if ctx_col:
        out["contextual_hit_003B4_resolved"] = out[ctx_col].map(boolish)
    else:
        out["contextual_hit_003B4_resolved"] = out["review_id_003E"].isin(CTX003B4_HITS)

    out["multi_object_shadow_hit_003E"] = hit
    out["multi_object_rule_003E"] = RULE_NAME
    out["multi_object_rule_detail_003E"] = RULE_TEXT
    out["multi_object_new_vs_003B4_003E"] = (
        out["multi_object_shadow_hit_003E"] & ~out["contextual_hit_003B4_resolved"]
    )

    return out, "metrics"


def apply_summary_replay(run_dir: Path) -> tuple[pd.DataFrame, str]:
    summary_path = discover_file(run_dir, ["*multi_object_audit_summary_003D.json", "*summary*003D*.json"])
    html_path = discover_file(run_dir, ["*multi_object_audit_003D.html", "*003D*.html"])

    if not summary_path or not html_path:
        raise RuntimeError(
            "Aucun CSV métrique 003D trouvé, et fallback HTML/JSON incomplet. "
            "Il faut soit le CSV complet 003D, soit multi_object_audit_003D.html + summary JSON."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    top = summary.get("top_rules", [{}])[0]
    caught = set(str(top.get("caught_ids", "")).replace(" ", "").split(","))
    caught.discard("")
    if not caught:
        caught = set(EXPECTED_HITS)

    tables = pd.read_html(html_path)
    rows = None
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        if "id" in cols and "target" in cols and "ctx003b4" in cols:
            rows = t.copy()
            break

    if rows is None:
        raise RuntimeError("Impossible de retrouver la table des 22 segments dans le HTML 003D.")

    id_col = first_col(rows, ID_ALIASES)
    target_col = first_col(rows, TARGET_ALIASES)
    ctx_col = first_col(rows, CTX_ALIASES)

    out = rows.copy()
    out["review_id_003E"] = out[id_col].astype(str)
    out["target_class_003E"] = out[target_col].map(norm_target)
    out["contextual_hit_003B4_resolved"] = out[ctx_col].map(boolish)
    out["multi_object_shadow_hit_003E"] = out["review_id_003E"].isin(caught)
    out["multi_object_rule_003E"] = RULE_NAME
    out["multi_object_rule_detail_003E"] = (
        RULE_TEXT + " ; summary_replay_from_003D_top_rule"
    )
    out["multi_object_new_vs_003B4_003E"] = (
        out["multi_object_shadow_hit_003E"] & ~out["contextual_hit_003B4_resolved"]
    )

    return out, "summary_replay"


def summarize(out: pd.DataFrame, mode: str, source: str) -> dict:
    caught_df = out[out["multi_object_shadow_hit_003E"]]
    caught_ids = caught_df["review_id_003E"].astype(str).tolist()

    keep_ids = caught_df[caught_df["target_class_003E"] == "keep"]["review_id_003E"].astype(str).tolist()
    partial_ids = caught_df[caught_df["target_class_003E"] == "human_partial"]["review_id_003E"].astype(str).tolist()

    prev_ids = out[out["contextual_hit_003B4_resolved"]]["review_id_003E"].astype(str).tolist()
    new_ids = sorted(set(caught_ids) - set(prev_ids))

    expected_missing = sorted(EXPECTED_HITS - set(caught_ids))
    expected_extra = sorted(set(caught_ids) - EXPECTED_HITS)

    status = "OK"
    warnings = []

    if keep_ids or partial_ids:
        status = "UNSAFE"

    if keep_ids:
        warnings.append("keep caught: " + ",".join(keep_ids))
    if partial_ids:
        warnings.append("human_partial caught: " + ",".join(partial_ids))
    if expected_missing:
        warnings.append("expected hits missing: " + ",".join(expected_missing))
    if expected_extra:
        warnings.append("unexpected hits: " + ",".join(expected_extra))
    if mode == "summary_replay":
        warnings.append("summary_replay = replay exact du top rule 003D ; OK pour shadow, pas pour généralisation finale.")

    return {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "mode": mode,
        "source": source,
        "rule_name": RULE_NAME,
        "rule": RULE_TEXT,
        "rows": int(len(out)),
        "caught_total": int(len(caught_ids)),
        "caught_ids": caught_ids,
        "contextual_003B4_caught_ids": prev_ids,
        "new_vs_003B4_ids": new_ids,
        "keep_caught": int(len(keep_ids)),
        "keep_caught_ids": keep_ids,
        "human_partial_caught": int(len(partial_ids)),
        "human_partial_caught_ids": partial_ids,
        "expected_missing": expected_missing,
        "expected_extra": expected_extra,
        "warnings": warnings,
    }


def write_html(path: Path, out: pd.DataFrame, summary: dict) -> None:
    cols = [
        "review_id_003E",
        "target_class_003E",
        "contextual_hit_003B4_resolved",
        "multi_object_shadow_hit_003E",
        "multi_object_new_vs_003B4_003E",
        "multi_object_rule_detail_003E",
    ]

    css = """
    body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
    section{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
    table{width:100%;border-collapse:collapse;margin-top:10px}
    td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
    th{background:#20242e}
    .hit td{box-shadow:inset 3px 0 0 rgba(116,217,159,.95)}
    .new td{background:rgba(116,217,159,.08)}
    .unsafe td{background:rgba(255,80,80,.13)}
    pre{white-space:pre-wrap;color:#dce4ff}
    """

    rows_html = []
    for _, r in out.iterrows():
        cls = []
        if bool(r["multi_object_shadow_hit_003E"]):
            cls.append("hit")
        if bool(r["multi_object_new_vs_003B4_003E"]):
            cls.append("new")
        if bool(r["multi_object_shadow_hit_003E"]) and r["target_class_003E"] in {"keep", "human_partial"}:
            cls.append("unsafe")
        tds = "".join(f"<td>{str(r.get(c, ''))}</td>" for c in cols)
        rows_html.append(f"<tr class='{' '.join(cls)}'>{tds}</tr>")

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003E multi-object shadow</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003E multi-object arbiter shadow</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Lignes</h2>
<table>
<thead><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
</section>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--input", default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"Run introuvable: {run_dir}")

    source = None

    if args.input:
        source = Path(args.input)
        df = pd.read_csv(source)
        out, mode = apply_metric_rule(df)
    else:
        source = discover_metric_csv(run_dir)
        if source:
            df = pd.read_csv(source)
            out, mode = apply_metric_rule(df)
        else:
            out, mode = apply_summary_replay(run_dir)
            source = discover_file(run_dir, ["*multi_object_audit_003D.html", "*003D*.html"])

    summary = summarize(out, mode, str(source))

    out_csv = run_dir / "multi_object_arbiter_shadow_003E.csv"
    out_json = run_dir / "multi_object_arbiter_shadow_summary_003E.json"
    out_html = run_dir / "multi_object_arbiter_shadow_003E.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print(f"003E status={summary['status']} mode={summary['mode']}")
    print(f"caught={','.join(summary['caught_ids'])}")
    print(f"new_vs_003B4={','.join(summary['new_vs_003B4_ids']) or '-'}")
    print(f"keep_caught={summary['keep_caught']} human_partial_caught={summary['human_partial_caught']}")
    for w in summary["warnings"]:
        print("warning:", w)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and (
        summary["status"] != "OK"
        or summary["expected_missing"]
        or summary["expected_extra"]
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
