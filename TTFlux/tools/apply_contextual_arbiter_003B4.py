from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


RULE = {
    "name": "003B3_top1_table_context_reject",
    "source": "contextual_audit_003B3 rank 1",
    "conditions": [
        {"feature": "point_inside_table_quad_ratio", "op": "<=", "threshold": 0.078599},
        {"feature": "v_range", "op": "<=", "threshold": 2.18656},
        {"feature": "below_table_risk", "op": "<=", "threshold": 0.2625},
    ],
    "expected_caught_ids_from_003B3": ["R0010", "R0009", "R0020", "R0008", "R0021"],
    "safety_policy": {
        "shadow_only": True,
        "do_not_override_keep": True,
        "do_not_override_human_partial": True,
        "require_has_table_model": True,
    },
}


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "target_class",
        "final_decision_002B",
        "proposed_decision_003B4",
        "changed_by_003B4",
        "contextual_reject_003B4",
        "contextual_rule_003B4",
        "clip_id_003B2",
        "segment_name",
        "has_table_model",
        "point_inside_table_quad_ratio",
        "v_range",
        "below_table_risk",
        "table_context_score_003B2",
        "median_accel",
        "sharp_turn_ratio",
    ]

    cols = []
    seen = set()

    for c in preferred:
        cols.append(c)
        seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def condition_pass(row: dict[str, Any], cond: dict[str, Any]) -> bool:
    value = fnum(row.get(cond["feature"]), None)

    if value is None:
        return False

    threshold = float(cond["threshold"])

    if cond["op"] == "<=":
        return value <= threshold

    if cond["op"] == ">=":
        return value >= threshold

    return False


def rule_pass(row: dict[str, Any]) -> bool:
    has_table = int(fnum(row.get("has_table_model"), 0) or 0) == 1

    if not has_table:
        return False

    return all(condition_pass(row, c) for c in RULE["conditions"])


def decision_before(row: dict[str, Any]) -> str:
    for col in ["final_decision_002B", "decision_001U", "action_001U", "action"]:
        v = str(row.get(col, "")).strip()
        if v:
            return v
    return "unknown"


def apply_shadow(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)

    before = decision_before(out)
    target = str(out.get("target_class", "")).strip()
    flag = rule_pass(out)

    # Shadow mode : on calcule une proposition, mais on conserve une politique prudente.
    proposed = before
    reason = ""

    if flag:
        reason = RULE["name"]

        if before == "keep":
            proposed = before
            reason += " | safety_keep_not_overridden"
        elif target == "human_partial":
            proposed = before
            reason += " | safety_human_partial_not_overridden"
        else:
            proposed = "reject"

    out["contextual_reject_003B4"] = 1 if flag else 0
    out["contextual_rule_003B4"] = reason
    out["decision_before_003B4"] = before
    out["proposed_decision_003B4"] = proposed
    out["changed_by_003B4"] = 1 if proposed != before else 0

    return out


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}

    for row in rows:
        v = str(row.get(key, "")).strip()
        out[v] = out.get(v, 0) + 1

    return dict(sorted(out.items()))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = [r for r in rows if int(fnum(r.get("contextual_reject_003B4"), 0) or 0) == 1]
    changed = [r for r in rows if int(fnum(r.get("changed_by_003B4"), 0) or 0) == 1]

    caught_by_target = count_by(caught, "target_class")

    safety = {
        "caught_keep": [r.get("review_id", "") for r in caught if r.get("target_class") == "keep"],
        "caught_human_partial": [r.get("review_id", "") for r in caught if r.get("target_class") == "human_partial"],
        "caught_auto_review": [r.get("review_id", "") for r in caught if r.get("target_class") == "auto_review"],
    }

    return {
        "version": "003B4",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "rule": RULE,
        "contextual_reject_count": len(caught),
        "contextual_reject_ids": [r.get("review_id", "") for r in caught],
        "contextual_reject_by_target": caught_by_target,
        "changed_count": len(changed),
        "changed_ids": [r.get("review_id", "") for r in changed],
        "decision_before_counts": count_by(rows, "decision_before_003B4"),
        "proposed_decision_counts": count_by(rows, "proposed_decision_003B4"),
        "safety": safety,
        "status": "shadow_only",
        "interpretation": [
            "This is a contextual reject flag, not a production override.",
            "The current top rule catches human_reject/auto_reject only on this batch.",
            "It must be validated on more clips before being hardened.",
            "Next recommended step is 003C coarse player/silhouette masks."
        ],
    }


def write_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    conds = " AND ".join(
        f'{c["feature"]} {c["op"]} {c["threshold"]}'
        for c in RULE["conditions"]
    )

    segment_rows = []

    for row in rows:
        flag = int(fnum(row.get("contextual_reject_003B4"), 0) or 0)
        changed = int(fnum(row.get("changed_by_003B4"), 0) or 0)
        target = str(row.get("target_class", "")).replace("_", "-")

        cls = target
        if flag:
            cls += " contextual-hit"
        if changed:
            cls += " changed"

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(row.get("review_id", "")))}</td>
<td>{html.escape(str(row.get("target_class", "")))}</td>
<td>{html.escape(str(row.get("decision_before_003B4", "")))}</td>
<td>{html.escape(str(row.get("proposed_decision_003B4", "")))}</td>
<td>{html.escape(str(row.get("contextual_reject_003B4", "")))}</td>
<td>{html.escape(str(row.get("changed_by_003B4", "")))}</td>
<td>{html.escape(str(row.get("clip_id_003B2", "")))}</td>
<td><code>{html.escape(str(row.get("segment_name", "")))}</code></td>
<td>{html.escape(str(row.get("has_table_model", "")))}</td>
<td>{html.escape(str(row.get("point_inside_table_quad_ratio", "")))}</td>
<td>{html.escape(str(row.get("v_range", "")))}</td>
<td>{html.escape(str(row.get("below_table_risk", "")))}</td>
<td>{html.escape(str(row.get("table_context_score_003B2", "")))}</td>
</tr>
""")

    caught_target_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["contextual_reject_by_target"].items()
    )

    before_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["decision_before_counts"].items()
    )

    after_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["proposed_decision_counts"].items()
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux contextual arbiter 003B4</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}}
th{{background:#20242e;position:sticky;top:0}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.human-reject td{{background:rgba(255,80,80,.09)}}
.human-partial td{{background:rgba(255,200,80,.08)}}
.keep td{{background:rgba(100,255,150,.055)}}
.auto-reject td{{background:rgba(255,80,80,.045)}}
.auto-review td{{background:rgba(150,170,255,.045)}}
.contextual-hit td{{outline:1px solid rgba(116,217,159,.65);background:rgba(116,217,159,.07)}}
.changed td{{font-weight:600}}
.warn{{color:#ffd37a}}
.ok{{color:#74d99f}}
</style>
</head>
<body>
<h1>TTFlux · contextual arbiter 003B4</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>rows</th><td>{summary["rows"]}</td></tr>
<tr><th>status</th><td class="warn">{html.escape(summary["status"])}</td></tr>
<tr><th>contextual_reject_count</th><td>{summary["contextual_reject_count"]}</td></tr>
<tr><th>contextual_reject_ids</th><td><code>{html.escape(", ".join(summary["contextual_reject_ids"]))}</code></td></tr>
<tr><th>changed_count</th><td>{summary["changed_count"]}</td></tr>
<tr><th>changed_ids</th><td><code>{html.escape(", ".join(summary["changed_ids"]))}</code></td></tr>
</table>
</section>

<section>
<h2>Règle shadow</h2>
<pre>{html.escape(conds)}</pre>
<p class="warn">Cette règle ne doit pas encore être durcie en production. Elle sert de signal contextuel.</p>
</section>

<section>
<h2>Caught by target</h2>
<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{caught_target_rows}</tbody>
</table>
</section>

<section>
<h2>Décisions avant / proposées</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<section>
<h3>Avant</h3>
<table><thead><tr><th>decision</th><th>count</th></tr></thead><tbody>{before_rows}</tbody></table>
</section>
<section>
<h3>Proposées 003B4</h3>
<table><thead><tr><th>decision</th><th>count</th></tr></thead><tbody>{after_rows}</tbody></table>
</section>
</div>
</section>

<section>
<h2>Segments</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>before</th><th>proposed</th><th>flag</th><th>changed</th>
<th>clip</th><th>segment</th><th>table</th><th>inside</th><th>v_range</th><th>below</th><th>table score</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contextual-csv", default="runs/batch_001E/contextual_audit_003B3/contextual_features_003B3.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/contextual_arbiter_003B4")
    args = parser.parse_args()

    contextual_csv = Path(args.contextual_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_csv(contextual_csv)
    rows = [apply_shadow(r) for r in base_rows]
    summary = summarize(rows)

    csv_path = out_dir / "contextual_arbiter_003B4_all.csv"
    json_path = out_dir / "contextual_arbiter_summary_003B4.json"
    html_path = out_dir / "contextual_arbiter_003B4.html"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, rows, summary)

    print(f"[003B4] rows                    : {summary['rows']}")
    print(f"[003B4] contextual rejects      : {summary['contextual_reject_count']}")
    print(f"[003B4] contextual reject ids   : {summary['contextual_reject_ids']}")
    print(f"[003B4] caught by target        : {summary['contextual_reject_by_target']}")
    print(f"[003B4] changed                 : {summary['changed_count']}")
    print(f"[003B4] out dir                 : {out_dir}")
    print(f"[003B4] html                    : {html_path}")


if __name__ == "__main__":
    main()
