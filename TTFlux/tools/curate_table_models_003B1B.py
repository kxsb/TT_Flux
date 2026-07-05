from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def normalize_quad(q: Any) -> list[Any]:
    labels = ["front_left", "front_right", "back_right", "back_left"]

    if not isinstance(q, list):
        q = []

    out = []

    for i in range(4):
        if i < len(q) and valid_point(q[i]):
            p = q[i]
            out.append({
                "label": p.get("label", labels[i]),
                "x": float(p["x"]),
                "y": float(p["y"]),
            })
        else:
            out.append(None)

    return out


def quad_count(q: Any) -> int:
    return sum(1 for p in normalize_quad(q) if valid_point(p))


def is_complete(q: Any) -> bool:
    return quad_count(q) == 4


def get_best_manual_clip(manual_clips: dict[str, Any], clip_id: str) -> dict[str, Any] | None:
    clip = manual_clips.get(clip_id)

    if not isinstance(clip, dict):
        return None

    # Nouveau format keyframes éventuel.
    if isinstance(clip.get("table_keyframes"), list):
        best = None
        best_count = -1

        for kf in clip["table_keyframes"]:
            if not isinstance(kf, dict):
                continue

            q = normalize_quad(kf.get("table_quad"))
            count = quad_count(q)

            if count > best_count:
                best_count = count
                best = {
                    "frame_ref": int(float(kf.get("frame") or 0)),
                    "table_quad": q,
                    "source": "manual_keyframe",
                    "manual_updated_at": kf.get("updated_at"),
                }

        if best and best_count == 4:
            return best

    # Ancien format 003A.
    q = normalize_quad(clip.get("table_quad"))

    if is_complete(q):
        return {
            "frame_ref": int(float(clip.get("frame_ref") or 0)),
            "table_quad": q,
            "source": "manual_seed_003A",
            "manual_updated_at": clip.get("updated_at"),
        }

    return None


def build_curated(
    payload: dict[str, Any],
    manual: dict[str, Any],
    bootstrap: dict[str, Any],
    reject_manual: set[str],
    allow_auto: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clips = payload.get("clips", [])
    manual_clips = manual.get("clips", {}) if isinstance(manual.get("clips"), dict) else {}
    auto_models = bootstrap.get("table_models", {}) if isinstance(bootstrap.get("table_models"), dict) else {}

    curated_models: dict[str, Any] = {}
    rows = []

    for clip in clips:
        clip_id = clip["clip_id"]
        manual_best = get_best_manual_clip(manual_clips, clip_id)
        auto = auto_models.get(clip_id)

        decision = "missing"
        reason = "no_manual_model"
        table_quad = [None, None, None, None]
        frame_ref = 0
        confidence = 0.0
        source = "missing"

        if clip_id in reject_manual:
            reason = "manual_rejected_by_user"
        elif manual_best is not None:
            decision = "use_manual"
            reason = "manual_complete_and_not_rejected"
            table_quad = manual_best["table_quad"]
            frame_ref = manual_best["frame_ref"]
            confidence = 0.90
            source = manual_best["source"]
        elif clip_id in allow_auto and isinstance(auto, dict) and is_complete(auto.get("table_quad")):
            decision = "use_auto"
            reason = "explicitly_allowed_auto"
            table_quad = normalize_quad(auto.get("table_quad"))
            frame_ref = int(float(auto.get("frame_ref") or 0))
            confidence = min(0.65, float(auto.get("confidence") or 0.0))
            source = auto.get("source", "auto")
        else:
            decision = "missing"
            if isinstance(auto, dict) and is_complete(auto.get("table_quad")):
                reason = "auto_available_but_not_trusted"
            else:
                reason = "no_complete_model"

        model = {
            "clip_id": clip_id,
            "filename": clip.get("filename", ""),
            "video_path": clip.get("video_path", ""),
            "width": clip.get("width", 0),
            "height": clip.get("height", 0),
            "fps": clip.get("fps", 0),
            "frame_count": clip.get("frame_count", 0),
            "frame_ref": frame_ref,
            "table_quad": table_quad,
            "has_table_model": is_complete(table_quad),
            "source": source,
            "decision_003B1B": decision,
            "reason_003B1B": reason,
            "confidence_003B1B": round(float(confidence), 6),
            "manual_available": manual_best is not None,
            "auto_available": isinstance(auto, dict) and is_complete(auto.get("table_quad")),
            "auto_source": auto.get("source", "") if isinstance(auto, dict) else "",
            "auto_confidence": auto.get("confidence", "") if isinstance(auto, dict) else "",
        }

        curated_models[clip_id] = model

        rows.append({
            "clip_id": clip_id,
            "filename": clip.get("filename", ""),
            "decision": decision,
            "reason": reason,
            "source": source,
            "confidence": model["confidence_003B1B"],
            "manual_available": model["manual_available"],
            "auto_available": model["auto_available"],
            "auto_confidence": model["auto_confidence"],
        })

    complete = [cid for cid, m in curated_models.items() if m["has_table_model"]]
    missing = [cid for cid, m in curated_models.items() if not m["has_table_model"]]

    result = {
        "version": "003B1B",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Curated table models: manual-first, auto only when explicitly allowed.",
        "policy": {
            "manual_first": True,
            "auto_replaces_manual": False,
            "auto_requires_explicit_allow": True,
            "reject_manual_clip_ids": sorted(reject_manual),
            "allow_auto_clip_ids": sorted(allow_auto),
        },
        "table_models": curated_models,
    }

    summary = {
        "clips_total": len(curated_models),
        "clips_with_table_model": len(complete),
        "clips_missing_table_model": len(missing),
        "complete_clip_ids": complete,
        "missing_clip_ids": missing,
        "rows": rows,
        "warning": "Use missing clips as has_table_model=0 in 003B2. Do not force bad table geometry.",
    }

    return result, summary


def write_html(path: Path, result: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = []

    for r in summary["rows"]:
        cls = "ok" if r["decision"] == "use_manual" else ("warn" if r["decision"] == "missing" else "auto")

        rows.append(f"""
<tr class="{cls}">
<td>{html.escape(str(r["clip_id"]))}</td>
<td>{html.escape(str(r["decision"]))}</td>
<td>{html.escape(str(r["reason"]))}</td>
<td>{html.escape(str(r["source"]))}</td>
<td>{html.escape(str(r["confidence"]))}</td>
<td>{html.escape(str(r["manual_available"]))}</td>
<td>{html.escape(str(r["auto_available"]))}</td>
<td>{html.escape(str(r["auto_confidence"]))}</td>
</tr>
""")

    detail_sections = []

    for clip_id, m in result["table_models"].items():
        detail_sections.append(f"""
<section>
<h2>{html.escape(clip_id)} · {html.escape(str(m["decision_003B1B"]))}</h2>
<p><b>Fichier :</b> {html.escape(str(m["filename"]))}</p>
<p><b>Raison :</b> {html.escape(str(m["reason_003B1B"]))}</p>
<p><b>Source :</b> {html.escape(str(m["source"]))} · <b>confiance :</b> {html.escape(str(m["confidence_003B1B"]))}</p>
<pre>{html.escape(json.dumps(m["table_quad"], indent=2, ensure_ascii=False))}</pre>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux curated table models 003B1B</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:13px}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px;color:#dce4ff}}
.ok td{{background:rgba(100,255,150,.055)}}
.warn td{{background:rgba(255,200,80,.075)}}
.auto td{{background:rgba(150,170,255,.055)}}
</style>
</head>
<body>
<h1>TTFlux · curated table models 003B1B</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>clips_total</th><td>{summary["clips_total"]}</td></tr>
<tr><th>clips_with_table_model</th><td>{summary["clips_with_table_model"]}</td></tr>
<tr><th>clips_missing_table_model</th><td>{summary["clips_missing_table_model"]}</td></tr>
<tr><th>missing_clip_ids</th><td>{html.escape(", ".join(summary["missing_clip_ids"]))}</td></tr>
<tr><th>warning</th><td>{html.escape(summary["warning"])}</td></tr>
</table>
</section>

<section>
<h2>Décisions</h2>
<table>
<thead>
<tr>
<th>clip</th><th>decision</th><th>reason</th><th>source</th><th>confidence</th>
<th>manual</th><th>auto</th><th>auto_conf</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</section>

{''.join(detail_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="runs/batch_001E/table_scene_003A/table_annotation_payload_003A.json")
    parser.add_argument("--manual-json", default="runs/batch_001E/table_scene_003A/table_annotations_003A_merged.json")
    parser.add_argument("--bootstrap-json", default="runs/batch_001E/table_bootstrap_003B1/table_bootstrap_models_003B1.json")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_bootstrap_003B1B")
    parser.add_argument("--reject-manual", default="C001_v62_0001_v61_1_M2forQBQaZc_s01")
    parser.add_argument("--allow-auto", default="")
    args = parser.parse_args()

    payload = read_json(Path(args.payload))
    manual = read_json(Path(args.manual_json)) or {}
    bootstrap = read_json(Path(args.bootstrap_json)) or {}

    if not payload or not isinstance(payload.get("clips"), list):
        raise SystemExit("[003B1B] Payload invalide.")

    reject_manual = {x.strip() for x in args.reject_manual.split(",") if x.strip()}
    allow_auto = {x.strip() for x in args.allow_auto.split(",") if x.strip()}

    result, summary = build_curated(
        payload=payload,
        manual=manual,
        bootstrap=bootstrap,
        reject_manual=reject_manual,
        allow_auto=allow_auto,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "table_models_curated_003B1B.json"
    summary_path = out_dir / "table_models_curated_summary_003B1B.json"
    html_path = out_dir / "table_models_curated_003B1B.html"

    write_json(result_path, result)
    write_json(summary_path, summary)
    write_html(html_path, result, summary)

    print(f"[003B1B] clips total      : {summary['clips_total']}")
    print(f"[003B1B] with model       : {summary['clips_with_table_model']}")
    print(f"[003B1B] missing          : {summary['clips_missing_table_model']}")
    print(f"[003B1B] missing ids      : {summary['missing_clip_ids']}")
    print(f"[003B1B] out dir          : {out_dir}")
    print(f"[003B1B] html             : {html_path}")


if __name__ == "__main__":
    main()
