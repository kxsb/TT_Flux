from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ttflux.core.paths import project_root, legacy_root, ensure_base_dirs
from ttflux.video.reader import read_video_meta, extract_frame_jpeg, read_frame_bgr, encode_frame_jpeg
from ttflux.video.indexer import build_video_library
from ttflux.video.contact_sheet import make_contact_sheet_jpeg
from ttflux.labels.session import create_label_session, list_sessions, read_session, read_annotations, append_annotation, delete_annotation, export_annotations_csv
from ttflux.scene.legacy_table_import import build_scene_table_report, load_scene_rows_near_frame, draw_scene_rows, list_scene_table_clips, topdown_payload_for_video, render_topdown_for_video
from ttflux.viz.labels_overlay import draw_label_annotations
from ttflux.legacy.import_index import build_legacy_artifact_report
from ttflux.datasets.legacy_dataset_audit import load_dataset_audit, dataset_pools, list_transversal_analyzed_videos
from ttflux.datasets.transversal_registry import load_transversal_registry, load_active_profile, role_records
from ttflux.datasets.transversal_clip_index import load_transversal_clip_index, list_clips
from ttflux.datasets.external_dataset_registry import load_external_dataset_registry, list_external_videos, external_dataset_summary
from ttflux.datasets.transversal_clip_index import list_unified_raw_clips
from ttflux.datasets.external_sample_index import load_external_sample_index, list_external_samples, sample_detail
from ttflux.datasets.openttgames_annotation_probe import probe_openttgames_annotations, extract_ball_points_for_sample, load_openttgames_ball_points_index
from ttflux.datasets.openttgames_ball_overlay import find_ball_points_for_frame, render_external_gt_frame_jpeg
from ttflux.datasets.openttgames_gt_contact_sheet import build_gt_contact_sheet, contact_sheet_jpeg
from ttflux.datasets.openttgames_gt_audit import load_openttgames_gt_audit
from ttflux.datasets.openttgames_gt_segments import load_openttgames_gt_segments, segment_points
from ttflux.datasets.openttgames_training_rows import load_openttgames_training_rows, csv_paths

ensure_base_dirs()

PROJECT_ROOT = project_root()
LEGACY_ROOT = legacy_root()
STATIC_DIR = PROJECT_ROOT / "app" / "static"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
LEGACY_INDEX_PATH = PROJECT_ROOT / "legacy_imports" / "legacy_artifacts_index.json"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

app = FastAPI(title="TTFlux V2", version="0.0.1")


class AnnotationIn(BaseModel):
    session_id: str
    frame: int
    label_type: str
    x: float | None = None
    y: float | None = None
    visible: bool = True
    object_id: str = ""
    notes: str = ""
    payload: dict[str, Any] = {}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def allowed_roots() -> list[Path]:
    roots = [
        PROJECT_ROOT,
        LEGACY_ROOT,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "videos",
    ]
    out = []
    for r in roots:
        try:
            out.append(r.resolve())
        except Exception:
            pass
    return out


def resolve_safe_path(path_value: str) -> Path:
    if not path_value:
        raise HTTPException(status_code=400, detail="missing_path")

    raw = Path(path_value).expanduser()
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw

    try:
        resolved = raw.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"path_resolve_failed: {exc}") from exc

    roots = allowed_roots()
    if not any(str(resolved).lower().startswith(str(root).lower()) for root in roots):
        raise HTTPException(status_code=403, detail="path_outside_allowed_roots")

    return resolved


def video_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]


def configured_scan_roots() -> list[Path]:
    cfg = read_json(CONFIG_PATH, {})
    roots = []
    for item in cfg.get("video_scan_roots", []):
        p = Path(item)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        roots.append(p)

    roots.append(PROJECT_ROOT / "data" / "videos")

    unique = []
    seen = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            continue
        key = str(rr).lower()
        if key not in seen:
            seen.add(key)
            unique.append(rr)
    return unique


def scan_videos(max_results: int = 250) -> list[dict[str, Any]]:
    results = []
    seen = set()

    for root in configured_scan_roots():
        if not root.exists():
            continue

        try:
            iterator = root.rglob("*")
            for p in iterator:
                if len(results) >= max_results:
                    return results
                if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                    continue
                rp = p.resolve()
                key = str(rp).lower()
                if key in seen:
                    continue
                seen.add(key)
                stat = rp.stat()
                results.append(
                    {
                        "id": video_id_for(rp),
                        "name": rp.name,
                        "path": str(rp),
                        "relative_to_project": str(rp.relative_to(PROJECT_ROOT)) if str(rp).lower().startswith(str(PROJECT_ROOT).lower()) else "",
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    }
                )
        except Exception:
            continue

    return results


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/debug-scene-clips", response_class=HTMLResponse)
def debug_scene_clips_page() -> HTMLResponse:
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Debug scene clips</title>
  <style>
    body { background:#0b1020; color:#e5e7eb; font-family:system-ui; padding:20px; }
    pre { background:#050816; border:1px solid #26324f; border-radius:8px; padding:12px; white-space:pre-wrap; }
    button { padding:10px 14px; margin-right:8px; background:#13213d; color:#e5e7eb; border:1px solid #26324f; border-radius:8px; }
  </style>
</head>
<body>
  <h1>Debug scene clips</h1>
  <button onclick="loadTrusted()">Trusted</button>
  <button onclick="loadAll()">All</button>
  <button onclick="loadPaths()">Paths</button>
  <pre id="out">ready</pre>
<script>
async function j(url) {
  const r = await fetch(url + (url.includes("?") ? "&" : "?") + "t=" + Date.now());
  const txt = await r.text();
  try { return JSON.stringify(JSON.parse(txt), null, 2); } catch(e) { return txt; }
}
async function loadTrusted() {
  document.getElementById("out").textContent = await j("/api/scene/table-clips?status_filter=METRIC_TABLE_TRUSTED&only_existing=true&limit=25");
}
async function loadAll() {
  document.getElementById("out").textContent = await j("/api/scene/table-clips?status_filter=&only_existing=true&limit=50");
}
async function loadPaths() {
  document.getElementById("out").textContent = await j("/api/debug/paths");
}
loadTrusted();
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/api/debug/paths")
def api_debug_paths() -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "project_exists": PROJECT_ROOT.exists(),
        "legacy_root": str(LEGACY_ROOT),
        "legacy_exists": LEGACY_ROOT.exists(),
        "env_TTFLUX_LEGACY_ROOT": os.environ.get("TTFLUX_LEGACY_ROOT", ""),
        "allowed_roots": [str(p) for p in allowed_roots()],
        "scene_csv_exists": (LEGACY_ROOT / "runs" / "rally_scene_table_objects_005C9B" / "scene_table_objects_005C9B.csv").exists(),
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "legacy_root": str(LEGACY_ROOT),
        "legacy_exists": LEGACY_ROOT.exists(),
    }


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    cfg = read_json(CONFIG_PATH, {})
    return {
        "config": cfg,
        "project_root": str(PROJECT_ROOT),
        "legacy_root": str(LEGACY_ROOT),
        "scan_roots": [str(p) for p in configured_scan_roots()],
    }


@app.get("/api/legacy-artifacts")
def api_legacy_artifacts() -> dict[str, Any]:
    payload = read_json(LEGACY_INDEX_PATH, {})
    legacy = Path(payload.get("legacy_root") or str(LEGACY_ROOT))
    checks = {}

    for group_key in ["validated_table_chain", "partial_ball_chain_reference"]:
        checks[group_key] = []
        for rel in payload.get(group_key, []):
            p = legacy / rel
            checks[group_key].append(
                {
                    "relative_path": rel,
                    "absolute_path": str(p),
                    "exists": p.exists(),
                    "kind": "dir" if p.is_dir() else "file" if p.is_file() else "missing",
                }
            )

    return {
        "index": payload,
        "checks": checks,
    }


@app.get("/api/videos")
def api_videos(max_results: int = Query(250, ge=1, le=2000)) -> dict[str, Any]:
    return {
        "videos": scan_videos(max_results=max_results),
        "scan_roots": [str(p) for p in configured_scan_roots()],
    }


@app.get("/api/video-meta")
def api_video_meta(path: str) -> dict[str, Any]:
    p = resolve_safe_path(path)
    return read_video_meta(p)


@app.get("/api/scene/table-clips")
def api_scene_table_clips(
    status_filter: str = Query(""),
    only_existing: bool = Query(True),
    limit: int = Query(250, ge=1, le=1000),
) -> dict[str, Any]:
    return list_scene_table_clips(
        status_filter=status_filter,
        only_existing=only_existing,
        limit=limit,
    )


@app.get("/api/scene/topdown-data")
def api_scene_topdown_data(path: str) -> dict[str, Any]:
    p = resolve_safe_path(path)
    return topdown_payload_for_video(str(p))


@app.get("/api/scene/topdown-image")
def api_scene_topdown_image(
    path: str,
    frame: int = Query(0, ge=0),
    width: int = Query(900, ge=420, le=1600),
    height: int = Query(560, ge=280, le=1000),
) -> Response:
    p = resolve_safe_path(path)
    try:
        jpg = render_topdown_for_video(str(p), frame=frame, width=width, height=height)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/scene/legacy-table-summary")
def api_scene_legacy_table_summary(rebuild: bool = Query(False)) -> dict[str, Any]:
    out_dir = PROJECT_ROOT / "runs" / "007E_scene_table_import"
    json_path = out_dir / "scene_table_import_summary_007E.json"

    if rebuild or not json_path.exists():
        return build_scene_table_report(out_dir)

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return build_scene_table_report(out_dir)


@app.get("/api/scene/legacy-table-rows")
def api_scene_legacy_table_rows(
    frame: int = Query(0, ge=0),
    tol: int = Query(0, ge=0, le=30),
    limit: int = Query(80, ge=1, le=500),
    path: str = Query(""),
) -> dict[str, Any]:
    video_path = ""
    if path:
        try:
            video_path = str(resolve_safe_path(path))
        except Exception:
            video_path = path
    return load_scene_rows_near_frame(frame, tol=tol, limit=limit, video_path=video_path)


@app.get("/api/frame-overlay")
def api_frame_overlay(
    path: str,
    frame: int = Query(0, ge=0),
    quality: int = Query(90, ge=40, le=100),
    legacy_scene: bool = Query(False),
    scene_tol: int = Query(0, ge=0, le=30),
    label_session_id: str = Query(""),
) -> Response:
    p = resolve_safe_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="video_not_found")

    try:
        img = read_frame_bgr(p, frame)

        if legacy_scene:
            scene_payload = load_scene_rows_near_frame(frame, tol=scene_tol, limit=80, video_path=str(p))
            img = draw_scene_rows(img, scene_payload)

        if label_session_id:
            anns = read_annotations(label_session_id)
            img = draw_label_annotations(img, anns, frame)

        jpg = encode_frame_jpeg(img, quality=quality)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/frame")
def api_frame(path: str, frame: int = Query(0, ge=0), quality: int = Query(90, ge=40, le=100)) -> Response:
    p = resolve_safe_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="video_not_found")
    try:
        jpg = extract_frame_jpeg(p, frame_idx=frame, quality=quality)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=jpg, media_type="image/jpeg")


@app.post("/api/label-session")
def api_create_label_session(video_path: str, name: str | None = Query(None)) -> dict[str, Any]:
    p = resolve_safe_path(video_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="video_not_found")
    try:
        return create_label_session(str(p), name=name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/label-sessions")
def api_label_sessions() -> dict[str, Any]:
    return {"sessions": list_sessions()}


@app.get("/api/label-session/{session_id}")
def api_label_session(session_id: str) -> dict[str, Any]:
    try:
        return {
            "session": read_session(session_id),
            "annotations": read_annotations(session_id),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/annotation")
def api_add_annotation(payload: AnnotationIn) -> dict[str, Any]:
    try:
        ann = append_annotation(payload.session_id, payload.model_dump())
        return {"ok": True, "annotation": ann}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/annotation/{session_id}/{uid}")
def api_delete_annotation(session_id: str, uid: str) -> dict[str, Any]:
    try:
        return delete_annotation(session_id, uid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/label-session/{session_id}/csv")
def api_label_session_csv(session_id: str) -> Response:
    try:
        csv_text = export_annotations_csv(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=csv_text.encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.csv"'},
    )


@app.get("/api/video-library")
def api_video_library(
    rebuild: bool = Query(False),
    include_generated: bool = Query(False),
    include_unknown: bool = Query(False),
    max_results: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    out_dir = PROJECT_ROOT / "runs" / "007C_video_library"
    json_path = out_dir / "video_library_007C.json"

    if rebuild or not json_path.exists():
        return build_video_library(
            output_dir=out_dir,
            include_generated=include_generated,
            include_unknown=include_unknown,
            max_results=max_results,
        )

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            bool(payload.get("include_generated")) != bool(include_generated)
            or bool(payload.get("include_unknown")) != bool(include_unknown)
        ):
            return build_video_library(
                output_dir=out_dir,
                include_generated=include_generated,
                include_unknown=include_unknown,
                max_results=max_results,
            )
        return payload
    except Exception:
        return build_video_library(
            output_dir=out_dir,
            include_generated=include_generated,
            include_unknown=include_unknown,
            max_results=max_results,
        )


@app.get("/api/contact-sheet")
def api_contact_sheet(
    path: str,
    start_frame: int = Query(0, ge=0),
    end_frame: int | None = Query(None),
    count: int = Query(24, ge=1, le=80),
    cols: int = Query(6, ge=1, le=10),
    thumb_w: int = Query(220, ge=120, le=420),
    quality: int = Query(88, ge=40, le=100),
) -> Response:
    p = resolve_safe_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="video_not_found")

    try:
        jpg = make_contact_sheet_jpeg(
            p,
            start_frame=start_frame,
            end_frame=end_frame,
            count=count,
            cols=cols,
            thumb_w=thumb_w,
            quality=quality,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    roots = [
        ("v2", PROJECT_ROOT / "runs"),
        ("legacy", LEGACY_ROOT / "runs"),
    ]
    out = []
    for source, root in roots:
        if not root.exists():
            continue
        for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)[:80]:
            try:
                stat = p.stat()
                out.append(
                    {
                        "source": source,
                        "name": p.name,
                        "path": str(p),
                        "kind": "dir" if p.is_dir() else "file",
                        "modified": stat.st_mtime,
                    }
                )
            except Exception:
                continue
    return {"runs": out}


@app.get("/api/legacy-artifacts-detailed")
def api_legacy_artifacts_detailed(rebuild: bool = Query(False)) -> dict[str, Any]:
    out_dir = PROJECT_ROOT / "runs" / "007B_legacy_artifacts_index"
    json_path = out_dir / "legacy_artifacts_detailed_007B.json"

    if rebuild or not json_path.exists():
        return build_legacy_artifact_report(out_dir)

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return build_legacy_artifact_report(out_dir)


def _scan_mp4s_under(root: Path, limit: int = 400) -> list[dict[str, Any]]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    out = []

    if not root.exists():
        return out

    for p in root.rglob("*"):
        if len(out) >= limit:
            break
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        try:
            st = p.stat()
            out.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(st.st_size / 1024 / 1024, 2),
                "mtime": st.st_mtime,
            })
        except Exception:
            continue

    out.sort(key=lambda x: (x.get("mtime") or 0), reverse=True)
    return out


def _attach_light_video_meta(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []

    for item in items:
        obj = dict(item)
        path_value = obj.get("path") or ""

        if not path_value:
            out.append(obj)
            continue

        try:
            meta = read_video_meta(Path(path_value))
            obj["width"] = meta.get("width", 0)
            obj["height"] = meta.get("height", 0)
            obj["fps"] = meta.get("fps", 0.0)
            obj["frame_count"] = meta.get("frame_count", 0)
            obj["duration_sec"] = meta.get("duration_sec", 0.0)
            obj["backend_ok"] = meta.get("backend_ok", False)
            obj["video_error"] = meta.get("error", "")
        except Exception as exc:
            obj["width"] = 0
            obj["height"] = 0
            obj["fps"] = 0.0
            obj["frame_count"] = 0
            obj["duration_sec"] = 0.0
            obj["backend_ok"] = False
            obj["video_error"] = f"{type(exc).__name__}: {exc}"

        out.append(obj)

    return out


@app.get("/api/light/raw-videos")
def api_light_raw_videos(
    limit: int = Query(250, ge=1, le=1000),
    with_meta: bool = Query(False),
) -> dict[str, Any]:
    """
    Vidéos brutes utilisables pour la V2.
    Inclut :
    - data/videos V2
    - clips legacy dataset_rebuild_004S
    - clips table 005C9B, car ce sont nos clips propres de travail
    """
    items = []
    seen = set()

    roots = [
        PROJECT_ROOT / "data" / "videos",
        LEGACY_ROOT / "runs" / "dataset_rebuild_004S_rally_clips_balanced" / "clips",
        LEGACY_ROOT / "clips",
        LEGACY_ROOT / "videos",
    ]

    for root in roots:
        for item in _scan_mp4s_under(root, limit=limit):
            key = item["path"].lower()
            if key in seen:
                continue
            seen.add(key)
            item["source_root"] = str(root)
            items.append(item)
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    # Bonus : clips table 005C9B déjà validés.
    try:
        table = list_scene_table_clips(status_filter="", only_existing=True, limit=limit)
        for c in table.get("clips", []):
            path = c.get("clip_path", "")
            key = path.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "name": c.get("clip_name") or Path(path).name,
                "path": path,
                "size_mb": 0,
                "mtime": 0,
                "source_root": "scene_table_005C9B",
                "review_id": c.get("review_id", ""),
                "status": c.get("scene_table_status_005C9B", ""),
            })
            if len(items) >= limit:
                break
    except Exception:
        pass

    items = items[:limit]
    if with_meta:
        items = _attach_light_video_meta(items)
    return {
        "items": items,
        "count": len(items),
        "with_meta": with_meta,
    }


@app.get("/api/datasets/openttgames-training-rows")
def api_datasets_openttgames_training_rows(
    rebuild: bool = Query(False),
    rebuild_segments: bool = Query(False),
    gap_threshold: int = Query(30, ge=1, le=10000),
    min_valid_points: int = Query(16, ge=1, le=1000),
    include_unusable_segments: bool = Query(False),
) -> dict[str, Any]:
    return load_openttgames_training_rows(
        rebuild=rebuild,
        rebuild_segments=rebuild_segments,
        gap_threshold=gap_threshold,
        min_valid_points=min_valid_points,
        include_unusable_segments=include_unusable_segments,
    )


@app.get("/api/datasets/openttgames-training-rows-csv")
def api_datasets_openttgames_training_rows_csv(
    rebuild: bool = Query(False),
) -> FileResponse:
    path = csv_paths(rebuild=rebuild, kind="rows")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/datasets/openttgames-training-segments-csv")
def api_datasets_openttgames_training_segments_csv(
    rebuild: bool = Query(False),
) -> FileResponse:
    path = csv_paths(rebuild=rebuild, kind="segments")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/datasets/openttgames-gt-segments")
def api_datasets_openttgames_gt_segments(
    rebuild: bool = Query(False),
    gap_threshold: int = Query(30, ge=1, le=10000),
    min_valid_points: int = Query(16, ge=1, le=1000),
) -> dict[str, Any]:
    return load_openttgames_gt_segments(
        rebuild=rebuild,
        gap_threshold=gap_threshold,
        min_valid_points=min_valid_points,
    )


@app.get("/api/datasets/openttgames-gt-segments-html")
def api_datasets_openttgames_gt_segments_html(
    rebuild: bool = Query(False),
    gap_threshold: int = Query(30, ge=1, le=10000),
    min_valid_points: int = Query(16, ge=1, le=1000),
) -> FileResponse:
    payload = load_openttgames_gt_segments(
        rebuild=rebuild,
        gap_threshold=gap_threshold,
        min_valid_points=min_valid_points,
    )
    path = Path(payload.get("outputs", {}).get("html", ""))
    if not path.exists():
        payload = load_openttgames_gt_segments(rebuild=True, gap_threshold=gap_threshold, min_valid_points=min_valid_points)
        path = Path(payload.get("outputs", {}).get("html", ""))
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.get("/api/datasets/openttgames-gt-segment-points")
def api_datasets_openttgames_gt_segment_points(
    segment_uid: str = Query(""),
    sample_id: str = Query(""),
    segment_idx: int | None = Query(None),
    gap_threshold: int = Query(30, ge=1, le=10000),
    min_valid_points: int = Query(16, ge=1, le=1000),
    limit: int = Query(10000, ge=1, le=200000),
) -> dict[str, Any]:
    return segment_points(
        segment_uid=segment_uid,
        sample_id=sample_id,
        segment_idx=segment_idx,
        gap_threshold=gap_threshold,
        min_valid_points=min_valid_points,
        limit=limit,
    )


@app.get("/api/datasets/openttgames-gt-audit")
def api_datasets_openttgames_gt_audit(
    rebuild: bool = Query(False),
    rebuild_ball_index: bool = Query(False),
    build_sheets: bool = Query(True),
) -> dict[str, Any]:
    return load_openttgames_gt_audit(
        rebuild=rebuild,
        rebuild_ball_index=rebuild_ball_index,
        build_sheets=build_sheets,
    )


@app.get("/api/datasets/openttgames-gt-audit-html")
def api_datasets_openttgames_gt_audit_html(
    rebuild: bool = Query(False),
    rebuild_ball_index: bool = Query(False),
    build_sheets: bool = Query(True),
) -> FileResponse:
    payload = load_openttgames_gt_audit(
        rebuild=rebuild,
        rebuild_ball_index=rebuild_ball_index,
        build_sheets=build_sheets,
    )
    path = Path(payload.get("outputs", {}).get("html", ""))
    if not path.exists():
        payload = load_openttgames_gt_audit(rebuild=True)
        path = Path(payload.get("outputs", {}).get("html", ""))
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.get("/api/datasets/openttgames-gt-contact-sheet")
def api_datasets_openttgames_gt_contact_sheet(
    path: str = Query(...),
    sample_id: str = Query(""),
    max_tiles: int = Query(24, ge=1, le=80),
    tile_w: int = Query(320, ge=120, le=960),
    tile_h: int = Query(180, ge=90, le=540),
    cols: int = Query(4, ge=1, le=8),
    only_valid_xy: bool = Query(True),
    window: int = Query(0, ge=0, le=30),
) -> Response:
    data = contact_sheet_jpeg(
        video_path=path,
        sample_id=sample_id,
        max_tiles=max_tiles,
        tile_w=tile_w,
        tile_h=tile_h,
        cols=cols,
        only_valid_xy=only_valid_xy,
        window=window,
    )
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/datasets/openttgames-gt-contact-sheet-json")
def api_datasets_openttgames_gt_contact_sheet_json(
    path: str = Query(...),
    sample_id: str = Query(""),
    max_tiles: int = Query(24, ge=1, le=80),
    tile_w: int = Query(320, ge=120, le=960),
    tile_h: int = Query(180, ge=90, le=540),
    cols: int = Query(4, ge=1, le=8),
    only_valid_xy: bool = Query(True),
    window: int = Query(0, ge=0, le=30),
) -> dict[str, Any]:
    return build_gt_contact_sheet(
        video_path=path,
        sample_id=sample_id,
        max_tiles=max_tiles,
        tile_w=tile_w,
        tile_h=tile_h,
        cols=cols,
        only_valid_xy=only_valid_xy,
        window=window,
    )


@app.get("/api/datasets/openttgames-ball-frame-point")
def api_datasets_openttgames_ball_frame_point(
    path: str = Query(...),
    frame: int = Query(..., ge=0),
    window: int = Query(0, ge=0, le=30),
    rebuild: bool = Query(False),
) -> dict[str, Any]:
    return find_ball_points_for_frame(
        video_path=path,
        frame=frame,
        window=window,
        rebuild=rebuild,
    )


@app.get("/api/frame-external-gt")
def api_frame_external_gt(
    path: str = Query(...),
    frame: int = Query(..., ge=0),
    window: int = Query(0, ge=0, le=30),
    quality: int = Query(90, ge=40, le=100),
) -> Response:
    data = render_external_gt_frame_jpeg(
        video_path=path,
        frame=frame,
        window=window,
        quality=quality,
    )
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/datasets/openttgames-ball-points-index")
def api_datasets_openttgames_ball_points_index(
    rebuild: bool = Query(False),
) -> dict[str, Any]:
    return load_openttgames_ball_points_index(rebuild=rebuild)


@app.get("/api/datasets/openttgames-ball-points-csv")
def api_datasets_openttgames_ball_points_csv(
    rebuild: bool = Query(False),
) -> FileResponse:
    payload = load_openttgames_ball_points_index(rebuild=rebuild)
    csv_path = Path(payload.get("outputs", {}).get("csv", ""))
    if not csv_path.exists():
        payload = load_openttgames_ball_points_index(rebuild=True)
        csv_path = Path(payload.get("outputs", {}).get("csv", ""))
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=csv_path.name,
    )


@app.get("/api/datasets/openttgames-annotation-probe")
def api_datasets_openttgames_annotation_probe(
    sample_id: str = Query(""),
    rebuild_sample_index: bool = Query(False),
) -> dict[str, Any]:
    return probe_openttgames_annotations(
        sample_id=sample_id,
        rebuild_sample_index=rebuild_sample_index,
    )


@app.get("/api/datasets/openttgames-ball-points")
def api_datasets_openttgames_ball_points(
    sample_id: str = Query(...),
    max_points: int = Query(50000, ge=1, le=200000),
) -> dict[str, Any]:
    return extract_ball_points_for_sample(
        sample_id=sample_id,
        max_points=max_points,
    )


@app.get("/api/datasets/external-sample-index")
def api_datasets_external_sample_index(rebuild: bool = Query(False)) -> dict[str, Any]:
    return load_external_sample_index(rebuild=rebuild)


@app.get("/api/datasets/external-samples")
def api_datasets_external_samples(
    rebuild: bool = Query(False),
    dataset_id: str = Query("all"),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, Any]:
    return list_external_samples(rebuild=rebuild, dataset_id=dataset_id, limit=limit)


@app.get("/api/datasets/external-sample-detail")
def api_datasets_external_sample_detail(
    dataset_id: str = Query(...),
    sample_id: str = Query(...),
    rebuild: bool = Query(False),
) -> dict[str, Any]:
    return sample_detail(dataset_id=dataset_id, sample_id=sample_id, rebuild=rebuild)


@app.get("/api/datasets/external-registry")
def api_datasets_external_registry(rebuild: bool = Query(False)) -> dict[str, Any]:
    return load_external_dataset_registry(rebuild=rebuild)


@app.get("/api/datasets/external-summary")
def api_datasets_external_summary(rebuild: bool = Query(False)) -> dict[str, Any]:
    return external_dataset_summary(rebuild=rebuild)


@app.get("/api/datasets/external-videos")
def api_datasets_external_videos(
    rebuild: bool = Query(False),
    dataset_id: str = Query("all"),
    limit: int = Query(250, ge=1, le=1000),
) -> dict[str, Any]:
    return list_external_videos(
        rebuild=rebuild,
        dataset_id=dataset_id,
        limit=limit,
    )


@app.get("/api/datasets/unified-raw-clips")
def api_datasets_unified_raw_clips(
    mode: str = Query("legacy_transversal"),
    rebuild: bool = Query(False),
    quality: str = Query("all"),
    external_dataset_id: str = Query("all"),
    min_ball_points: int = Query(0, ge=0),
    limit: int = Query(250, ge=1, le=1000),
) -> dict[str, Any]:
    return list_unified_raw_clips(
        mode=mode,
        rebuild=rebuild,
        quality=quality,
        external_dataset_id=external_dataset_id,
        min_ball_points=min_ball_points,
        limit=limit,
    )


@app.get("/api/datasets/clip-index")
def api_datasets_clip_index(
    rebuild: bool = Query(False),
    quality: str = Query("all"),
    min_ball_points: int = Query(0, ge=0),
    limit: int = Query(250, ge=1, le=1000),
) -> dict[str, Any]:
    return list_clips(
        rebuild=rebuild,
        quality=quality,
        min_ball_points=min_ball_points,
        limit=limit,
    )


@app.get("/api/datasets/clip-index-summary")
def api_datasets_clip_index_summary(rebuild: bool = Query(False)) -> dict[str, Any]:
    payload = load_transversal_clip_index(rebuild=rebuild)
    return {
        "clip_count": payload.get("clip_count", 0),
        "scene_rows": payload.get("scene_rows", 0),
        "source_video_rows": payload.get("source_video_rows", 0),
        "quality_counts": payload.get("quality_counts", {}),
        "scene_status_counts": payload.get("scene_status_counts", {}),
        "source_counts": payload.get("source_counts", {}),
        "outputs": payload.get("outputs", {}),
    }


@app.get("/api/datasets/transversal-registry")
def api_datasets_transversal_registry(rebuild: bool = Query(False)) -> dict[str, Any]:
    return load_transversal_registry(rebuild=rebuild)


@app.get("/api/datasets/active-profile")
def api_datasets_active_profile(rebuild: bool = Query(False)) -> dict[str, Any]:
    return load_active_profile(rebuild=rebuild)


@app.get("/api/datasets/role")
def api_datasets_role(
    role: str = Query(...),
    rebuild: bool = Query(False),
) -> dict[str, Any]:
    return role_records(role=role, rebuild=rebuild)


@app.get("/api/datasets/audit")
def api_datasets_audit(rebuild: bool = Query(False)) -> dict[str, Any]:
    return load_dataset_audit(rebuild=rebuild)


@app.get("/api/datasets/pools")
def api_datasets_pools(rebuild: bool = Query(False)) -> dict[str, Any]:
    return dataset_pools(rebuild=rebuild)


@app.get("/api/datasets/analyzed-videos")
def api_datasets_analyzed_videos(
    mode: str = Query("canonical"),
    limit: int = Query(200, ge=1, le=1000),
    rebuild: bool = Query(False),
) -> dict[str, Any]:
    return list_transversal_analyzed_videos(mode=mode, limit=limit, rebuild=rebuild)


@app.get("/api/light/analyzed-videos")
def api_light_analyzed_videos(
    limit: int = Query(250, ge=1, le=1000),
    with_meta: bool = Query(False),
) -> dict[str, Any]:
    """
    Vidéos générées : overlays, previews, exports.
    """
    items = []
    seen = set()

    roots = [
        PROJECT_ROOT / "runs",
        LEGACY_ROOT / "runs",
    ]

    priority_terms = [
        "overlay",
        "display",
        "preview",
        "score",
        "table",
        "tracking",
        "topdown",
        "analysis",
        "annot",
    ]

    for root in roots:
        scanned = _scan_mp4s_under(root, limit=2000)
        for item in scanned:
            name_low = item["name"].lower()
            path_low = item["path"].lower()
            if not any(t in name_low or t in path_low for t in priority_terms):
                continue

            key = item["path"].lower()
            if key in seen:
                continue
            seen.add(key)
            item["source_root"] = str(root)
            items.append(item)
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    items = items[:limit]
    if with_meta:
        items = _attach_light_video_meta(items)
    return {
        "items": items,
        "count": len(items),
        "with_meta": with_meta,
    }


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return {
        "phase": "007A_ui_shell",
        "modules": {
            "interface": "ready_shell",
            "environment": "placeholder",
            "rules": "placeholder",
            "ball_tracking": "placeholder",
            "training": "placeholder",
            "dataset": "placeholder",
        },
        "next_patch": "007B_import_legacy_artifacts_index",
    }
