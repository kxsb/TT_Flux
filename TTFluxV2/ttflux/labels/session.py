from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from ttflux.core.paths import data_dir
from ttflux.video.reader import read_video_meta


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def labels_root() -> Path:
    root = data_dir() / "labels" / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_session_id(session_id: str) -> str:
    session_id = str(session_id or "").strip()
    if not session_id or not SAFE_ID_RE.match(session_id):
        raise ValueError("invalid_session_id")
    return session_id


def session_dir(session_id: str) -> Path:
    session_id = validate_session_id(session_id)
    return labels_root() / session_id


def session_json_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def annotations_jsonl_path(session_id: str) -> Path:
    return session_dir(session_id) / "annotations.jsonl"


def make_session_id(video_path: str, name: str | None = None) -> str:
    stem = Path(video_path).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:48].strip("_") or "video"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(str(video_path).encode("utf-8", errors="ignore")).hexdigest()[:8]
    if name:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:32].strip("_")
        if clean:
            stem = f"{clean}_{stem}"
    return f"{stamp}_{stem}_{digest}"


def create_label_session(video_path: str, name: str | None = None) -> dict[str, Any]:
    session_id = make_session_id(video_path, name)
    out_dir = session_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_video_meta(Path(video_path))

    payload = {
        "session_id": session_id,
        "name": name or "",
        "video_path": str(video_path),
        "created_at": time.time(),
        "updated_at": time.time(),
        "annotation_count": 0,
        "video_meta": meta,
        "schema": {
            "format": "jsonl",
            "fields": [
                "uid",
                "created_at",
                "frame",
                "label_type",
                "x",
                "y",
                "visible",
                "object_id",
                "notes",
                "payload",
            ],
        },
    }

    session_json_path(session_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    annotations_jsonl_path(session_id).touch(exist_ok=True)

    return payload


def read_session(session_id: str) -> dict[str, Any]:
    p = session_json_path(session_id)
    if not p.exists():
        raise FileNotFoundError(f"session_not_found: {session_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_sessions() -> list[dict[str, Any]]:
    out = []
    for p in labels_root().glob("*/session.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            out.append(payload)
        except Exception:
            continue
    out.sort(key=lambda x: float(x.get("updated_at") or x.get("created_at") or 0), reverse=True)
    return out


def read_annotations(session_id: str) -> list[dict[str, Any]]:
    p = annotations_jsonl_path(session_id)
    if not p.exists():
        return []

    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def append_annotation(session_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
    session_id = validate_session_id(session_id)
    out_dir = session_dir(session_id)
    if not out_dir.exists():
        raise FileNotFoundError(f"session_not_found: {session_id}")

    ann = {
        "uid": annotation.get("uid") or uuid.uuid4().hex,
        "created_at": time.time(),
        "frame": int(annotation.get("frame", 0)),
        "label_type": str(annotation.get("label_type") or "point"),
        "x": annotation.get("x"),
        "y": annotation.get("y"),
        "visible": bool(annotation.get("visible", True)),
        "object_id": str(annotation.get("object_id") or ""),
        "notes": str(annotation.get("notes") or ""),
        "payload": annotation.get("payload") or {},
    }

    with annotations_jsonl_path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    meta = read_session(session_id)
    meta["updated_at"] = time.time()
    meta["annotation_count"] = len(read_annotations(session_id))
    session_json_path(session_id).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ann


def delete_annotation(session_id: str, uid: str) -> dict[str, Any]:
    anns = read_annotations(session_id)
    kept = [a for a in anns if a.get("uid") != uid]
    removed = len(anns) - len(kept)

    p = annotations_jsonl_path(session_id)
    with p.open("w", encoding="utf-8") as f:
        for ann in kept:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    meta = read_session(session_id)
    meta["updated_at"] = time.time()
    meta["annotation_count"] = len(kept)
    session_json_path(session_id).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"removed": removed, "annotation_count": len(kept)}


def export_annotations_csv(session_id: str) -> str:
    anns = read_annotations(session_id)
    fields = [
        "uid",
        "created_at",
        "frame",
        "label_type",
        "x",
        "y",
        "visible",
        "object_id",
        "notes",
        "payload_json",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()

    for ann in anns:
        row = {
            "uid": ann.get("uid", ""),
            "created_at": ann.get("created_at", ""),
            "frame": ann.get("frame", ""),
            "label_type": ann.get("label_type", ""),
            "x": ann.get("x", ""),
            "y": ann.get("y", ""),
            "visible": ann.get("visible", ""),
            "object_id": ann.get("object_id", ""),
            "notes": ann.get("notes", ""),
            "payload_json": json.dumps(ann.get("payload") or {}, ensure_ascii=False),
        }
        writer.writerow(row)

    return buf.getvalue()
