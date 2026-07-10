from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ttflux import __version__
from ttflux.video.catalog import find_video, scan_videos


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(
    title="TTFlux",
    version=__version__,
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    videos = scan_videos()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "videos": videos,
            "video_count": len(videos),
            "version": __version__,
        },
    )


@app.get("/api/videos")
def api_videos():
    videos = scan_videos()

    return {
        "video_count": len(videos),
        "videos": videos,
    }


@app.get("/media/{video_id}", name="media")
def media(video_id: str):
    video = find_video(video_id)

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="VidÃ©o introuvable",
        )

    path = Path(video["absolute_path"])

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Fichier vidÃ©o absent",
        )

    media_type = (
        mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )

    return FileResponse(
        path=str(path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
    }
