from __future__ import annotations

import argparse
import re
from pathlib import Path


CSS = r"""
/* === PATCH 001Y4 brush + coordinate fix === */
.tool-btn.brush.active {
  background: #1f3050;
}
.brush-extra {
  margin-top: 8px;
}
.brush-extra label {
  display: block;
  color: #9ea7b8;
  font-size: 12px;
  margin-top: 6px;
}
.brush-extra input[type="range"] {
  width: 100%;
  padding: 0;
  margin-top: 5px;
}
"""


NEW_TOOL_ROW = r"""
<div class="tool-row">
  <button class="tool-btn brush" id="toolBrushBtn" onclick="setToolMode('brush')">Pinceau</button>
  <button class="tool-btn point active" id="toolPointBtn" onclick="setToolMode('point')">Point / Ligne</button>
  <button class="tool-btn erase" id="toolEraseBtn" onclick="setToolMode('erase')">Gomme</button>
</div>

<div class="brush-extra">
  <label>
    Pas pinceau : <span id="brushFrameStepText">1</span> frame
    <input id="brushFrameStep" type="range" min="1" max="5" value="1" step="1" oninput="setBrushFrameStep001Y4(this.value)">
  </label>
  <label>
    Espacement pinceau : <span id="brushSpacingText">10</span> px
    <input id="brushSpacing" type="range" min="4" max="40" value="10" step="1" oninput="setBrushSpacing001Y4(this.value)">
  </label>
</div>
"""


OLD_CLICK_001Y3 = r"""canvas.addEventListener("click", (evt) => {
  if (toolMode001Y3 === "erase") {
    eraseNearCanvasEvent(evt);
    return;
  }

  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});"""


NEW_CLICK_001Y4 = r"""canvas.addEventListener("click", (evt) => {
  if (toolMode001Y3 === "erase") {
    eraseNearCanvasEvent(evt);
    return;
  }

  if (toolMode001Y3 === "brush") {
    return;
  }

  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});"""


JS = r"""
// === PATCH 001Y4 brush + coordinate fix ===
let brushFrameStep001Y4 = 1;
let brushSpacing001Y4 = 10;
let brushing001Y4 = false;
let brushStartFrame001Y4 = 0;
let brushIndex001Y4 = 0;
let brushLastCanvas001Y4 = null;

function getDisplayedVideoGeometry001Y4() {
  if (!currentSegment) {
    return { ox: 0, oy: 0, w: canvas.width, h: canvas.height, vw: canvas.width, vh: canvas.height };
  }

  const vw = Number(video.videoWidth || currentSegment.width || canvas.width || 1);
  const vh = Number(video.videoHeight || currentSegment.height || canvas.height || 1);

  const cw = Number(canvas.width || 1);
  const ch = Number(canvas.height || 1);

  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;

  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;

  return { ox, oy, w: dw, h: dh, vw, vh };
}

// Remplace la conversion naïve ancienne.
// Corrige les décalages causés par zoom navigateur, marges noires, object-fit, redimensionnement.
videoToCanvas = function(x, y) {
  const g = getDisplayedVideoGeometry001Y4();

  return {
    x: g.ox + (Number(x) / g.vw) * g.w,
    y: g.oy + (Number(y) / g.vh) * g.h,
  };
};

canvasToVideo = function(evt) {
  const rect = canvas.getBoundingClientRect();
  const cx = (evt.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
  const cy = (evt.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));

  const g = getDisplayedVideoGeometry001Y4();

  let x = ((cx - g.ox) / Math.max(1, g.w)) * g.vw;
  let y = ((cy - g.oy) / Math.max(1, g.h)) * g.vh;

  x = Math.max(0, Math.min(g.vw, x));
  y = Math.max(0, Math.min(g.vh, y));

  return { x, y };
};

resizeCanvasNoDraw = function() {
  const rect = video.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
};

resizeCanvas = function() {
  resizeCanvasNoDraw();
  draw();
};

function setToolMode(mode) {
  toolMode001Y3 = ["brush", "point", "erase"].includes(mode) ? mode : "point";

  const brushBtn = document.getElementById("toolBrushBtn");
  const pointBtn = document.getElementById("toolPointBtn");
  const eraseBtn = document.getElementById("toolEraseBtn");

  if (brushBtn) brushBtn.classList.toggle("active", toolMode001Y3 === "brush");
  if (pointBtn) pointBtn.classList.toggle("active", toolMode001Y3 === "point");
  if (eraseBtn) eraseBtn.classList.toggle("active", toolMode001Y3 === "erase");

  draw();
}

function setBrushFrameStep001Y4(value) {
  brushFrameStep001Y4 = Math.max(1, Math.min(5, Number(value || 1)));

  const txt = document.getElementById("brushFrameStepText");
  if (txt) txt.textContent = String(brushFrameStep001Y4);
}

function setBrushSpacing001Y4(value) {
  brushSpacing001Y4 = Math.max(4, Math.min(40, Number(value || 10)));

  const txt = document.getElementById("brushSpacingText");
  if (txt) txt.textContent = String(brushSpacing001Y4);
}

function addHumanPointAtFrame001Y4(frame, x, y, quality) {
  if (!currentSegment) return;

  const bStart = Number(currentSegment.first_frame || 0);
  const bEnd = Number(currentSegment.last_frame || frame);
  const clampedFrame = Math.max(bStart, Math.min(bEnd, Math.round(frame)));

  const { state, segState } = getSegState(currentSegment.review_id);

  segState.points[String(clampedFrame)] = {
    frame: clampedFrame,
    x: Math.round(Number(x) * 100) / 100,
    y: Math.round(Number(y) * 100) / 100,
    visible: quality === "visible" ? 1 : 0,
    quality,
    updated_at: new Date().toISOString(),
  };

  saveState(state);
}

function canvasPointFromEvent001Y4(evt) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (evt.clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
    y: (evt.clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
  };
}

function distance001Y4(a, b) {
  if (!a || !b) return Infinity;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function brushAddSample001Y4(evt, force = false) {
  if (!currentSegment) return;

  const canvasPoint = canvasPointFromEvent001Y4(evt);

  if (!force && distance001Y4(canvasPoint, brushLastCanvas001Y4) < brushSpacing001Y4) {
    return;
  }

  const frame = brushStartFrame001Y4 + brushIndex001Y4 * brushFrameStep001Y4;
  const end = Number(currentSegment.last_frame || frame);

  if (frame > end) {
    brushing001Y4 = false;
    return;
  }

  const videoPoint = canvasToVideo(evt);

  addHumanPointAtFrame001Y4(frame, videoPoint.x, videoPoint.y, "visible");

  brushLastCanvas001Y4 = canvasPoint;
  brushIndex001Y4 += 1;

  refreshInfo();
  refreshPointList();
  draw();

  if (typeof updateTimeline === "function") {
    updateTimeline();
  }
}

function startBrush001Y4(evt) {
  if (toolMode001Y3 !== "brush") return;

  brushing001Y4 = true;
  brushStartFrame001Y4 = currentFrame();
  brushIndex001Y4 = 0;
  brushLastCanvas001Y4 = null;

  brushAddSample001Y4(evt, true);
}

function moveBrush001Y4(evt) {
  if (toolMode001Y3 !== "brush") return;
  if (!brushing001Y4) return;

  brushAddSample001Y4(evt, false);
}

function stopBrush001Y4() {
  brushing001Y4 = false;
}

canvas.addEventListener("mousedown", evt => {
  if (toolMode001Y3 === "brush") {
    evt.preventDefault();
    startBrush001Y4(evt);
  }
});

canvas.addEventListener("mousemove", evt => {
  if (toolMode001Y3 === "brush" && brushing001Y4) {
    evt.preventDefault();
    moveBrush001Y4(evt);
  }
});

window.addEventListener("mouseup", () => {
  stopBrush001Y4();
});

function toggleToolMode001Y3() {
  if (toolMode001Y3 === "erase") {
    setToolMode("point");
  } else {
    setToolMode("erase");
  }
}

document.addEventListener("keydown", ev => {
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;

  const k = ev.key.toLowerCase();

  if (k === "b") {
    ev.preventDefault();
    setToolMode("brush");
  } else if (k === "p") {
    ev.preventDefault();
    setToolMode("point");
  } else if (k === "g") {
    ev.preventDefault();
    setToolMode("erase");
  }
});

setTimeout(() => {
  setToolMode("point");
  setBrushFrameStep001Y4(1);
  setBrushSpacing001Y4(10);
  resizeCanvas();
}, 300);
"""


def replace_tool_row(text: str) -> str:
    if "toolBrushBtn" in text:
        return text

    pattern = re.compile(
        r'<div class="tool-row">\s*'
        r'<button class="tool-btn point active" id="toolPointBtn".*?</button>\s*'
        r'<button class="tool-btn erase" id="toolEraseBtn".*?</button>\s*'
        r'</div>',
        re.S,
    )

    new_text, n = pattern.subn(NEW_TOOL_ROW, text, count=1)

    if n == 0:
        raise RuntimeError("Could not replace tool-row. Check 001Y3 HTML structure.")

    return new_text


def patch_click_handler(text: str) -> str:
    if OLD_CLICK_001Y3 in text:
        return text.replace(OLD_CLICK_001Y3, NEW_CLICK_001Y4, 1)

    if NEW_CLICK_001Y4 in text:
        return text

    raise RuntimeError("Could not find canvas click handler from 001Y3.")


def patch_html(in_html: Path, out_html: Path) -> None:
    text = in_html.read_text(encoding="utf-8")

    if "/* === PATCH 001Y4 brush + coordinate fix === */" not in text:
        if "</style>" not in text:
            raise RuntimeError("Cannot find </style>.")
        text = text.replace("</style>", CSS + "\n</style>", 1)

    text = replace_tool_row(text)
    text = patch_click_handler(text)

    if "// === PATCH 001Y4 brush + coordinate fix ===" not in text:
        if "</body>" not in text:
            raise RuntimeError("Cannot find </body>.")
        text = text.replace("</body>", f"<script>\n{JS}\n</script>\n</body>", 1)

    out_html.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/ball_annotation_001Y3_eraser.html")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y4_brush_fixed.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    patch_html(in_html, out_html)

    print(f"[001Y4] input : {in_html}")
    print(f"[001Y4] output: {out_html}")


if __name__ == "__main__":
    main()
