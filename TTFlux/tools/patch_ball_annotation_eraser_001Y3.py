from __future__ import annotations

import argparse
from pathlib import Path


CSS = r"""
/* === PATCH 001Y3 eraser === */
.eraser-panel {
  position: fixed;
  right: 14px;
  bottom: 148px;
  z-index: 60;
  background: #181b22;
  border: 1px solid #2b303b;
  border-radius: 14px;
  padding: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,.35);
  width: 250px;
}
.eraser-panel h3 {
  margin: 0 0 8px 0;
  font-size: 15px;
}
.tool-row {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
}
.tool-btn {
  flex: 1;
  background: #242936;
  color: #edf0f7;
  border: 1px solid #3a4150;
  border-radius: 10px;
  padding: 8px 9px;
  cursor: pointer;
  font-weight: 800;
}
.tool-btn.active {
  outline: 2px solid #ffffff;
}
.tool-btn.point.active {
  background: #16331f;
}
.tool-btn.erase.active {
  background: #3a1518;
}
.eraser-panel label {
  display: block;
  color: #9ea7b8;
  font-size: 12px;
}
.eraser-panel input[type="range"] {
  width: 100%;
  padding: 0;
  margin-top: 6px;
}
.eraser-help {
  margin-top: 8px;
  color: #9ea7b8;
  font-size: 12px;
  line-height: 1.35;
}
"""


HTML = r"""
<!-- === PATCH 001Y3 eraser === -->
<div class="eraser-panel" id="eraserPanel">
  <h3>Outil correction</h3>

  <div class="tool-row">
    <button class="tool-btn point active" id="toolPointBtn" onclick="setToolMode('point')">Point</button>
    <button class="tool-btn erase" id="toolEraseBtn" onclick="setToolMode('erase')">Gomme</button>
  </div>

  <label>
    Rayon gomme : <span id="eraserRadiusText">18</span> px
    <input id="eraserRadius" type="range" min="6" max="50" value="18" step="1" oninput="setEraserRadius(this.value)">
  </label>

  <div class="eraser-help">
    <b>Point</b> : clic = poser/corriger la balle.<br>
    <b>Gomme</b> : clic ou glisser = supprimer les points humains proches.<br>
    Raccourci : <b>G</b> bascule point/gomme.
  </div>
</div>
"""


JS = r"""
// === PATCH 001Y3 eraser ===
let toolMode001Y3 = "point";
let eraserRadius001Y3 = 18;
let erasing001Y3 = false;

function setToolMode(mode) {
  toolMode001Y3 = mode === "erase" ? "erase" : "point";

  const pointBtn = document.getElementById("toolPointBtn");
  const eraseBtn = document.getElementById("toolEraseBtn");

  if (pointBtn) pointBtn.classList.toggle("active", toolMode001Y3 === "point");
  if (eraseBtn) eraseBtn.classList.toggle("active", toolMode001Y3 === "erase");

  draw();
}

function setEraserRadius(value) {
  eraserRadius001Y3 = Number(value || 18);

  const txt = document.getElementById("eraserRadiusText");
  if (txt) txt.textContent = String(eraserRadius001Y3);

  draw();
}

function toggleToolMode001Y3() {
  setToolMode(toolMode001Y3 === "point" ? "erase" : "point");
}

function nearestHumanPointToCanvas(evt) {
  if (!currentSegment) return null;

  const rect = canvas.getBoundingClientRect();
  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;

  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };

  let best = null;

  Object.values(segState.points || {}).forEach(p => {
    if (p.quality !== "visible") return;
    if (p.x === "" || p.y === "") return;

    const q = videoToCanvas(Number(p.x), Number(p.y));
    const dx = q.x - mx;
    const dy = q.y - my;
    const d = Math.sqrt(dx * dx + dy * dy);

    if (!best || d < best.distance) {
      best = {
        frame: String(p.frame),
        point: p,
        distance: d,
      };
    }
  });

  return best;
}

function eraseNearCanvasEvent(evt) {
  if (!currentSegment) return false;

  const nearest = nearestHumanPointToCanvas(evt);
  if (!nearest) return false;

  if (nearest.distance > eraserRadius001Y3) {
    return false;
  }

  const { state, segState } = getSegState(currentSegment.review_id);
  delete segState.points[String(nearest.frame)];

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();

  return true;
}

function drawEraserPreview001Y3() {
  if (toolMode001Y3 !== "erase") return;
  if (!mousePos || !currentSegment) return;

  const q = videoToCanvas(mousePos.x, mousePos.y);

  ctx.strokeStyle = "rgba(255,120,120,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, eraserRadius001Y3, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,120,120,0.10)";
  ctx.beginPath();
  ctx.arc(q.x, q.y, eraserRadius001Y3, 0, Math.PI * 2);
  ctx.fill();
}

// Patch draw pour afficher le cercle de gomme.
(function patchDrawForEraser001Y3() {
  const originalDraw = window.draw;
  if (typeof originalDraw !== "function") return;

  window.draw = function(...args) {
    const result = originalDraw.apply(this, args);
    drawEraserPreview001Y3();
    return result;
  };
})();

// Glisser en mode gomme.
canvas.addEventListener("mousedown", evt => {
  if (toolMode001Y3 !== "erase") return;
  erasing001Y3 = true;
  eraseNearCanvasEvent(evt);
});

canvas.addEventListener("mousemove", evt => {
  if (toolMode001Y3 === "erase" && erasing001Y3) {
    eraseNearCanvasEvent(evt);
  }
});

window.addEventListener("mouseup", () => {
  erasing001Y3 = false;
});

// Raccourci G.
document.addEventListener("keydown", ev => {
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;

  if (ev.key.toLowerCase() === "g") {
    ev.preventDefault();
    toggleToolMode001Y3();
  }
});

setTimeout(() => {
  setToolMode("point");
  setEraserRadius(18);
}, 200);
"""


OLD_CLICK = r"""canvas.addEventListener("click", (evt) => {
  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});"""


NEW_CLICK = r"""canvas.addEventListener("click", (evt) => {
  if (toolMode001Y3 === "erase") {
    eraseNearCanvasEvent(evt);
    return;
  }

  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});"""


def patch_html(in_html: Path, out_html: Path) -> None:
    text = in_html.read_text(encoding="utf-8")

    if "/* === PATCH 001Y3 eraser === */" not in text:
        if "</style>" not in text:
            raise RuntimeError("Cannot find </style>.")
        text = text.replace("</style>", CSS + "\n</style>", 1)

    if "<!-- === PATCH 001Y3 eraser === -->" not in text:
        if "</main>" not in text:
            raise RuntimeError("Cannot find </main>.")
        text = text.replace("</main>", "</main>\n" + HTML, 1)

    if OLD_CLICK in text:
        text = text.replace(OLD_CLICK, NEW_CLICK, 1)
    elif NEW_CLICK not in text:
        raise RuntimeError("Cannot find original canvas click handler to patch.")

    if "// === PATCH 001Y3 eraser ===" not in text:
        if "</body>" not in text:
            raise RuntimeError("Cannot find </body>.")
        text = text.replace("</body>", f"<script>\n{JS}\n</script>\n</body>", 1)

    out_html.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/ball_annotation_001Y2_timeline.html")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y3_eraser.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    patch_html(in_html, out_html)

    print(f"[001Y3] input : {in_html}")
    print(f"[001Y3] output: {out_html}")


if __name__ == "__main__":
    main()
