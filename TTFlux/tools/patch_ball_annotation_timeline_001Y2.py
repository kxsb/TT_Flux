from __future__ import annotations

import argparse
from pathlib import Path


CSS = r"""
/* === PATCH 001Y2 timeline === */
body {
  padding-bottom: 142px;
}
.timeline-panel {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 50;
  background: #181b22;
  border-top: 1px solid #2b303b;
  padding: 10px 14px 12px 14px;
  box-shadow: 0 -8px 22px rgba(0,0,0,.35);
}
.timeline-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 6px;
}
.timeline-title {
  font-weight: 800;
}
.timeline-frame {
  color: #9dffbc;
  font-family: Consolas, monospace;
  font-weight: 800;
}
.timeline-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}
.timeline-controls button {
  background: #101218;
  color: #edf0f7;
  border: 1px solid #3a4150;
  border-radius: 8px;
  padding: 6px 9px;
  cursor: pointer;
  font-weight: 800;
}
.timeline-controls button:hover {
  filter: brightness(1.2);
}
#timelineCanvas {
  width: 100%;
  height: 46px;
  display: block;
  background: #0b0d12;
  border: 1px solid #3a4150;
  border-radius: 10px;
  cursor: pointer;
}
#timelineSlider {
  width: 100%;
  margin-top: 6px;
}
.timeline-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 4px;
  color: #9ea7b8;
  font-size: 12px;
}
.timeline-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  margin-right: 4px;
  vertical-align: -1px;
}
.timeline-dot.current { background: #ffffff; }
.timeline-dot.human { background: #50ff8c; }
.timeline-dot.auto { background: #ffd800; }
.timeline-dot.invisible { background: #9bc2ff; }
"""


HTML = r"""
<!-- === PATCH 001Y2 timeline === -->
<div class="timeline-panel" id="timelinePanel">
  <div class="timeline-top">
    <div class="timeline-title">Timeline segment</div>
    <div class="timeline-frame" id="timelineFrameText">f?</div>
  </div>

  <div class="timeline-controls">
    <button onclick="jumpStart()">Début</button>
    <button onclick="stepFrame(-25)">-25</button>
    <button onclick="stepFrame(-10)">-10</button>
    <button onclick="stepFrame(-1)">-1</button>
    <button onclick="togglePlay()">Play/Pause</button>
    <button onclick="stepFrame(1)">+1</button>
    <button onclick="stepFrame(10)">+10</button>
    <button onclick="stepFrame(25)">+25</button>
    <button onclick="jumpEnd()">Fin</button>
  </div>

  <canvas id="timelineCanvas"></canvas>
  <input id="timelineSlider" type="range" min="0" max="1" value="0" step="1">

  <div class="timeline-legend">
    <span><span class="timeline-dot current"></span>frame courante</span>
    <span><span class="timeline-dot human"></span>point humain visible</span>
    <span><span class="timeline-dot invisible"></span>invisible/incertain</span>
    <span><span class="timeline-dot auto"></span>points auto</span>
  </div>
</div>
"""


JS = r"""
// === PATCH 001Y2 timeline ===
function timelineBounds() {
  if (!currentSegment) return { start: 0, end: 1 };

  const start = Number(currentSegment.first_frame || 0);
  const end = Number(currentSegment.last_frame || Math.max(start + 1, start));

  return {
    start: Math.min(start, end),
    end: Math.max(start + 1, end)
  };
}

function frameToTimelineX(frame, width) {
  const b = timelineBounds();
  return ((frame - b.start) / (b.end - b.start)) * width;
}

function timelineXToFrame(x, width) {
  const b = timelineBounds();
  const ratio = Math.max(0, Math.min(1, x / Math.max(1, width)));
  return Math.round(b.start + ratio * (b.end - b.start));
}

function ensureTimelineCanvasSize() {
  const canvasEl = document.getElementById("timelineCanvas");
  if (!canvasEl) return null;

  const rect = canvasEl.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (canvasEl.width !== w || canvasEl.height !== h) {
    canvasEl.width = w;
    canvasEl.height = h;
  }

  return canvasEl;
}

function updateTimeline() {
  if (!currentSegment) return;

  const slider = document.getElementById("timelineSlider");
  const text = document.getElementById("timelineFrameText");
  const canvasEl = ensureTimelineCanvasSize();

  if (!slider || !text || !canvasEl) return;

  const b = timelineBounds();
  const f = currentFrame();

  slider.min = String(b.start);
  slider.max = String(b.end);
  slider.step = "1";
  slider.value = String(Math.max(b.start, Math.min(f, b.end)));

  text.textContent = `f${f} · ${b.start} → ${b.end}`;

  const tctx = canvasEl.getContext("2d");
  const w = canvasEl.width;
  const h = canvasEl.height;

  tctx.clearRect(0, 0, w, h);

  // fond
  tctx.fillStyle = "#0b0d12";
  tctx.fillRect(0, 0, w, h);

  // ligne centrale
  const y = Math.round(h * 0.50);
  tctx.strokeStyle = "#3a4150";
  tctx.lineWidth = 4;
  tctx.beginPath();
  tctx.moveTo(8, y);
  tctx.lineTo(w - 8, y);
  tctx.stroke();

  // points automatiques
  if (currentSegment.track_points) {
    tctx.fillStyle = "rgba(255,216,0,.78)";
    currentSegment.track_points.forEach(p => {
      const pf = Number(p.frame);
      if (pf < b.start || pf > b.end) return;
      const x = frameToTimelineX(pf, w);
      tctx.beginPath();
      tctx.arc(x, y, 2.7, 0, Math.PI * 2);
      tctx.fill();
    });
  }

  // points humains
  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };
  const pts = Object.values(segState.points || {});

  pts.forEach(p => {
    const pf = Number(p.frame);
    if (pf < b.start || pf > b.end) return;

    const x = frameToTimelineX(pf, w);

    if (p.quality === "visible") {
      tctx.fillStyle = "rgba(80,255,140,.95)";
    } else {
      tctx.fillStyle = "rgba(155,194,255,.95)";
    }

    tctx.beginPath();
    tctx.arc(x, y - 11, 5, 0, Math.PI * 2);
    tctx.fill();
  });

  // frame courante
  const cx = frameToTimelineX(f, w);
  tctx.strokeStyle = "#ffffff";
  tctx.lineWidth = 2;
  tctx.beginPath();
  tctx.moveTo(cx, 4);
  tctx.lineTo(cx, h - 4);
  tctx.stroke();

  tctx.fillStyle = "#ffffff";
  tctx.beginPath();
  tctx.arc(cx, y, 5, 0, Math.PI * 2);
  tctx.fill();
}

function installTimelineHandlers() {
  const slider = document.getElementById("timelineSlider");
  const canvasEl = document.getElementById("timelineCanvas");

  if (slider) {
    slider.addEventListener("input", () => {
      setFrame(Number(slider.value));
      updateTimeline();
    });
  }

  if (canvasEl) {
    canvasEl.addEventListener("click", evt => {
      const rect = canvasEl.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const frame = timelineXToFrame(x, rect.width);
      setFrame(frame);
      updateTimeline();
    });

    canvasEl.addEventListener("mousemove", evt => {
      const rect = canvasEl.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const frame = timelineXToFrame(x, rect.width);
      canvasEl.title = `aller à f${frame}`;
    });
  }
}

// On enveloppe quelques fonctions existantes pour rafraîchir la timeline.
(function patchTimelineRefresh() {
  const names = [
    "loadSegment",
    "setFrame",
    "stepFrame",
    "jumpStart",
    "jumpEnd",
    "addHumanPoint",
    "markInvisible",
    "markUncertain",
    "deleteCurrentFrame",
    "copyPreviousPoint",
    "clearCurrent",
    "draw",
    "refreshInfo"
  ];

  names.forEach(name => {
    const original = window[name];
    if (typeof original !== "function") return;

    window[name] = function(...args) {
      const result = original.apply(this, args);
      setTimeout(updateTimeline, 40);
      return result;
    };
  });
})();

window.addEventListener("resize", () => {
  setTimeout(updateTimeline, 40);
});

document.addEventListener("DOMContentLoaded", () => {
  installTimelineHandlers();
  setInterval(updateTimeline, 250);
  setTimeout(updateTimeline, 300);
});

// Si le script est injecté après chargement, on installe quand même.
setTimeout(() => {
  installTimelineHandlers();
  updateTimeline();
}, 300);
"""


def remove_block(text: str, start: str, end: str) -> str:
    while start in text and end in text:
        a = text.index(start)
        b = text.index(end, a) + len(end)
        text = text[:a] + text[b:]
    return text


def patch_html(in_html: Path, out_html: Path) -> None:
    text = in_html.read_text(encoding="utf-8")

    # Idempotence simple.
    text = remove_block(
        text,
        "/* === PATCH 001Y2 timeline === */",
        ".timeline-dot.invisible { background: #9bc2ff; }",
    )
    text = remove_block(
        text,
        "<!-- === PATCH 001Y2 timeline === -->",
        "</div>\n\n<script>\n// === PATCH 001Y2 timeline ===",
    )

    # Réinjection propre.
    if "</style>" not in text:
        raise RuntimeError("Cannot find </style> in HTML.")
    text = text.replace("</style>", CSS + "\n</style>", 1)

    if "</main>" not in text:
        raise RuntimeError("Cannot find </main> in HTML.")
    text = text.replace("</main>", "</main>\n" + HTML, 1)

    if "</body>" not in text:
        raise RuntimeError("Cannot find </body> in HTML.")
    text = text.replace("</body>", f"<script>\n{JS}\n</script>\n</body>", 1)

    out_html.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/ball_annotation_001Y.html")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y2_timeline.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    patch_html(in_html, out_html)

    print(f"[001Y2] input : {in_html}")
    print(f"[001Y2] output: {out_html}")


if __name__ == "__main__":
    main()
