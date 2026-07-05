from __future__ import annotations

import argparse
from pathlib import Path


CSS = r"""
/* === PATCH 001Y6 observed/inferred quality === */
.quality-panel-001Y6 {
  margin-top: 10px;
  border-top: 1px solid #2b303b;
  padding-top: 9px;
}
.quality-panel-001Y6 h4 {
  margin: 0 0 7px 0;
  color: #edf0f7;
  font-size: 13px;
}
.quality-row-001Y6 {
  display: flex;
  gap: 6px;
}
.quality-btn-001Y6 {
  flex: 1;
  background: #242936;
  color: #edf0f7;
  border: 1px solid #3a4150;
  border-radius: 10px;
  padding: 8px 8px;
  cursor: pointer;
  font-weight: 900;
  font-size: 12px;
}
.quality-btn-001Y6.active {
  outline: 2px solid #ffffff;
}
.quality-btn-001Y6.visible.active {
  background: #16331f;
}
.quality-btn-001Y6.inferred.active {
  background: #2a2048;
}
.quality-help-001Y6 {
  margin-top: 7px;
  color: #9ea7b8;
  font-size: 12px;
  line-height: 1.35;
}
"""


HTML = r"""
<!-- === PATCH 001Y6 observed/inferred quality === -->
<div class="quality-panel-001Y6">
  <h4>Type de point posé</h4>
  <div class="quality-row-001Y6">
    <button
      class="quality-btn-001Y6 visible active"
      id="qualityVisibleBtn001Y6"
      onclick="setPointQuality001Y6('visible')">
      V · Observé
    </button>
    <button
      class="quality-btn-001Y6 inferred"
      id="qualityInferredBtn001Y6"
      onclick="setPointQuality001Y6('occluded_inferred')">
      I · Inféré / caché
    </button>
  </div>
  <div class="quality-help-001Y6">
    <b>Observé</b> : je vois la balle.<br>
    <b>Inféré</b> : balle cachée, mais position probable.
  </div>
</div>
"""


JS = r"""
// === PATCH 001Y6 observed/inferred quality ===
let pointQuality001Y6 = "visible";

function currentPointQuality001Y6() {
  return pointQuality001Y6 || "visible";
}

function setPointQuality001Y6(q) {
  pointQuality001Y6 = q === "occluded_inferred" ? "occluded_inferred" : "visible";

  const visibleBtn = document.getElementById("qualityVisibleBtn001Y6");
  const inferredBtn = document.getElementById("qualityInferredBtn001Y6");

  if (visibleBtn) visibleBtn.classList.toggle("active", pointQuality001Y6 === "visible");
  if (inferredBtn) inferredBtn.classList.toggle("active", pointQuality001Y6 === "occluded_inferred");

  draw();
}

function isDrawableHumanPoint001Y6(p) {
  if (!p) return false;
  if (p.x === "" || p.y === "") return false;
  return p.quality === "visible" || p.quality === "occluded_inferred" || p.quality === "inferred";
}

function qualityLabel001Y6(q) {
  if (q === "visible") return "observé";
  if (q === "occluded_inferred" || q === "inferred") return "inféré/caché";
  return q || "";
}

function qualityColor001Y6(q, alpha = 1) {
  if (q === "occluded_inferred" || q === "inferred") {
    return `rgba(190,120,255,${alpha})`;
  }
  return `rgba(80,255,140,${alpha})`;
}

function drawHumanPoint001Y6(p, isCurrent) {
  const q = videoToCanvas(Number(p.x), Number(p.y));
  const color = qualityColor001Y6(p.quality, 0.95);

  if (p.quality === "occluded_inferred" || p.quality === "inferred") {
    ctx.setLineDash([4, 4]);
    drawCircle(q.x, q.y, color, isCurrent ? 9 : 6);
    ctx.setLineDash([]);
  } else {
    drawCircle(q.x, q.y, color, isCurrent ? 8 : 5);
  }

  if (isCurrent) {
    drawCross(q.x, q.y, color, 13);
  }
}

// Remplace l'affichage humain : vert = observé, violet pointillé = inféré.
drawHuman = function() {
  if (!document.getElementById("showHuman").checked) return;
  if (!currentSegment) return;

  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };

  const pts = Object.values(segState.points || {})
    .filter(isDrawableHumanPoint001Y6)
    .sort((a, b) => a.frame - b.frame);

  // segments entre points humains
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];

    const qa = videoToCanvas(Number(a.x), Number(a.y));
    const qb = videoToCanvas(Number(b.x), Number(b.y));

    const inferred = (
      a.quality === "occluded_inferred" ||
      b.quality === "occluded_inferred" ||
      a.quality === "inferred" ||
      b.quality === "inferred"
    );

    ctx.strokeStyle = inferred ? "rgba(190,120,255,0.82)" : "rgba(80,255,140,0.95)";
    ctx.lineWidth = 2;

    if (inferred) {
      ctx.setLineDash([7, 5]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(qa.x, qa.y);
    ctx.lineTo(qb.x, qb.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const frame = currentFrame();

  pts.forEach(p => {
    drawHumanPoint001Y6(p, Number(p.frame) === frame);
  });
};

// Remplace la liste des points humains pour afficher les points inférés avec coordonnées.
refreshPointList = function() {
  if (!currentSegment) return;

  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };

  const pts = Object.values(segState.points || {})
    .sort((a, b) => a.frame - b.frame);

  if (!pts.length) {
    document.getElementById("pointList").textContent = "aucun point humain";
    return;
  }

  document.getElementById("pointList").innerHTML = pts.map(p => {
    const label = qualityLabel001Y6(p.quality);

    if (isDrawableHumanPoint001Y6(p)) {
      return `f${p.frame} · x=${p.x} y=${p.y} · ${label}`;
    }

    return `f${p.frame} · ${label}`;
  }).join("<br>");
};

// Souris : bleu clair en mode observé, violet en mode inféré.
drawMouse = function() {
  if (!mousePos || !currentSegment) return;

  const q = videoToCanvas(mousePos.x, mousePos.y);
  const color = currentPointQuality001Y6() === "occluded_inferred"
    ? "rgba(190,120,255,0.90)"
    : "rgba(120,180,255,0.85)";

  drawCross(q.x, q.y, color, 8);

  if (currentPointQuality001Y6() === "occluded_inferred") {
    ctx.setLineDash([4, 4]);
    drawCircle(q.x, q.y, color, 12);
    ctx.setLineDash([]);
  }
};

// Raccourcis V/I.
document.addEventListener("keydown", ev => {
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;

  const k = ev.key.toLowerCase();

  if (k === "v") {
    ev.preventDefault();
    setPointQuality001Y6("visible");
  } else if (k === "i") {
    ev.preventDefault();
    setPointQuality001Y6("occluded_inferred");
  }
}, true);

setTimeout(() => {
  setPointQuality001Y6("visible");
  refreshInfo();
  refreshPointList();
  draw();
}, 300);
"""


def patch_html(in_html: Path, out_html: Path) -> None:
    text = in_html.read_text(encoding="utf-8")

    if "/* === PATCH 001Y6 observed/inferred quality === */" not in text:
        if "</style>" not in text:
            raise RuntimeError("Cannot find </style>.")
        text = text.replace("</style>", CSS + "\n</style>", 1)

    if "<!-- === PATCH 001Y6 observed/inferred quality === -->" not in text:
        marker = '<label>\n    Rayon gomme'
        if marker not in text:
            raise RuntimeError("Cannot find eraser radius label.")
        text = text.replace(marker, HTML + "\n\n  " + marker, 1)

    # Point / Ligne : clic utilise le type de point courant.
    text = text.replace(
        'addHumanPoint(p.x, p.y, "visible");',
        'addHumanPoint(p.x, p.y, currentPointQuality001Y6());',
    )

    # Pinceau : chaque échantillon utilise le type courant.
    text = text.replace(
        'addHumanPointAtFrame001Y4(frame, videoPoint.x, videoPoint.y, "visible");',
        'addHumanPointAtFrame001Y4(frame, videoPoint.x, videoPoint.y, currentPointQuality001Y6());',
    )

    if "// === PATCH 001Y6 observed/inferred quality ===" not in text:
        if "</body>" not in text:
            raise RuntimeError("Cannot find </body>.")
        text = text.replace("</body>", f"<script>\n{JS}\n</script>\n</body>", 1)

    out_html.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/ball_annotation_001Y5_undo.html")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y6_inferred.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    patch_html(in_html, out_html)

    print(f"[001Y6] input : {in_html}")
    print(f"[001Y6] output: {out_html}")


if __name__ == "__main__":
    main()
