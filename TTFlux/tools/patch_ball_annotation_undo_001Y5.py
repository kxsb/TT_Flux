from __future__ import annotations

import argparse
from pathlib import Path


CSS = r"""
/* === PATCH 001Y5 undo === */
.undo-row-001Y5 {
  margin-top: 10px;
  border-top: 1px solid #2b303b;
  padding-top: 9px;
}
.undo-btn-001Y5 {
  width: 100%;
  background: #2b3140;
  color: #edf0f7;
  border: 1px solid #4b5365;
  border-radius: 10px;
  padding: 8px 10px;
  cursor: pointer;
  font-weight: 900;
}
.undo-btn-001Y5:hover {
  filter: brightness(1.18);
}
.undo-count-001Y5 {
  display: block;
  margin-top: 6px;
  color: #9ea7b8;
  font-size: 12px;
  text-align: center;
}
"""


HTML = r"""
<!-- === PATCH 001Y5 undo === -->
<div class="undo-row-001Y5">
  <button class="undo-btn-001Y5" onclick="undoLastAction001Y5()">Ctrl+Z · Annuler dernière action</button>
  <span class="undo-count-001Y5" id="undoCount001Y5">historique : 0</span>
</div>
"""


JS = r"""
// === PATCH 001Y5 undo ===
const UNDO_STACK_KEY_001Y5 = "ttflux_001Y5_undo_stack_v1";
let undoSuppress001Y5 = false;
let undoBrushOrEraseActive001Y5 = false;

function annotationStorageKey001Y5() {
  try {
    if (typeof STORAGE_KEY !== "undefined") return STORAGE_KEY;
  } catch (e) {}
  return "ttflux_001Y_ball_trace_v1";
}

function getUndoStack001Y5() {
  try {
    return JSON.parse(localStorage.getItem(UNDO_STACK_KEY_001Y5) || "[]");
  } catch (e) {
    return [];
  }
}

function saveUndoStack001Y5(stack) {
  const trimmed = stack.slice(-80);
  localStorage.setItem(UNDO_STACK_KEY_001Y5, JSON.stringify(trimmed));
  refreshUndoCount001Y5();
}

function currentAnnotationSnapshot001Y5() {
  return localStorage.getItem(annotationStorageKey001Y5()) || "{}";
}

function pushUndoSnapshot001Y5(reason = "action") {
  if (undoSuppress001Y5) return;

  const snap = currentAnnotationSnapshot001Y5();
  const stack = getUndoStack001Y5();

  if (stack.length && stack[stack.length - 1].snapshot === snap) {
    return;
  }

  stack.push({
    snapshot: snap,
    reason,
    at: new Date().toISOString(),
  });

  saveUndoStack001Y5(stack);
}

function refreshAfterUndo001Y5() {
  try { refreshInfo(); } catch (e) {}
  try { refreshPointList(); } catch (e) {}
  try { draw(); } catch (e) {}
  try { updateTimeline(); } catch (e) {}
}

function undoLastAction001Y5() {
  const stack = getUndoStack001Y5();

  if (!stack.length) {
    refreshUndoCount001Y5();
    return;
  }

  const last = stack.pop();
  localStorage.setItem(annotationStorageKey001Y5(), last.snapshot || "{}");
  saveUndoStack001Y5(stack);
  refreshAfterUndo001Y5();
}

function refreshUndoCount001Y5() {
  const el = document.getElementById("undoCount001Y5");
  if (!el) return;

  const n = getUndoStack001Y5().length;
  el.textContent = "historique : " + n;
}

function wrapUndo001Y5(name, reason) {
  const original = window[name];
  if (typeof original !== "function") return;

  if (original.__undoWrapped001Y5) return;

  const wrapped = function(...args) {
    if (!undoSuppress001Y5) {
      pushUndoSnapshot001Y5(reason || name);
    }
    return original.apply(this, args);
  };

  wrapped.__undoWrapped001Y5 = true;
  window[name] = wrapped;
}

function installUndoWrappers001Y5() {
  wrapUndo001Y5("addHumanPoint", "point");
  wrapUndo001Y5("markInvisible", "invisible");
  wrapUndo001Y5("markUncertain", "uncertain");
  wrapUndo001Y5("deleteCurrentFrame", "delete_frame");
  wrapUndo001Y5("copyPreviousPoint", "copy_previous");
  wrapUndo001Y5("clearCurrent", "clear_segment");

  // Pour le pinceau, on veut un seul Ctrl+Z par trait complet.
  const originalStartBrush = window.startBrush001Y4;
  if (typeof originalStartBrush === "function" && !originalStartBrush.__undoWrapped001Y5) {
    const wrappedStartBrush = function(...args) {
      pushUndoSnapshot001Y5("brush_stroke");
      undoSuppress001Y5 = true;
      undoBrushOrEraseActive001Y5 = true;
      return originalStartBrush.apply(this, args);
    };
    wrappedStartBrush.__undoWrapped001Y5 = true;
    window.startBrush001Y4 = wrappedStartBrush;
  }

  const originalStopBrush = window.stopBrush001Y4;
  if (typeof originalStopBrush === "function" && !originalStopBrush.__undoWrapped001Y5) {
    const wrappedStopBrush = function(...args) {
      const result = originalStopBrush.apply(this, args);
      setTimeout(() => {
        if (undoBrushOrEraseActive001Y5) {
          undoSuppress001Y5 = false;
          undoBrushOrEraseActive001Y5 = false;
        }
      }, 0);
      return result;
    };
    wrappedStopBrush.__undoWrapped001Y5 = true;
    window.stopBrush001Y4 = wrappedStopBrush;
  }

  // Sécurité : si addHumanPointAtFrame001Y4 est appelé hors pinceau, il reste annulable.
  wrapUndo001Y5("addHumanPointAtFrame001Y4", "point_at_frame");

  // Pour la gomme glissée, un seul Ctrl+Z pour tout le geste.
  const originalErase = window.eraseNearCanvasEvent;
  if (typeof originalErase === "function" && !originalErase.__undoWrapped001Y5) {
    const wrappedErase = function(...args) {
      if (!undoSuppress001Y5) {
        pushUndoSnapshot001Y5("erase");
      }
      return originalErase.apply(this, args);
    };
    wrappedErase.__undoWrapped001Y5 = true;
    window.eraseNearCanvasEvent = wrappedErase;
  }
}

// Capture avant les handlers gomme existants.
canvas.addEventListener("mousedown", evt => {
  try {
    if (typeof toolMode001Y3 !== "undefined" && toolMode001Y3 === "erase") {
      pushUndoSnapshot001Y5("erase_drag");
      undoSuppress001Y5 = true;
      undoBrushOrEraseActive001Y5 = true;
    }
  } catch (e) {}
}, true);

window.addEventListener("mouseup", () => {
  if (undoBrushOrEraseActive001Y5) {
    setTimeout(() => {
      undoSuppress001Y5 = false;
      undoBrushOrEraseActive001Y5 = false;
    }, 0);
  }
}, true);

// Ctrl+Z / Cmd+Z.
document.addEventListener("keydown", ev => {
  const tag = ev.target && ev.target.tagName ? ev.target.tagName : "";

  // Dans un champ texte, on laisse le navigateur gérer son propre undo.
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;

  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
    ev.preventDefault();
    undoLastAction001Y5();
  }
}, true);

setTimeout(() => {
  installUndoWrappers001Y5();
  refreshUndoCount001Y5();
}, 300);

setInterval(refreshUndoCount001Y5, 1000);
"""


OLD_CLICK = r"""canvas.addEventListener("click", (evt) => {
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


NEW_CLICK = r"""canvas.addEventListener("click", (evt) => {
  if (toolMode001Y3 === "erase") {
    // La gomme agit déjà au mousedown / drag.
    // On évite un second effacement au click.
    return;
  }

  if (toolMode001Y3 === "brush") {
    return;
  }

  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});"""


def patch_html(in_html: Path, out_html: Path) -> None:
    text = in_html.read_text(encoding="utf-8")

    if "/* === PATCH 001Y5 undo === */" not in text:
        if "</style>" not in text:
            raise RuntimeError("Cannot find </style>.")
        text = text.replace("</style>", CSS + "\n</style>", 1)

    if "<!-- === PATCH 001Y5 undo === -->" not in text:
        marker = '<div class="eraser-help">'
        if marker in text:
            text = text.replace(marker, HTML + "\n" + marker, 1)
        else:
            marker2 = "</div>\n</main>"
            if marker2 not in text:
                raise RuntimeError("Cannot find eraser panel or </main>.")
            text = text.replace(marker2, HTML + "\n" + marker2, 1)

    if OLD_CLICK in text:
        text = text.replace(OLD_CLICK, NEW_CLICK, 1)

    if "// === PATCH 001Y5 undo ===" not in text:
        if "</body>" not in text:
            raise RuntimeError("Cannot find </body>.")
        text = text.replace("</body>", f"<script>\n{JS}\n</script>\n</body>", 1)

    out_html.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/ball_annotation_001Y4_brush_fixed.html")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y5_undo.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    patch_html(in_html, out_html)

    print(f"[001Y5] input : {in_html}")
    print(f"[001Y5] output: {out_html}")


if __name__ == "__main__":
    main()
