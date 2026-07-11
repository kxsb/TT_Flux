from __future__ import annotations

import sys
from pathlib import Path


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: bloc trouvé {count} fois au lieu de 1."
        )

    return text.replace(old, new, 1)


path = Path(sys.argv[1]).resolve()
text = path.read_text(encoding="utf-8-sig")

text = replace_once(
    text,
    "_ttnet_gt_evaluation_003D_I10",
    "_ttnet_temporal_sampling_003D_I11D",
    "répertoire de sortie",
)

text = text.replace(
    "003D_I10_TTNet_last_frame_target_vs_OpenTTGames_GT",
    "003D_I11D_TTNet_120fps_with_50fps_temporal_sampling",
)

text = replace_once(
    text,
    """HALF_SEQUENCE = 4
""",
    """HALF_SEQUENCE = 4

# Neuf échantillons espacés comme une source à 50 fps,
# exprimés dans la timeline OpenTTGames à 120 fps.
TEMPORAL_OFFSETS_50FPS_ON_120FPS = (
    -19,
    -17,
    -14,
    -12,
    -10,
    -7,
    -5,
    -2,
    0,
)

TEMPORAL_HISTORY_SPAN = 19
""",
    "constantes temporelles",
)

text = replace_once(
    text,
    """INPUT_WIDTH = 320
""",
    """CHECKPOINTS = [
    item
    for item in CHECKPOINTS
    if item["model_id"] == "ttnet_120fps"
]

INPUT_WIDTH = 320
""",
    "filtrage du checkpoint",
)

old_indices = """    # TTNet est entraîné pour prédire la balle sur la dernière
    # image des neuf images d'entrée, et non sur l'image centrale.
    indices = [
        target_frame
        - (SEQUENCE_LENGTH - 1 - sequence_index)
        * temporal_stride
        for sequence_index
        in range(SEQUENCE_LENGTH)
    ]
"""

new_indices = """    if temporal_stride != 1:
        raise RuntimeError(
            "I11D doit utiliser le checkpoint ttnet_120fps "
            "avec temporal_stride=1."
        )

    indices = [
        target_frame + offset
        for offset
        in TEMPORAL_OFFSETS_50FPS_ON_120FPS
    ]
"""

text = replace_once(
    text,
    old_indices,
    new_indices,
    "indices de séquence",
)

old_history = """    history_span = (
        (SEQUENCE_LENGTH - 1)
        * temporal_stride
    )
"""

new_history = """    history_span = TEMPORAL_HISTORY_SPAN
"""

text = replace_once(
    text,
    old_history,
    new_history,
    "historique temporel",
)

old_summary = """        "target_frame_policy":
            "last_input_frame",
        "history_span_frames":
            history_span,
"""

new_summary = """        "target_frame_policy":
            "last_input_frame",
        "temporal_sampling_policy":
            "simulate_50fps_on_120fps_source",
        "temporal_offsets":
            list(TEMPORAL_OFFSETS_50FPS_ON_120FPS),
        "history_span_frames":
            history_span,
"""

text = replace_once(
    text,
    old_summary,
    new_summary,
    "métadonnées du résumé",
)

path.write_text(
    text,
    encoding="utf-8",
)

print("PATCH_003D_I11D_OK")
print("script =", path)
