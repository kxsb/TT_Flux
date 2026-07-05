from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_scene_priors() -> dict[str, Any]:
    table_length_m = 2.74
    table_width_m = 1.525
    table_height_m = 0.76
    net_height_m = 0.1525
    ball_diameter_m = 0.040
    ball_radius_m = ball_diameter_m / 2.0

    return {
        "version": "003B0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Shared scene-object prior model for TTFlux ball tracking, table context, player masks, and later camera/homography reasoning.",

        "source_notes": {
            "table_tennis_rules": {
                "authority": "ITTF Laws of Table Tennis / ITTF Handbook",
                "notes": [
                    "Playing surface is rectangular, 2.74 m long and 1.525 m wide.",
                    "Playing surface lies in a horizontal plane 76 cm above the floor.",
                    "Net top is 15.25 cm above the playing surface.",
                    "Ball diameter is 40 mm and mass is 2.7 g."
                ]
            },
            "human_anthropometry": {
                "authority": "ANSUR II / CDC-NHANES",
                "notes": [
                    "Human dimensions are soft priors only.",
                    "ANSUR II is rich but military-population biased.",
                    "CDC/NHANES is useful for broad adult height references.",
                    "Do not use human priors as hard rejection rules without calibration."
                ]
            }
        },

        "coordinate_systems": {
            "image_px": {
                "x": "pixels to the right",
                "y": "pixels downward",
                "origin": "top-left image corner"
            },
            "table_uv": {
                "u": "0..1 across table width from annotated front_left to front_right",
                "v": "0..1 along table depth from annotated front edge to back edge",
                "corners": {
                    "front_left": {"u": 0.0, "v": 0.0},
                    "front_right": {"u": 1.0, "v": 0.0},
                    "back_right": {"u": 1.0, "v": 1.0},
                    "back_left": {"u": 0.0, "v": 1.0}
                },
                "net_line": {
                    "v": 0.5,
                    "meaning": "approximate visual/metric half-table division"
                }
            },
            "table_metric_m": {
                "origin": "table center on playing surface",
                "x_m": "table width axis, left/right, range [-0.7625, 0.7625]",
                "y_m": "table length axis, play direction, range [-1.37, 1.37]",
                "z_m": "height above playing surface; table surface z=0, floor z=-0.76",
                "corners": {
                    "front_left": {"x_m": -table_width_m / 2, "y_m": -table_length_m / 2, "z_m": 0.0},
                    "front_right": {"x_m": table_width_m / 2, "y_m": -table_length_m / 2, "z_m": 0.0},
                    "back_right": {"x_m": table_width_m / 2, "y_m": table_length_m / 2, "z_m": 0.0},
                    "back_left": {"x_m": -table_width_m / 2, "y_m": table_length_m / 2, "z_m": 0.0}
                }
            }
        },

        "regulatory_objects": {
            "table": {
                "length_m": table_length_m,
                "width_m": table_width_m,
                "height_above_floor_m": table_height_m,
                "aspect_ratio_length_over_width": round(table_length_m / table_width_m, 6),
                "side_line_width_m": 0.020,
                "end_line_width_m": 0.020,
                "doubles_center_line_width_m": 0.003,
                "surface": "dark, matt, rectangular playing surface"
            },
            "net": {
                "height_above_table_m": net_height_m,
                "post_height_m": net_height_m,
                "post_outside_sideline_m": 0.1525,
                "uv_line": {"v": 0.5},
                "metric_line": {
                    "y_m": 0.0,
                    "x_min_m": -table_width_m / 2,
                    "x_max_m": table_width_m / 2
                }
            },
            "ball": {
                "diameter_m": ball_diameter_m,
                "radius_m": ball_radius_m,
                "mass_kg": 0.0027,
                "allowed_visual_colors": ["white", "orange"],
                "expected_appearance": "small, matte, often motion-blurred, often partly occluded"
            }
        },

        "scene_object_taxonomy": {
            "environment": {
                "stable": [
                    "table_surface",
                    "table_edges",
                    "net",
                    "floor",
                    "barriers",
                    "background",
                    "logos",
                    "scoreboard"
                ],
                "dynamic_or_distracting": [
                    "referee",
                    "spectators",
                    "lighting_flares",
                    "compression_blocks",
                    "white_text_or_logos"
                ],
                "purpose_for_tracker": [
                    "define plausible ball zones",
                    "detect false positives on floor/logos/barriers",
                    "project trajectory into table plane",
                    "estimate camera/homography"
                ]
            },
            "players": {
                "instances": ["player_A", "player_B"],
                "parts": [
                    "head",
                    "torso",
                    "upper_arm",
                    "forearm",
                    "hand",
                    "racket",
                    "thigh",
                    "lower_leg",
                    "foot"
                ],
                "high_false_positive_parts": [
                    "hand",
                    "racket_edge",
                    "white shoe",
                    "wrist",
                    "logo_on_shirt"
                ],
                "occlusion_parts": [
                    "hand",
                    "forearm",
                    "torso",
                    "racket"
                ],
                "purpose_for_tracker": [
                    "mark occlusion zones",
                    "downweight candidate inside body mask",
                    "detect false ball on shoe/hand/logo",
                    "allow inferred ball state near racket/hand"
                ]
            },
            "ball": {
                "states": [
                    "visible",
                    "motion_blurred",
                    "partially_occluded",
                    "fully_occluded_inferred",
                    "not_visible",
                    "false_candidate"
                ],
                "events": [
                    "serve",
                    "racket_contact",
                    "table_bounce",
                    "net_contact",
                    "out_of_play",
                    "occlusion_enter",
                    "occlusion_exit"
                ],
                "purpose_for_tracker": [
                    "separate visual evidence from inferred trajectory",
                    "combine appearance, motion, table context, and player context"
                ]
            },
            "camera": {
                "states": [
                    "fixed",
                    "slow_pan",
                    "zoom",
                    "cut",
                    "shake"
                ],
                "models": [
                    "single_homography_per_clip",
                    "keyframed_homography",
                    "interpolated_homography",
                    "cut_segmented_homography"
                ],
                "purpose_for_tracker": [
                    "normalize scene coordinates",
                    "avoid using stale table geometry after camera movement",
                    "detect when scene model must be reset"
                ]
            }
        },

        "human_priors_soft": {
            "adult_height_m": {
                "very_low": 1.45,
                "reference_female_mean_cdc_m": 1.613,
                "reference_male_mean_cdc_m": 1.750,
                "very_high": 2.10,
                "usage": "soft plausibility prior only; not a hard classifier"
            },
            "silhouette_ratios_against_table": {
                "standing_height_over_table_height_soft_range": [1.9, 3.0],
                "visible_torso_width_over_table_width_soft_range": [0.12, 0.45],
                "player_bbox_height_over_table_image_height_soft_range": [0.8, 3.5],
                "usage": "use only after camera/table scale is known"
            },
            "risk_zones": {
                "hand_or_racket": {
                    "ball_occlusion_likelihood": "high",
                    "false_positive_likelihood": "medium"
                },
                "shoes_or_floor_near_feet": {
                    "ball_occlusion_likelihood": "low",
                    "false_positive_likelihood": "high"
                },
                "torso_or_shirt_logo": {
                    "ball_occlusion_likelihood": "medium",
                    "false_positive_likelihood": "medium"
                }
            }
        },

        "planned_features": {
            "table_context_003B": [
                "has_table_model",
                "table_keyframe_distance_frames",
                "point_inside_table_quad_ratio",
                "point_distance_to_table_px_median",
                "point_distance_to_net_px_median",
                "point_uv_u_min_max",
                "point_uv_v_min_max",
                "trajectory_crosses_net_line",
                "trajectory_crosses_table_bounds",
                "trajectory_side_consistency",
                "below_table_risk",
                "near_table_edge_ratio",
                "far_from_table_ratio"
            ],
            "player_context_003C": [
                "inside_player_mask_ratio",
                "near_player_mask_ratio",
                "inside_hand_or_racket_proxy_ratio",
                "near_foot_or_floor_proxy_ratio",
                "occlusion_entry_exit_count",
                "candidate_on_body_risk"
            ],
            "ball_context_003D": [
                "visual_ball_score",
                "motion_plausibility_score",
                "table_context_score",
                "player_occlusion_score",
                "visible_vs_inferred_state",
                "final_ball_track_score"
            ]
        },

        "implementation_order": [
            {
                "step": "003B0",
                "name": "scene priors and object schema",
                "status": "current"
            },
            {
                "step": "003B1",
                "name": "table geometry bootstrap",
                "goal": "use existing partial annotations + table aspect ratio + line/edge detection to infer usable table models"
            },
            {
                "step": "003B2",
                "name": "trajectory projection to table context",
                "goal": "compute table_context features for existing 22 segments"
            },
            {
                "step": "003C",
                "name": "coarse player/silhouette masks",
                "goal": "detect player zones and common false-positive body parts"
            },
            {
                "step": "003D",
                "name": "contextual ball arbiter",
                "goal": "combine trajectory, table, player, and ball states"
            }
        ],

        "design_decisions": {
            "do_not_require_perfect_manual_table_annotation": True,
            "use_table_dimensions_as_constraint": True,
            "estimate_camera_angle_later": True,
            "first_camera_model": "2D homography or keyframed homography, not full 3D camera solve",
            "first_player_model": "coarse masks/zones, not full pose estimation",
            "first_ball_model": "stateful visible/occluded/inferred candidate, not pure crop classifier"
        }
    }


def build_schema(priors: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "003B0_schema",
        "scene_state_record": {
            "clip_id": "string",
            "frame": "int",
            "camera_state": {
                "mode": "fixed | slow_pan | zoom | cut | unknown",
                "homography_id": "string|null",
                "confidence": "float 0..1"
            },
            "environment": {
                "table": {
                    "visible": "bool",
                    "quad_px": "list of 4 image points or null",
                    "homography_image_to_table_uv": "3x3 matrix or null",
                    "confidence": "float 0..1"
                },
                "net": {
                    "line_px": "two image points or null",
                    "confidence": "float 0..1"
                },
                "floor": {
                    "mask_id": "optional",
                    "confidence": "float 0..1"
                }
            },
            "players": {
                "player_A": {
                    "mask_id": "optional",
                    "bbox_px": "optional",
                    "confidence": "float 0..1"
                },
                "player_B": {
                    "mask_id": "optional",
                    "bbox_px": "optional",
                    "confidence": "float 0..1"
                }
            },
            "ball": {
                "candidates": [
                    {
                        "x_px": "float",
                        "y_px": "float",
                        "appearance_score": "float",
                        "motion_score": "float",
                        "table_context_score": "float",
                        "player_context_score": "float",
                        "state": "visible | blurred | partial | inferred | false_candidate"
                    }
                ]
            }
        },
        "segment_feature_record": {
            "review_id": "string",
            "clip_id": "string",
            "segment_name": "string",
            "first_frame": "int",
            "last_frame": "int",
            "table_context_features": priors["planned_features"]["table_context_003B"],
            "player_context_features": priors["planned_features"]["player_context_003C"],
            "ball_context_features": priors["planned_features"]["ball_context_003D"]
        }
    }


def html_list(items: list[Any]) -> str:
    return "".join(f"<li><code>{x}</code></li>" for x in items)


def write_html(path: Path, priors: dict[str, Any], schema: dict[str, Any]) -> None:
    table = priors["regulatory_objects"]["table"]
    net = priors["regulatory_objects"]["net"]
    ball = priors["regulatory_objects"]["ball"]

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux scene priors 003B0</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px;max-height:360px;overflow:auto}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>TTFlux · scene priors 003B0</h1>

<section>
<h2>Constantes réglementaires</h2>
<table>
<tr><th>Table</th><td>{table["length_m"]} m × {table["width_m"]} m · hauteur {table["height_above_floor_m"]} m · ratio {table["aspect_ratio_length_over_width"]}</td></tr>
<tr><th>Filet</th><td>hauteur {net["height_above_table_m"]} m au-dessus du plan de jeu</td></tr>
<tr><th>Balle</th><td>diamètre {ball["diameter_m"]} m · masse {ball["mass_kg"]} kg</td></tr>
</table>
</section>

<section>
<h2>Familles d'objets</h2>
<div class="grid">
<section>
<h3>Environnement</h3>
<ul>{html_list(priors["scene_object_taxonomy"]["environment"]["stable"])}</ul>
</section>
<section>
<h3>Joueurs</h3>
<ul>{html_list(priors["scene_object_taxonomy"]["players"]["parts"])}</ul>
</section>
<section>
<h3>Balle</h3>
<ul>{html_list(priors["scene_object_taxonomy"]["ball"]["states"])}</ul>
</section>
<section>
<h3>Caméra</h3>
<ul>{html_list(priors["scene_object_taxonomy"]["camera"]["models"])}</ul>
</section>
</div>
</section>

<section>
<h2>Features prévues</h2>
<h3>003B table_context</h3>
<ul>{html_list(priors["planned_features"]["table_context_003B"])}</ul>
<h3>003C player_context</h3>
<ul>{html_list(priors["planned_features"]["player_context_003C"])}</ul>
<h3>003D ball_context</h3>
<ul>{html_list(priors["planned_features"]["ball_context_003D"])}</ul>
</section>

<section>
<h2>Décisions</h2>
<pre>{json.dumps(priors["design_decisions"], indent=2, ensure_ascii=False)}</pre>
</section>

<section>
<h2>Schéma scene_state</h2>
<pre>{json.dumps(schema, indent=2, ensure_ascii=False)}</pre>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_readme(path: Path, priors: dict[str, Any]) -> None:
    text = f"""# TTFlux scene priors 003B0

But : stabiliser le passage du tracking balle vers une scène structurée en objets.

## Familles

- Environnement : table, filet, sol, barrières, logos, score, arrière-plan.
- Joueurs : silhouettes, bras, mains, jambes, pieds, raquettes, zones d'occlusion.
- Balle : candidats, visible, floue, partielle, occultée, inférée, fausse candidate.
- Caméra : homographie, keyframes, mouvements, cuts.

## Constantes

- Table : {priors["regulatory_objects"]["table"]["length_m"]} m × {priors["regulatory_objects"]["table"]["width_m"]} m.
- Hauteur table : {priors["regulatory_objects"]["table"]["height_above_floor_m"]} m.
- Filet : {priors["regulatory_objects"]["net"]["height_above_table_m"]} m.
- Balle : diamètre {priors["regulatory_objects"]["ball"]["diameter_m"]} m.

## Ordre prévu

1. 003B1 : bootstrap table par géométrie et contraintes.
2. 003B2 : projection des trajectoires dans le repère table.
3. 003C : silhouettes / zones joueurs grossières.
4. 003D : arbitre contextuel balle-table-joueur.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/batch_001E/scene_priors_003B0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priors = build_scene_priors()
    schema = build_schema(priors)

    priors_path = out_dir / "scene_priors_003B0.json"
    schema_path = out_dir / "scene_object_schema_003B0.json"
    html_path = out_dir / "scene_priors_003B0.html"
    readme_path = out_dir / "README_scene_003B0.md"

    write_json(priors_path, priors)
    write_json(schema_path, schema)
    write_html(html_path, priors, schema)
    write_readme(readme_path, priors)

    print(f"[003B0] out dir : {out_dir}")
    print(f"[003B0] priors  : {priors_path}")
    print(f"[003B0] schema  : {schema_path}")
    print(f"[003B0] html    : {html_path}")
    print(f"[003B0] readme  : {readme_path}")


if __name__ == "__main__":
    main()
