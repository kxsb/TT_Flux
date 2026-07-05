from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


RULESET_ID = "003AA_combined_context_appearance_shadow_rules"


def main() -> int:
    root = Path.cwd()
    out_dir = root / "runs" / "shadow_rules_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    ruleset = {
        "version": "003AA",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ruleset_id": RULESET_ID,
        "policy": "shadow_only_no_live_filter_no_delete",
        "strict_reject_rules": [
            {
                "rule_id": "003L_context_player_table_accel",
                "family": "context",
                "expression": (
                    "inside_player_motion_mask_ratio_003C >= 0.60 "
                    "AND point_distance_to_table_px_median <= 210.0 "
                    "AND max_accel >= 20.0"
                ),
                "features": {
                    "inside_player_motion_mask_ratio_003C": {">=": 0.60},
                    "point_distance_to_table_px_median": {"<=": 210.0},
                    "max_accel": {">=": 20.0}
                },
                "validated_hits": [
                    "batch_001E/R0008",
                    "batch_001E/R0009",
                    "batch_001E/R0010",
                    "batch_001E/R0014",
                    "batch_001E/R0020",
                    "batch_001E/R0021"
                ],
                "known_dangerous_hits": []
            },
            {
                "rule_id": "003Y_strict_appearance_logo_shoes",
                "family": "appearance",
                "expression": (
                    "center_blob_fill_med >= 0.75924 "
                    "AND micro_distance_med_001O >= 506.0725"
                ),
                "features": {
                    "center_blob_fill_med": {">=": 0.75924},
                    "micro_distance_med_001O": {">=": 506.0725}
                },
                "validated_hits": [
                    "batch_002A/R0005",
                    "batch_002A/R0015"
                ],
                "known_dangerous_hits": []
            }
        ],
        "review_rules": [
            {
                "rule_id": "003Y_review_appearance_logo_shoes",
                "family": "appearance",
                "expression": (
                    "center_blob_fill_med >= 0.79137 "
                    "AND micro_distance_med_001O >= 9.1336"
                ),
                "features": {
                    "center_blob_fill_med": {">=": 0.79137},
                    "micro_distance_med_001O": {">=": 9.1336}
                },
                "validated_reject_hits": [
                    "batch_002A/R0005",
                    "batch_002A/R0015",
                    "batch_001E/R0010",
                    "batch_001E/R0016",
                    "batch_002B/R0002"
                ],
                "known_dangerous_hits": [],
                "blocking_unsure_hits": [
                    "batch_001E/R0004"
                ],
                "decision": "human_review_queue_only"
            }
        ],
        "combined_validation_snapshot": {
            "strict_reject_total": 8,
            "strict_reject_ids": [
                "batch_001E/R0008",
                "batch_001E/R0009",
                "batch_001E/R0010",
                "batch_001E/R0014",
                "batch_001E/R0020",
                "batch_001E/R0021",
                "batch_002A/R0005",
                "batch_002A/R0015"
            ],
            "strict_dangerous_hit_total": 0,
            "review_extra_ids": [
                "batch_001E/R0004",
                "batch_001E/R0016",
                "batch_002B/R0002"
            ],
            "review_known_reject_ids": [
                "batch_001E/R0016",
                "batch_002B/R0002"
            ],
            "review_unsure_ids": [
                "batch_001E/R0004"
            ]
        },
        "next_required_validation": [
            "Run on next fresh batch when inventory grows or new clips are available.",
            "Annotate all strict/review hits.",
            "Do not promote live before at least one larger fresh batch confirms zero keep/partial hits."
        ]
    }

    out_json = out_dir / f"{RULESET_ID}.json"
    out_json.write_text(json.dumps(ruleset, ensure_ascii=False, indent=2), encoding="utf-8")

    print("003AA status=OK_RULESET_FROZEN")
    print("ruleset_json=", out_json)
    print("strict_rules=2")
    print("review_rules=1")
    print("strict_known_dangerous=0")
    print("review_blocking_unsure=batch_001E/R0004")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
