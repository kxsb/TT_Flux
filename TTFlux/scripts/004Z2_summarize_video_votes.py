from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004Z2"


RATING_SCORE = {
    "bon": 3,
    "moyen": 2,
    "mauvais": 1,
    "inutilisable": 0,
}


def split_tags(s: str) -> list[str]:
    return [x.strip() for x in str(s or "").split("|") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", default="runs/rally_video_vote_004Z_p080_r030/votes_004Z.csv")
    ap.add_argument("--packet", default="runs/rally_video_vote_004Z_p080_r030/vote_packet_004Z.csv")
    ap.add_argument("--out-dir", default="runs/rally_video_vote_004Z2_summary_p080_r030")
    args = ap.parse_args()

    root = Path.cwd()

    votes_path = Path(args.votes)
    packet_path = Path(args.packet)
    out_dir = Path(args.out_dir)

    if not votes_path.is_absolute():
        votes_path = root / votes_path
    if not packet_path.is_absolute():
        packet_path = root / packet_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    if not votes_path.is_file():
        raise SystemExit(f"votes absent: {votes_path}")

    votes = pd.read_csv(votes_path).fillna("")

    if packet_path.is_file():
        packet = pd.read_csv(packet_path).fillna("")
    else:
        packet = pd.DataFrame()

    votes["rating_norm"] = votes["rating"].astype(str).str.strip().str.lower()
    votes["rating_score"] = votes["rating_norm"].map(RATING_SCORE).fillna(-1).astype(int)
    votes["is_accepted_004Z2"] = votes["rating_norm"].isin(["bon", "moyen"]).astype(int)
    votes["is_strong_004Z2"] = votes["rating_norm"].eq("bon").astype(int)
    votes["is_rejected_004Z2"] = votes["rating_norm"].isin(["mauvais", "inutilisable"]).astype(int)

    tag_counter = Counter()
    for tags in votes.get("tags", pd.Series([], dtype=str)).astype(str):
        tag_counter.update(split_tags(tags))

    rating_counts = votes["rating_norm"].value_counts().to_dict()

    merged = votes.copy()

    if not packet.empty and "review_id" in packet.columns:
        keep_cols = [
            c for c in [
                "review_id",
                "duration_vote_sec",
                "duration_src_sec",
                "path_points",
                "emitted_points",
                "coverage_points",
                "tracklets",
                "longest_tracklet_points",
                "score_med_emitted",
                "review_score_med_004Y",
                "overlay_src",
                "asset_name",
            ]
            if c in packet.columns
        ]

        merged = votes.merge(
            packet[keep_cols],
            on="review_id",
            how="left",
            suffixes=("", "_packet"),
        )

    accepted = merged[merged["is_accepted_004Z2"] == 1].copy()
    rejected = merged[merged["is_rejected_004Z2"] == 1].copy()
    strong = merged[merged["is_strong_004Z2"] == 1].copy()

    votes_n = len(merged)
    accepted_n = int(merged["is_accepted_004Z2"].sum()) if votes_n else 0
    strong_n = int(merged["is_strong_004Z2"].sum()) if votes_n else 0
    rejected_n = int(merged["is_rejected_004Z2"].sum()) if votes_n else 0

    accepted_ratio = round(accepted_n / max(1, votes_n), 4)
    strong_ratio = round(strong_n / max(1, votes_n), 4)
    rejected_ratio = round(rejected_n / max(1, votes_n), 4)

    false_tags = [
        "faux_joueur",
        "faux_raquette",
        "faux_table",
        "faux_logo",
    ]
    false_tag_total = sum(tag_counter.get(t, 0) for t in false_tags)
    too_rare = tag_counter.get("points_trop_rares", 0) + tag_counter.get("no_ball_trop_agressif", 0)
    good_tags = tag_counter.get("points_sur_balle", 0) + tag_counter.get("no_ball_ok", 0)

    if votes_n < 8:
        recommendation = "vote_more"
        decision = "Pas assez de votes. Vise au moins 10 à 15 vidéos."
    elif accepted_ratio >= 0.65 and false_tag_total <= max(2, votes_n * 0.35):
        recommendation = "run_all160_p080_r030"
        decision = "Gate p080_r030 validé pour lancer les 160 rallys."
    elif too_rare > false_tag_total and accepted_ratio >= 0.45:
        recommendation = "try_more_permissive_gate"
        decision = "Qualité correcte mais trop peu de points. Tester p075_r025 en vidéo."
    elif false_tag_total > too_rare:
        recommendation = "tighten_or_retrain"
        decision = "Trop de faux positifs. Ne pas lancer all160 ; resserrer le gate ou enrichir le modèle."
    else:
        recommendation = "review_mixed"
        decision = "Résultat mixte. Regarder les commentaires et tags avant all160."

    tags_rows = [
        {"tag": tag, "count": count}
        for tag, count in tag_counter.most_common()
    ]

    out_votes = out_dir / "video_votes_merged_004Z2.csv"
    out_accepted = out_dir / "video_votes_accepted_004Z2.csv"
    out_rejected = out_dir / "video_votes_rejected_004Z2.csv"
    out_tags = out_dir / "video_vote_tags_004Z2.csv"
    out_json = out_dir / "video_vote_summary_004Z2.json"

    merged.to_csv(out_votes, index=False, encoding="utf-8")
    accepted.to_csv(out_accepted, index=False, encoding="utf-8")
    rejected.to_csv(out_rejected, index=False, encoding="utf-8")
    pd.DataFrame(tags_rows).to_csv(out_tags, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "summarize_human_video_votes_for_gated_ball_tracking",
        "votes_csv": str(votes_path),
        "packet_csv": str(packet_path),
        "votes": votes_n,
        "rating_counts": rating_counts,
        "accepted": accepted_n,
        "strong": strong_n,
        "rejected": rejected_n,
        "accepted_ratio": accepted_ratio,
        "strong_ratio": strong_ratio,
        "rejected_ratio": rejected_ratio,
        "tag_counts": dict(tag_counter.most_common()),
        "false_tag_total": int(false_tag_total),
        "too_rare_tag_total": int(too_rare),
        "good_tag_total": int(good_tags),
        "recommendation": recommendation,
        "decision": decision,
        "outputs": {
            "merged": str(out_votes),
            "accepted": str(out_accepted),
            "rejected": str(out_rejected),
            "tags": str(out_tags),
        },
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004Z2 status=OK")
    print("votes=", votes_n)
    print("rating_counts=", json.dumps(rating_counts, ensure_ascii=False))
    print("accepted_ratio=", accepted_ratio)
    print("strong_ratio=", strong_ratio)
    print("rejected_ratio=", rejected_ratio)
    print("tag_counts=", json.dumps(dict(tag_counter.most_common()), ensure_ascii=False))
    print("recommendation=", recommendation)
    print("decision=", decision)
    print("wrote", out_votes)
    print("wrote", out_accepted)
    print("wrote", out_rejected)
    print("wrote", out_tags)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
