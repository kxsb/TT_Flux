from pathlib import Path
import json
import pandas as pd

run = Path("runs/batch_004F_full240")

def read_json(p: Path):
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return {"_error": repr(e)}

def read_csv(p: Path):
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"CSV_ERROR {p}: {e}")
        return None

def find_col(df, contains_any):
    if df is None:
        return None
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in contains_any):
            return c
    return None

def count_col(df, col):
    if df is None or col is None or col not in df.columns:
        return {}
    return df[col].fillna("").astype(str).value_counts().to_dict()

def print_file(label, p):
    print(f"{label}: {p} exists={p.is_file()}")

print("\n=== 004G2 QUICK SUMMARY ===")
print("run =", run)

# 004G1 operational
op_csv = run / "operational_manifest_001T2.csv"
op = read_csv(op_csv)

print("\n[004G1 / operational]")
if op is None:
    print("operational_csv_missing =", op_csv)
else:
    print("segments =", len(op))
    guess_col = find_col(op, ["guess"])
    if guess_col:
        print("guess_col =", guess_col)
        print("guesses =", count_col(op, guess_col))

# 001G trajectory
traj_csv = run / "review_manifest_001T2_trajectory.csv"
traj = read_csv(traj_csv)

print("\n[001G]")
if traj is None:
    print("trajectory_csv_missing =", traj_csv)
else:
    print("segments =", len(traj))
    found_col = find_col(traj, ["trajectory_found", "found"])
    guess_col = find_col(traj, ["review_guess", "guess", "risk", "recommendation"])
    if found_col:
        print("trajectory_found_col =", found_col)
        print("trajectory_found_counts =", count_col(traj, found_col))
    if guess_col:
        print("guess_col =", guess_col)
        print("guesses =", count_col(traj, guess_col))

# 001L appearance
app_csv = run / "appearance_features_001T2.csv"
app_json = read_json(run / "appearance_summary_001T2.json")
app = read_csv(app_csv)

print("\n[001L]")
if app is None:
    print("appearance_csv_missing =", app_csv)
else:
    print("appearance_rows =", len(app))
    ok_col = find_col(app, ["appearance_ok", "ok"])
    guess_col = find_col(app, ["human_label", "guess", "target", "label"])
    if ok_col:
        print("appearance_ok_col =", ok_col)
        print("appearance_ok_counts =", count_col(app, ok_col))
    if guess_col:
        print("appearance_guess_col =", guess_col)
        print("appearance_guess_counts =", count_col(app, guess_col))
if app_json:
    print("appearance_summary_json =", app_json)

# 001N center
center_csv = run / "center_blob_features_001T2.csv"
center_json = read_json(run / "center_blob_summary_001T2.json")
center = read_csv(center_csv)

print("\n[001N]")
if center is None:
    print("center_csv_missing =", center_csv)
else:
    print("center_rows =", len(center))
    guess_col = find_col(center, ["center_blob_guess", "guess"])
    if guess_col:
        print("center_guess_col =", guess_col)
        print("center_guess_counts =", count_col(center, guess_col))
if center_json:
    print("center_summary_json =", center_json)

# 001O micro
micro_csv = run / "micro_blob_features_001T2.csv"
micro_json = read_json(run / "micro_blob_summary_001T2.json")
micro = read_csv(micro_csv)

print("\n[001O]")
if micro is None:
    print("micro_csv_missing =", micro_csv)
else:
    print("micro_rows =", len(micro))
    guess_col = find_col(micro, ["micro_blob_guess", "guess"])
    if guess_col:
        print("micro_guess_col =", guess_col)
        print("micro_guess_counts =", count_col(micro, guess_col))
if micro_json:
    print("micro_summary_json =", micro_json)

# 001S arbiter
arb_dir = run / "arbiter_001T2"
arb_csvs = list(arb_dir.glob("*.csv")) if arb_dir.is_dir() else []

print("\n[001S]")
print("arbiter_dir_exists =", arb_dir.is_dir())
print("arbiter_csvs =", [str(p) for p in arb_csvs])

for p in arb_csvs:
    df = read_csv(p)
    if df is None:
        continue
    print(f"arbiter_file = {p.name}")
    print("rows =", len(df))
    dec_col = find_col(df, ["decision_001S", "decision", "arbiter", "recommendation", "status"])
    if dec_col:
        print("decision_col =", dec_col)
        print("decisions =", count_col(df, dec_col))

print("\n[FILES]")
print_file("operational_manifest", run / "operational_manifest_001T2.html")
print_file("trajectory_manifest", run / "review_manifest_001T2_trajectory.html")
print_file("appearance_html", run / "appearance_features_001T2.html")
print_file("center_html", run / "center_blob_features_001T2.html")
print_file("micro_html", run / "micro_blob_features_001T2.html")
print_file("arbiter_html", run / "arbiter_001T2" / "arbiter_001S.html")

print("\n=== END 004G2 ===")
