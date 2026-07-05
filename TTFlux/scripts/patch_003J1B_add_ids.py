from pathlib import Path

p = Path("scripts/003J1B_unordered_table_points.py")
s = p.read_text(encoding="utf-8")

if 'ap.add_argument("--ids"' not in s:
    s = s.replace(
        'ap.add_argument("--max", type=int, default=0)',
        'ap.add_argument("--max", type=int, default=0)\n'
        '    ap.add_argument("--ids", default="", help="liste de review ids separee par virgules, ex: R0022,R0019")'
    )

if 'ids_filter_003J1B' not in s:
    s = s.replace(
        'df["review_id_003G"] = df["review_id_003G"].astype(str)\n\n'
        '    if not args.all:',
        'df["review_id_003G"] = df["review_id_003G"].astype(str)\n\n'
        '    ids_filter_003J1B = [x.strip() for x in str(args.ids or "").split(",") if x.strip()]\n'
        '    if ids_filter_003J1B:\n'
        '        df = df[df["review_id_003G"].isin(ids_filter_003J1B)].copy()\n\n'
        '    if not args.all:'
    )

p.write_text(s, encoding="utf-8")
print("patched", p)
