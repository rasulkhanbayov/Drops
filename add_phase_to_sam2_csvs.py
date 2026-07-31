"""
Patch all existing SAM2 v3 CSVs to add a 'phase' column using
impact_frame and liftoff_frame from feature_table.json.

Phases:
  falling    — frame < impact_frame
  spreading  — impact_frame <= frame <= liftoff_frame
  rebounding — frame > liftoff_frame

Run once; idempotent (skips files that already have the phase column).
"""
import csv
import json
import io
from pathlib import Path

FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")

FOLDER_MAP = {
    "02182026":  Path("/home/ubuntu/materials/results_drops/02182026_sam2_v3_results"),
    "03242026":  Path("/home/ubuntu/materials/results_drops/03242026_sam2_v3_results"),
    "05052026":  Path("/home/ubuntu/materials/results_drops/05052026_sam2_v3_results"),
    "05112026":  Path("/home/ubuntu/materials/results_drops/05112026_sam2_v3_results"),
    "05122026":  Path("/home/ubuntu/materials/results_drops/05122026_sam2_v3_results"),
    "05172026":  Path("/home/ubuntu/materials/results_drops/05172026_sam2_v3_results"),
}

NEW_FIELDNAMES = [
    'frame', 'phase', 'drop_id', 'cx', 'cy', 'area_px', 'percentage',
    'detection_method', 'distance_px', 'velocity_px_per_s', 'velocity_mm_s',
]


def assign_phase(fi, impact, liftoff):
    if impact is None or liftoff is None:
        return ''
    if fi < impact:
        return 'falling'
    elif fi <= liftoff:
        return 'spreading'
    else:
        return 'rebounding'


def patch_csv(csv_path, impact_frame, liftoff_frame):
    text = csv_path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    if 'phase' in (reader.fieldnames or []):
        print(f"  [skip] {csv_path.name} — already has phase column")
        return

    rows = list(reader)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=NEW_FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        fi = int(row['frame'])
        row['phase'] = assign_phase(fi, impact_frame, liftoff_frame)
        writer.writerow(row)

    csv_path.write_text(out.getvalue())
    print(f"  [done] {csv_path.name}  (impact={impact_frame} liftoff={liftoff_frame})")


def main():
    features = json.loads(FEATURE_JSON.read_text())
    # Build lookup: (folder, stem) -> (impact_frame, liftoff_frame)
    lookup = {}
    for e in features:
        stem = Path(e['video']).stem
        lookup[(e['folder'], stem)] = (e.get('impact_frame'), e.get('liftoff_frame'))

    total_patched = 0
    for folder_key, out_dir in FOLDER_MAP.items():
        if not out_dir.exists():
            print(f"[skip] {out_dir} — directory not found")
            continue
        csvs = sorted(out_dir.glob("*_sam2.csv"))
        print(f"\n=== {folder_key} ({len(csvs)} CSVs) ===")
        for csv_path in csvs:
            # Derive video stem from CSV name: strip trailing _sam2
            stem = csv_path.stem  # e.g. "cainhcg1_sam2"
            if stem.endswith('_sam2'):
                stem = stem[:-5]  # "cainhcg1"
            impact, liftoff = lookup.get((folder_key, stem), (None, None))
            if impact is None:
                print(f"  [warn] {csv_path.name} — no feature_table entry for '{stem}'")
            patch_csv(csv_path, impact, liftoff)
            total_patched += 1

    print(f"\nDone. Processed {total_patched} CSVs.")


if __name__ == "__main__":
    main()
