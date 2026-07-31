"""
Extract retraction and rebound velocities from v3 timeseries CSVs.

Retraction velocity:
  - Start at β_max frame (peak spread_width_px in the spreading phase)
  - Track the right edge of the contact line: x_right = cx_px + spread_width_px / 2
  - Fit linear regression to the first N_FIT frames where width is still decreasing
  - slope (px/s, negative → contact line moving left → retraction) → |slope| / px_per_mm → mm/s

Rebound velocity:
  - Start at liftoff_frame (first rebounding frame with cx/cy/radius)
  - Track the topmost pixel: y_top = cy_px - radius_px (image coords: smaller y = higher)
  - Fit linear regression to the first N_FIT valid rebounding frames
  - slope is negative (droplet rises → y decreases) → |slope| / px_per_mm → mm/s

Outputs:
  - Adds retraction_velocity_mm_s, rebound_velocity_mm_s to feature_table.json
  - Adds right_edge_px, top_edge_px columns to each timeseries CSV
"""

import csv
import io
import json
import math
from pathlib import Path

FPS        = 2996.766489
DT         = 1.0 / FPS   # seconds per frame
N_FIT      = 10           # frames used for linear regression
MIN_FIT    = 3            # minimum fit points required
MAX_RET_MM_S = 2000.0     # physical cap: retraction can't exceed ~2x impact speed
MAX_REB_MM_S = 1500.0     # physical cap: rebound vel < impact vel (~1100-1600 mm/s)
MAX_SPREAD_PX = 600       # β_max above this is a measurement artifact, not lamella

FEATURE_JSON = Path("feature_table.json")

V3_DIRS = {
    "02182026": Path("results_drops/02182026_v3_results"),
    "03242026": Path("results_drops/03242026_v3_results"),
    "05052026": Path("results_drops/05052026_v3_results"),
    "05112026": Path("results_drops/05112026_v3_results"),
    "05122026": Path("results_drops/05122026_v3_results"),
    "05172026": Path("results_drops/05172026_v3_results"),
}

PX_PER_MM = {
    "02182026": 65.625,
    "03242026": 65.625,
    "05052026": 66.0,
    "05112026": 66.5,
    "05122026": 56.0,
    "05172026": 56.0,
}


def linreg_slope(xs, ys):
    """Ordinary least-squares slope for paired lists xs, ys."""
    n = len(xs)
    if n < 2:
        return None
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    return (n * sxy - sx * sy) / denom


def process_video(csv_path, px_per_mm):
    """
    Read a v3 timeseries CSV, compute retraction and rebound velocities,
    write back with new right_edge_px and top_edge_px columns.
    Returns (retraction_vel_mm_s, rebound_vel_mm_s) or (None, None).
    """
    text = csv_path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    orig_fields = reader.fieldnames or []
    rows = list(reader)
    if not rows:
        return None, None

    # ── Parse numeric fields ──────────────────────────────────────────────────
    def flt(v):
        try:
            return float(v) if v not in ('', None) else None
        except ValueError:
            return None

    def itg(v):
        try:
            return int(v) if v not in ('', None) else None
        except ValueError:
            return None

    for r in rows:
        r['_frame']       = itg(r.get('frame'))
        r['_cx']          = flt(r.get('cx_px'))
        r['_cy']          = flt(r.get('cy_px'))
        r['_radius']      = flt(r.get('radius_px'))
        r['_spread']      = flt(r.get('spread_width_px'))
        r['_phase']       = r.get('phase', '')
        r['right_edge_px']           = ''
        r['top_edge_px']             = ''
        r['retraction_velocity_mm_s'] = ''
        r['rebound_velocity_mm_s']    = ''

    # ── Retraction: right edge during spreading, starting at β_max ───────────
    # cx_px is empty for spreading rows (contact_width method has no centroid).
    # Use last pre-impact cx_px as the approximate lateral center of the lamella.
    pre_impact_cx = None
    for r in rows:
        if r['_phase'] == 'falling' and r['_cx'] is not None:
            pre_impact_cx = r['_cx']  # keep updating → last falling cx

    spreading = [r for r in rows
                 if r['_phase'] == 'spreading'
                 and r['_spread'] is not None]

    retraction_vel = None
    # valid_spread: nonzero, below artifact threshold, sorted by frame order
    valid_spread = [r for r in spreading if 0 < r['_spread'] <= MAX_SPREAD_PX]
    if not valid_spread:
        valid_spread = [r for r in spreading if r['_spread'] > 0]

    if valid_spread and pre_impact_cx is not None:
        # Compute right_edge_px = center_x + half contact width for all spreading rows
        for r in spreading:
            if r['_spread'] is not None:
                r['right_edge_px'] = round(pre_impact_cx + r['_spread'] / 2.0, 3)

        # β_max frame = peak spread_width_px in valid_spread
        beta_max_row = max(valid_spread, key=lambda r: r['_spread'])
        beta_max_idx = valid_spread.index(beta_max_row)
        peak_val     = beta_max_row['_spread']

        # Collect frames from β_max onward where spread is strictly decreasing.
        # Skip plateau at the top (consecutive frames equal to peak_val) before descent.
        # valid_spread already excludes zero-spread frames so no need to check sw==0.
        post_beta   = valid_spread[beta_max_idx:]
        fit_rows    = []
        prev_spread = None
        in_plateau  = True  # skip equal-valued frames at the very top
        for r in post_beta:
            sw = r['_spread']
            if in_plateau:
                if sw >= peak_val:
                    continue  # still at or above peak — skip
                in_plateau  = False
                prev_spread = sw
                fit_rows.append(r)
                continue
            if sw >= prev_spread:
                break   # spread stopped decreasing
            fit_rows.append(r)
            prev_spread = sw
            if len(fit_rows) >= N_FIT:
                break

        if len(fit_rows) >= MIN_FIT:
            t0    = fit_rows[0]['_frame'] * DT
            xs    = [(r['_frame'] * DT - t0) for r in fit_rows]
            ys    = [r['right_edge_px'] for r in fit_rows]
            slope = linreg_slope(xs, ys)  # px/s, negative = contact line retracting
            if slope is not None:
                vel = abs(slope) / px_per_mm
                if vel <= MAX_RET_MM_S:
                    retraction_vel = round(vel, 4)
                    # Stamp on β_max frame row so it's anchored in the timeline
                    beta_max_row['retraction_velocity_mm_s'] = retraction_vel

    # ── Rebound: top edge during rebounding ───────────────────────────────────
    rebounding = [r for r in rows
                  if r['_phase'] == 'rebounding'
                  and r['_cy'] is not None
                  and r['_radius'] is not None]

    rebound_vel = None
    if rebounding:
        # Compute top_edge_px = cy - radius (image coords: smaller = higher)
        for r in rebounding:
            r['top_edge_px'] = round(r['_cy'] - r['_radius'], 3)

        # Use first N_FIT rebounding frames with a valid, upward-moving top edge
        fit_rows = []
        prev_top = None
        for r in rebounding:
            top = r['top_edge_px']
            if top == '':
                continue
            # Only include frames where droplet is still rising (y decreasing)
            if prev_top is not None and float(top) >= prev_top:
                if len(fit_rows) >= 2:
                    break   # stop once droplet stops rising
                # allow a single non-monotone frame as noise, don't break immediately
            fit_rows.append(r)
            prev_top = float(top)
            if len(fit_rows) >= N_FIT:
                break

        if len(fit_rows) >= MIN_FIT:
            t0   = fit_rows[0]['_frame'] * DT
            xs   = [(r['_frame'] * DT - t0) for r in fit_rows]
            ys   = [float(r['top_edge_px']) for r in fit_rows]
            slope = linreg_slope(xs, ys)  # px/s, negative = droplet rising
            if slope is not None:
                vel = abs(slope) / px_per_mm
                if vel <= MAX_REB_MM_S:
                    rebound_vel = round(vel, 4)
                    # Stamp on first rebound fit frame (= liftoff frame)
                    fit_rows[0]['rebound_velocity_mm_s'] = rebound_vel

    # ── Write back CSV with new columns ──────────────────────────────────────
    new_fields = list(orig_fields)
    for col in ('right_edge_px', 'top_edge_px',
                'retraction_velocity_mm_s', 'rebound_velocity_mm_s'):
        if col not in new_fields:
            new_fields.append(col)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=new_fields, extrasaction='ignore')
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    csv_path.write_text(out.getvalue())

    return retraction_vel, rebound_vel


def main():
    features = json.loads(FEATURE_JSON.read_text())

    # Build lookup: (folder, stem) -> feature entry index
    lookup = {}
    for i, e in enumerate(features):
        stem = Path(e['video']).stem
        lookup[(e['folder'], stem)] = i

    total = 0
    found_ret = 0
    found_reb = 0

    for folder_key, v3_dir in V3_DIRS.items():
        if not v3_dir.exists():
            print(f"[skip] {folder_key} — directory not found")
            continue
        px_per_mm = PX_PER_MM[folder_key]
        csvs = sorted(v3_dir.glob("*_timeseries.csv"))
        print(f"\n=== {folder_key} ({len(csvs)} CSVs, px/mm={px_per_mm}) ===")

        for csv_path in csvs:
            stem = csv_path.stem
            if stem.endswith('_timeseries'):
                stem = stem[:-11]

            ret_vel, reb_vel = process_video(csv_path, px_per_mm)
            total += 1

            tag = f"ret={ret_vel or 'None':>10}  reb={reb_vel or 'None':>10}"
            print(f"  {stem:<40} {tag}")

            if ret_vel is not None:
                found_ret += 1
            if reb_vel is not None:
                found_reb += 1

            # Update feature_table.json entry
            key = (folder_key, stem)
            if key in lookup:
                idx = lookup[key]
                features[idx]['retraction_velocity_mm_s'] = ret_vel
                features[idx]['rebound_velocity_mm_s']    = reb_vel
            else:
                print(f"    [warn] no feature_table entry for {folder_key}/{stem}")

    # Save updated feature_table.json
    FEATURE_JSON.write_text(json.dumps(features, indent=2))
    print(f"\nDone. {total} videos — retraction: {found_ret}, rebound: {found_reb}")


if __name__ == "__main__":
    main()
