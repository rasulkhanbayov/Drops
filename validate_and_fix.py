"""
Accuracy Validation + Spread Width Fix + VLM Comparison
=========================================================
Runs four checks in sequence:

  1. Spread width fix     — background-subtraction contact footprint per video
  2. Scale validation     — re-measures PX_PER_MM from scale.mp4 / scale v.mp4
  3. Velocity validation  — compares measured U0 vs free-fall theory (h=6.6 cm)
                            and checks water-replicate repeatability
  4. VLM vs CV comparison — aligns VLM frame-level predictions against the
                            classical-CV ground truth in eval_finetuned_results.json
                            and vlm_stress_test_results.json

Outputs
-------
  feature_table.json          — updated spread widths + spread factors
  validation_report.json      — all check results
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path

PX_PER_MM  = 65.625
FPS_ACTUAL = 2996.766489
G          = 9.81
DROP_HEIGHT_M = 0.066          # 6.6 cm nominal drop height

VIDEOS_02 = Path("/home/ubuntu/materials/02182026")
VIDEOS_03 = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")
MATERIALS = Path("/home/ubuntu/materials")


# ── helpers ───────────────────────────────────────────────────────────────────
def read_frame(path, fi):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None


def video_path(entry):
    d = VIDEOS_02 if entry["folder"] == "02182026" else VIDEOS_03
    return str(d / entry["video"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPREAD WIDTH FIX — background subtraction
# ══════════════════════════════════════════════════════════════════════════════
def measure_spread_background(vpath, impact, liftoff, surface_y):
    """
    Build a background from 5 frames just before impact (no droplet at surface).
    For each spreading frame subtract background; the contact footprint is the
    changed dark region in a band just above the surface.
    Returns (max_spread_px, spread_per_frame list).
    """
    # Background: median of 5 frames before impact
    bg_frames = []
    for off in range(8, 3, -1):
        fi = impact - off
        if fi >= 0:
            g = read_frame(vpath, fi)
            if g is not None:
                bg_frames.append(g.astype(np.float32))
    if not bg_frames:
        return None, []
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)

    max_w   = None
    spreads = []

    for fi in range(impact + 1, liftoff + 5):   # scan a bit past liftoff too
        g = read_frame(vpath, fi)
        if g is None:
            break

        # Absolute difference from background
        diff = cv2.absdiff(g, bg)

        # Focus on a band around the surface: surface_y-40 to surface_y+10
        band_top = max(0, surface_y - 40)
        band_bot = min(g.shape[0], surface_y + 10)
        band = diff[band_top:band_bot, :]

        # Threshold: only pixels that changed significantly (>25 grey levels)
        _, mask = cv2.threshold(band, 25, 255, cv2.THRESH_BINARY)

        cols = np.where(mask.max(axis=0) > 0)[0]
        if len(cols) >= 5:
            w = float(cols[-1] - cols[0])
            spreads.append((fi, w))
            if max_w is None or w > max_w:
                max_w = w

    return max_w, spreads


def fix_spread_widths(features):
    print("\n── 1. Spread width fix (background subtraction) ──────────────────")
    for f in features:
        vpath = video_path(f)
        impact   = f["impact_frame"]
        liftoff  = f["liftoff_frame"]
        surf     = f["surface_row_px"]

        max_w, _ = measure_spread_background(vpath, impact, liftoff, surf)

        if max_w is not None:
            D0_px = (f["pre_impact_radius_px"] or 0) * 2
            f["max_spread_width_px"]  = round(max_w, 1)
            f["max_spread_width_mm"]  = round(max_w / PX_PER_MM, 4)
            f["max_spread_factor"]    = round(max_w / D0_px, 4) if D0_px > 0 else None
            status = f"spread={max_w:.0f}px  β_max={f['max_spread_factor']}"
        else:
            status = "failed"

        print(f"  {f['video']:<42} {status}")

    return features


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCALE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_scale():
    print("\n── 2. Scale validation ───────────────────────────────────────────")
    results = {}
    for name in ["scale.mp4", "scale v.mp4"]:
        vpath = str(VIDEOS_02 / name)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  {name}: not found")
            continue

        # Read first clear frame
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ruler ticks: look for regularly-spaced vertical dark lines
        # Collapse to a 1-D horizontal profile (mean across vertical band)
        # Use the middle third of the frame (avoid edges)
        h, w = gray.shape
        mid_band = gray[h//3 : 2*h//3, :]
        profile = mid_band.mean(axis=0).astype(np.float32)

        # Smooth and find local minima (tick marks)
        kernel = np.ones(5) / 5
        smoothed = np.convolve(profile, kernel, mode="same")
        threshold = smoothed.mean() - 0.5 * smoothed.std()
        is_dark = smoothed < threshold

        # Find transitions: rising edges = right edge of each dark tick
        ticks = []
        in_dark = False
        tick_start = 0
        for i, d in enumerate(is_dark):
            if d and not in_dark:
                tick_start = i
                in_dark = True
            elif not d and in_dark:
                ticks.append((tick_start + i) // 2)
                in_dark = False

        if len(ticks) >= 2:
            spacings = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
            # Filter to spacings within 20% of median
            med = np.median(spacings)
            valid = [s for s in spacings if abs(s - med) / med < 0.2]
            if valid:
                px_per_tick = float(np.mean(valid))
                # Assume ticks are 1mm apart (standard ruler)
                measured_px_per_mm = px_per_tick
                results[name] = {
                    "ticks_found": len(ticks),
                    "mean_tick_spacing_px": round(px_per_tick, 2),
                    "assumed_tick_mm": 1.0,
                    "measured_px_per_mm": round(measured_px_per_mm, 3),
                    "used_px_per_mm": PX_PER_MM,
                    "error_pct": round(abs(measured_px_per_mm - PX_PER_MM) / PX_PER_MM * 100, 2),
                }
                print(f"  {name}: {len(ticks)} ticks, spacing={px_per_tick:.1f}px "
                      f"→ {measured_px_per_mm:.2f} px/mm  (used={PX_PER_MM}, "
                      f"err={results[name]['error_pct']}%)")
            else:
                print(f"  {name}: ticks found but spacings inconsistent")
        else:
            print(f"  {name}: <2 ticks detected ({len(ticks)} found)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. VELOCITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_velocity(features):
    print("\n── 3. Velocity validation ────────────────────────────────────────")
    expected_u0 = math.sqrt(2 * G * DROP_HEIGHT_M) * 1000   # mm/s

    print(f"  Free-fall expected: U0 = √(2×9.81×0.066) = {expected_u0:.1f} mm/s  "
          f"({expected_u0/1000:.3f} m/s)")

    measured = [(f["video"], f["impact_velocity_mm_per_s"])
                for f in features if f["impact_velocity_mm_per_s"] is not None
                and f["impact_velocity_mm_per_s"] > 200]

    u0_vals = [u for _, u in measured]
    if u0_vals:
        mean_u0 = np.mean(u0_vals)
        std_u0  = np.std(u0_vals)
        err_pct = abs(mean_u0 - expected_u0) / expected_u0 * 100
        print(f"  Measured mean: {mean_u0:.1f} ± {std_u0:.1f} mm/s  "
              f"(error vs theory: {err_pct:.1f}%)")

    # Water replicates repeatability
    water_vids = [f for f in features if f["video"].startswith("water")
                  and f["impact_velocity_mm_per_s"] and f["impact_velocity_mm_per_s"] > 200]
    print(f"\n  Water replicates (same fluid, same height):")
    print(f"  {'Video':<12} {'D0 (mm)':>10} {'U0 (mm/s)':>12} {'U0 err vs theory':>18}")
    water_stats = []
    for f in water_vids:
        d0 = f["pre_impact_diameter_mm"]
        u0 = f["impact_velocity_mm_per_s"]
        err = abs(u0 - expected_u0) / expected_u0 * 100 if u0 else None
        water_stats.append({"video": f["video"], "D0_mm": d0, "U0_mm_s": u0, "U0_err_pct": round(err, 1) if err else None})
        err_str = f"{err:.1f}%" if err else "—"
        print(f"  {f['video']:<12} {str(d0):>10} {str(u0):>12} {err_str:>18}")

    d0_waters = [f["pre_impact_diameter_mm"] for f in water_vids if f["pre_impact_diameter_mm"]]
    if d0_waters:
        print(f"\n  Water D0: mean={np.mean(d0_waters):.3f}mm  "
              f"std={np.std(d0_waters):.3f}mm  "
              f"CV={np.std(d0_waters)/np.mean(d0_waters)*100:.1f}%")

    return {
        "expected_U0_mm_s": round(expected_u0, 2),
        "measured_mean_U0_mm_s": round(mean_u0, 2) if u0_vals else None,
        "measured_std_U0_mm_s": round(std_u0, 2) if u0_vals else None,
        "error_vs_theory_pct": round(err_pct, 2) if u0_vals else None,
        "water_replicates": water_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. VLM vs CV COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def compare_vlm_cv():
    print("\n── 4. VLM vs Classical CV comparison ────────────────────────────")

    ft = json.loads((MATERIALS / "feature_table.json").read_text())
    ft_map = {f["video"]: f for f in ft}

    # Load both VLM result files
    zeroshot  = json.loads((MATERIALS / "vlm_stress_test_results.json").read_text())
    finetuned = json.loads((MATERIALS / "eval_finetuned_results.json").read_text())

    zs_frames = zeroshot["frames"]    # list of frame dicts per model
    ft_frames = finetuned["frames"]   # list of frame dicts for fine-tuned model

    # Helper: compute MAE ignoring None
    def mae(pairs):
        vals = [abs(a - b) for a, b in pairs if a is not None and b is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    # ── Per-video: compare VLM D0 estimate vs feature_table D0 ──────────────
    # VLM radius on falling frames → diameter → compare with feature D0
    print("\n  Per-video D0 comparison (VLM falling-frame radius × 2 vs feature D0):")
    print(f"  {'Video':<20} {'CV D0(mm)':>10} {'Gemini D0(mm)':>14} {'GPT D0(mm)':>12} {'FT D0(mm)':>10}")

    video_comparison = []
    for video in ["water.mp4", "caonly1.mp4", "cainhsds1.mp4"]:
        cv_d0  = ft_map.get(video, {}).get("pre_impact_diameter_mm")
        cv_u0  = ft_map.get(video, {}).get("impact_velocity_mm_per_s")

        # Gemini and GPT from zeroshot (falling phase frames)
        gemini_radii, gpt_radii, ft_radii = [], [], []
        for rec in zs_frames:
            if rec["video"] != video or rec["phase"] != "falling":
                continue
            r = rec.get("vlm_radius")
            if r:
                if rec["model"] == "google/gemini-2.0-flash-001":
                    gemini_radii.append(r * 2 / PX_PER_MM)
                elif rec["model"] == "openai/gpt-4o-mini":
                    gpt_radii.append(r * 2 / PX_PER_MM)

        for rec in ft_frames:
            if rec["video"] != video or rec["phase"] != "falling":
                continue
            r = rec.get("vlm_radius")
            if r:
                ft_radii.append(r * 2 / PX_PER_MM)

        gemini_d0 = round(float(np.mean(gemini_radii)), 3) if gemini_radii else None
        gpt_d0    = round(float(np.mean(gpt_radii)),    3) if gpt_radii    else None
        ft_d0     = round(float(np.mean(ft_radii)),     3) if ft_radii     else None

        print(f"  {video:<20} {str(cv_d0):>10} {str(gemini_d0):>14} {str(gpt_d0):>12} {str(ft_d0):>10}")
        video_comparison.append({
            "video": video,
            "cv_D0_mm": cv_d0, "cv_U0_mm_s": cv_u0,
            "gemini_D0_mm": gemini_d0, "gpt_D0_mm": gpt_d0, "ft_D0_mm": ft_d0,
        })

    # ── Frame-level: centroid, radius, spread errors ─────────────────────────
    print("\n  Frame-level MAE — Gemini vs CV ground truth:")
    gemini_cx = mae([(r["vlm_cx"], r["gt_cx"]) for r in zs_frames
                     if r["model"] == "google/gemini-2.0-flash-001"])
    gemini_r  = mae([(r["vlm_radius"], r["gt_radius"]) for r in zs_frames
                     if r["model"] == "google/gemini-2.0-flash-001"])
    gemini_sw = mae([(r["vlm_spread_width"], r["gt_spread_width"]) for r in zs_frames
                     if r["model"] == "google/gemini-2.0-flash-001"])
    print(f"    cx MAE={gemini_cx}px  radius MAE={gemini_r}px  spread MAE={gemini_sw}px")

    print("  Frame-level MAE — Fine-tuned vs CV ground truth:")
    ft_cx = mae([(r["vlm_cx"], r["gt_cx"]) for r in ft_frames])
    ft_r  = mae([(r["vlm_radius"], r["gt_radius"]) for r in ft_frames])
    ft_sw = mae([(r["vlm_spread_width"], r["gt_spread_width"]) for r in ft_frames])
    print(f"    cx MAE={ft_cx}px  radius MAE={ft_r}px  spread MAE={ft_sw}px")

    # ── Phase accuracy ────────────────────────────────────────────────────────
    print("\n  Phase classification accuracy:")
    for model_key, label in [
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("openai/gpt-4o-mini",          "GPT-4o-mini      "),
    ]:
        correct = sum(1 for r in zs_frames
                      if r["model"] == model_key and r.get("phase_correct"))
        total   = sum(1 for r in zs_frames if r["model"] == model_key
                      and r.get("phase_correct") is not None)
        print(f"    {label}: {correct}/{total} = {correct/total*100:.1f}%")

    ft_correct = sum(1 for r in ft_frames if r.get("phase_correct"))
    ft_total   = sum(1 for r in ft_frames if r.get("phase_correct") is not None)
    print(f"    Fine-tuned Qwen  : {ft_correct}/{ft_total} = {ft_correct/ft_total*100:.1f}%")

    # ── Spread width: VLM vs corrected CV ────────────────────────────────────
    print("\n  Spread width comparison (mm) — fine-tuned VLM vs feature_table:")
    print(f"  {'Video':<20} {'CV spread(mm)':>14} {'FT VLM spread(mm)':>18} {'Diff':>8}")
    for video in ["water.mp4", "caonly1.mp4", "cainhsds1.mp4"]:
        cv_sw = ft_map.get(video, {}).get("max_spread_width_mm")
        ft_sw_vals = [r["vlm_spread_width"] for r in ft_frames
                      if r["video"] == video and r["phase"] == "spreading"
                      and r.get("vlm_spread_width") is not None]
        ft_sw_mm = round(float(np.mean(ft_sw_vals)) / PX_PER_MM, 3) if ft_sw_vals else None
        diff = round(abs((cv_sw or 0) - (ft_sw_mm or 0)), 3) if cv_sw and ft_sw_mm else None
        print(f"  {video:<20} {str(cv_sw):>14} {str(ft_sw_mm):>18} {str(diff):>8}")

    return {
        "video_level_D0_comparison": video_comparison,
        "frame_level_MAE": {
            "gemini": {"cx_px": gemini_cx, "radius_px": gemini_r, "spread_px": gemini_sw},
            "finetuned": {"cx_px": ft_cx, "radius_px": ft_r, "spread_px": ft_sw},
        },
        "phase_accuracy_pct": {
            "gemini": round(correct/total*100, 1) if total else None,
            "finetuned": round(ft_correct/ft_total*100, 1),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    features = json.loads((MATERIALS / "feature_table.json").read_text())

    # 1. Fix spread widths
    features = fix_spread_widths(features)
    (MATERIALS / "feature_table.json").write_text(json.dumps(features, indent=2))
    print(f"  → feature_table.json updated")

    # 2. Scale validation
    scale_results = validate_scale()

    # 3. Velocity validation
    vel_results = validate_velocity(features)

    # 4. VLM vs CV comparison
    vlm_results = compare_vlm_cv()

    # ── Summary spread stats ──────────────────────────────────────────────────
    spreads = [f["max_spread_factor"] for f in features if f.get("max_spread_factor")]
    if spreads:
        print(f"\n  β_max after fix: min={min(spreads):.2f}  max={max(spreads):.2f}  "
              f"mean={np.mean(spreads):.2f}  (n={len(spreads)})")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "scale_validation": scale_results,
        "velocity_validation": vel_results,
        "vlm_vs_cv_comparison": vlm_results,
        "beta_max_summary": {
            "min": round(min(spreads), 3) if spreads else None,
            "max": round(max(spreads), 3) if spreads else None,
            "mean": round(float(np.mean(spreads)), 3) if spreads else None,
            "n": len(spreads),
        },
    }
    out = MATERIALS / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nValidation report → {out}")


if __name__ == "__main__":
    main()
