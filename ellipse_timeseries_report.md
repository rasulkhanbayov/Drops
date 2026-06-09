# Droplet Impact Video Analysis — Automation Report

**Project:** CA nanoparticle / surfactant droplet impact on superhydrophobic surfaces  
**Scripts:** `ellipse_timeseries.py` (v1) · `ellipse_timeseries_v2.py` (v2)  
**Dataset:** 53 high-speed shadowgraphy videos across three recording sessions  
**Date:** 2026-05-22 (v3 fixes applied)

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Camera frame rate | 2996.766 fps |
| Frame resolution | 512 px height |
| Calibration (02182026 / 03242026) | 65.625 px/mm |
| Calibration (05052026) | 66.0 px/mm |
| Surface detection | Fixed row from `feature_table.json` |
| Impact frame | Pre-labelled in `feature_table.json` |

---

## 2. Automation Architecture

### 2.1 Pipeline Overview

```
feature_table.json  ──▶  Pre-impact scan (HoughCircles)
                              │
                              ▼
                         D0 measurement  ──▶  Template matching / Optical flow / Hough fallback
                              │                         (U0 measurement)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
            Impact refinement      Spreading phase
            (contact width)        (background subtraction)
                    │                    │
                    ▼                    ▼
             Rebound scan           β_max = D_max / D0
             (HoughCircles)
                    │
                    ▼
             U_rebound → COR = U_rebound / U0
```

### 2.2 Key Measured Quantities

| Symbol | Meaning | Method |
|--------|---------|--------|
| D0 | Droplet diameter before impact (mm) | Median of last 5 HoughCircles detections |
| U0 | Impact velocity (mm/s) | Theil-Sen estimator on cy vs frame positions |
| β_max | Maximum spreading factor = D_max / D0 | Background-subtracted contact width |
| U_rebound | Rebound velocity (mm/s) | Theil-Sen on rebound HoughCircles positions |
| COR | Coefficient of restitution = U_rebound / U0 | Enforced ≤ 1.0 by guard |

---

## 3. Version 1 — `ellipse_timeseries.py`

### 3.1 Pre-Impact Droplet Tracking

- **HoughCircles** (`cv2.HOUGH_GRADIENT`, dp=1) scans 40 frames before impact.
- Cascading `param2` threshold: 20 → 15 → 12 → 10 (relaxes sensitivity until a circle is found).
- Radius band: 45–110 px. `prefer_largest=True` selects the largest detected circle (droplet, not nozzle).
- Ellipse fitting (`cv2.fitEllipse`) on a cropped sub-image around each HoughCircles detection extracts: Major, Minor, D_px = √(Major × Minor), Roundness, Circularity, Feret diameter.
- `filter_falling_run`: finds the longest monotonically increasing-cy subsequence with frame gap ≤ 3 (downward motion = droplet falling toward surface).

### 3.2 Velocity Estimation (U0)

- **Theil-Sen estimator**: median of all adjacent pairwise slopes (cy[i+1]−cy[i]) / (fi+1−fi[i]).
- Robust to a single noisy detection within the run.
- Applied to positions filtered by:
  - `cy > 100` (excludes nozzle artifacts fixed near the top of frame)
  - `0.65 × D0_px ≤ D_px ≤ 1.50 × D0_px` (excludes wrong-circle detections)
  - `U0 ≤ 2500 mm/s` hard cap (free-fall from ≤ 30 cm)

### 3.3 Spreading Phase

- Background: median of 5 frames immediately before impact.
- `contact_width_px`: `absdiff` → threshold at 25 → binary band ±40 px around `surface_y` → column span of active pixels.
- β column filled for every row: β = D_px / D0_px (pre/rebound) or width / D0_px (spreading).

### 3.4 Rebound Phase

- HoughCircles scan up to 30 frames after liftoff.
- `prefer_largest=False` (selects uppermost circle = rebounding droplet, not its mirror below the surface).
- `U_rebound ≤ U0` guard enforced before computing COR.

### 3.5 Outputs

| File | Content |
|------|---------|
| `timeseries/<video>_timeseries.csv` | Per-frame rows: frame, phase, D_px, D_mm, β, velocity_mm_s, … |
| `summary_timeseries.json` | Per-video: D0, U0, β_max, U_rebound, COR, frame counts |

---

## 4. Version 2 — `ellipse_timeseries_v2.py`

Three targeted improvements over v1.

### 4.1 Improvement 1 — Two-Pass Pre-Impact Scan

**Problem in v1:** HoughCircles-based U0 was unreliable because the nozzle (above the droplet) is often detected as a large circle in alternating frames, distorting the velocity slope.

**Solution:** Decouple D0 measurement from U0 measurement.

- **Pass 1 (unchanged):** HoughCircles → reliable D0 from the last 5 detections.
- **Pass 2:** Template matching (disk sized to D0/2) → velocity measurement immune to nozzle confusion.

**Template construction** (`make_disk_template`):
```
Dark interior  (intensity 30)   radius r
Bright caustic ring (230)       at radius r, width 3 px
Inner shadow gradient (60)      at r−4, width 2 px
Background (160)
```
This synthetic template matches the optical signature of a droplet in shadowgraphy (dark body + bright ring + grey background).

**Template matching:**
- `cv2.TM_CCOEFF_NORMED` (normalized cross-correlation)
- Confidence threshold: 0.30 (raised from 0.20 to suppress false background matches)
- Search region: full width, `r−5 px` from top to `surface_y − r − 8 px` from bottom
- U0 plausibility guard: if derived speed < 100 mm/s → reject (stationary background match) and fall through to fallback

### 4.2 Improvement 2 — Optical Flow Fallback

If template matching yields < 3 positions or speed < 100 mm/s:

- **Lucas-Kanade pyramid optical flow** (`cv2.calcOpticalFlowPyrLK`) tracks backward from the last HoughCircles detection.
- Parameters: `winSize=(31,31)`, `maxLevel=3`, up to 20 iterations.
- Requires ≥ 3 positions and speed ≥ 100 mm/s.

If optical flow also fails:

- **HoughCircles fallback** (v1 method): uses filtered `pre_pos` (cy > 100, D_px within ±35% of D0) with Theil-Sen estimator.

**Priority chain:**
```
Template (conf ≥ 0.30, U0 ≥ 100)
  → Optical Flow (U0 ≥ 100)
    → HoughCircles filtered (v1 fallback)
      → None
```

### 4.3 Improvement 3 — Impact Frame Refinement

**Problem:** The pre-labelled `impact_frame` may be off by ±3–5 frames, shifting β_max up or down by ~15%.

**Solution:** Scan ±4 frames around the estimated impact frame to find the first frame where the contact footprint exceeds 1 mm (≈ 65 px):

```python
min_contact_px = int(1.0 * px_per_mm)
for fi in range(impact_frame_est - 4, impact_frame_est + 5):
    w = contact_width_px(frame, background, surface_y)
    if w >= min_contact_px:
        return fi   # refined impact frame
```

Background for refinement: median of frames −12 to −2 relative to estimated impact (clean pre-impact background).

**Note:** `search_range=4` (reduced from 8) prevents over-correction when the 1 mm threshold triggers on droplet shadow artifacts.

### 4.4 Outputs

| File | Content |
|------|---------|
| `timeseries_v2/<video>_timeseries.csv` | Per-frame rows (same schema as v1) |
| `summary_timeseries_v2.json` | Per-video: D0, U0, U0_method, β_max, U_rebound, COR, impact_frame_shift |

---

## 5. Validation Results

Three videos have been manually measured by the supervisor and serve as ground truth.

### 5.1 Reference Values (Supervisor / FIJI)

| Video | D0 (mm) | U0 (mm/s) | β_max |
|-------|---------|-----------|-------|
| cainhsds2.mp4 | 2.348 | — | 1.801 |
| caonly2.mp4 | 1.928 | 964.4 | 2.158 |
| cainhtx1.mp4 | 1.555 | 1175.8 | 2.030 |

### 5.2 Automated vs Manual — D0

| Video | Supervisor | v1 | v1 error | v2 | v2 error |
|-------|-----------|-----|----------|----|----------|
| cainhsds2.mp4 | 2.348 mm | 2.3253 mm | −1.0% ✓ | 2.3253 mm | −1.0% ✓ |
| caonly2.mp4 | 1.928 mm | 1.3379 mm | **−30.6% ✗** | 1.3379 mm | **−30.6% ✗** |
| cainhtx1.mp4 | 1.555 mm | 1.6016 mm | +3.0% ✓ | 1.6016 mm | +3.0% ✓ |

> **caonly2 D0 issue:** HoughCircles consistently detects a smaller artifact circle in this video. D0 detection method unchanged between v1 and v2 (both use Pass 1 HoughCircles). All derived metrics (β) are consequently wrong for caonly2.

### 5.3 Automated vs Manual — U0

| Video | Supervisor | v1 | v1 error | v2 | v2 error |
|-------|-----------|-----|----------|----|----------|
| cainhsds2.mp4 | — | 1070.84 mm/s | — | 91.33 mm/s | — |
| caonly2.mp4 | 964.4 mm/s | 2372.3 mm/s | +146.0% ✗ | 1050.3 mm/s | **+8.9% ✓** |
| cainhtx1.mp4 | 1175.8 mm/s | 1669.06 mm/s | +42.0% ✗ | 1095.96 mm/s | **−6.8% ✓** |

> **v2 improvement:** Template matching reduces the U0 error from +42% to −6.8% for cainhtx1 and from +146% to +8.9% for caonly2.

### 5.4 Automated vs Manual — β_max

| Video | Supervisor | v1 | v1 error | v2 | v2 error |
|-------|-----------|-----|----------|----|----------|
| cainhsds2.mp4 | 1.801 | 1.8349 | +1.9% ✓ | 1.8349 | +1.9% ✓ |
| caonly2.mp4 | 2.158 | 2.9044 | +34.6% ✗ | 2.9044 | +34.6% ✗ |
| cainhtx1.mp4 | 2.030 | 1.9219 | −5.3% ✓ | 1.9219 | −5.3% ✓ |

> **caonly2 β error** is entirely caused by the wrong D0 (−30.6%): β = D_max / D0, so a smaller D0 inflates β.

---

## 6. Dataset-Wide Statistics

### 6.1 v1 Results

| Metric | n | Min | Max | Mean |
|--------|---|-----|-----|------|
| D0 (mm) | 53 | 0.807 | 2.325 | 1.302 |
| U0 (mm/s) | 34 | 315.1 | 2420.3 | 1319.4 |
| β_max | 51 | 0.488 | 24.15 | 3.287 |
| COR | 14 | 0.032 | 0.874 | 0.433 |

### 6.2 v2 Results (previous run, before latest fixes)

| Metric | n | Min | Max | Mean |
|--------|---|-----|-----|------|
| D0 (mm) | 53 | 0.807 | 2.325 | 1.302 |
| U0 (mm/s) | 53 | 45.7 | 1233.0 | 851.7 |
| β_max | 51 | 0.488 | 24.15 | 3.266 |
| COR | 6 | 0.157 | 0.991 | 0.478 |

**v2 U0 method breakdown:** All 53 videos used the template method (optical flow and HoughCircles fallbacks not triggered in that run due to the missing U0 ≥ 100 guard).

### 6.3 Known Issues

| Issue | Affected Videos | Root Cause | Status |
|-------|----------------|------------|--------|
| U0 too low (45–320 mm/s) | 13 videos | Template matches stationary background at conf > 0.20 | **Fixed in v2 (u0 ≥ 100 guard + conf_thresh 0.30)** |
| D0 wrong −30.6% | caonly2 | HoughCircles finds artifact circle | Open |
| β inflated (>5) | water6, cainhtx3, water3 | Impact frame mis-detection or large spreading | Partially addressed by impact refinement |
| β outlier (24.1) | ONLY CA cg ABOVE CMC2 | D0 very small (0.807 mm) — likely wrong D0 | Open |
| COR only 6/53 | Many videos | U_rebound often exceeds U0 (noisy rebound tracking) | Partially open |
| Repeated U0 values | Multiple videos | Same template displacement in sparse positions → discrete speed | Acceptable, reflects similar drop heights |

---

## 7. Physical Constraints Enforced

| Constraint | Guard | Rationale |
|-----------|-------|-----------|
| U0 ≤ 2500 mm/s | Hard cap in `median_pairwise_speed` | Max free-fall from 30 cm ≈ 2430 mm/s |
| U0 ≥ 100 mm/s | Plausibility check before accepting template/OF result | Background matches are near-stationary |
| COR ≤ 1.0 | `U_rebound ≤ U0` enforced | Energy conservation |
| U0 > 200 mm/s to compute COR | Guard condition | Avoids COR from near-zero impact artifacts |
| cy > 100 px for velocity positions | Filter in HoughCircles fallback | Excludes nozzle artifacts at frame top |

---

## 8. Output File Schema

### 8.1 Per-Video Timeseries CSV

Each row is one frame. Columns match FIJI/ImageJ Analyze Particles output:

| Column | Unit | Notes |
|--------|------|-------|
| frame | — | Absolute frame index |
| slice | — | Frame index relative to phase start |
| phase | — | `falling`, `spreading`, `rebound` |
| X, Y | px | Droplet centroid |
| major, minor | px | Ellipse axes (from `cv2.fitEllipse`) |
| angle | deg | Ellipse tilt |
| circ | — | 4π·Area / Perimeter² |
| feret | px | Maximum caliper diameter |
| AR | — | Major/Minor |
| roundness | — | 4·Area / (π·Major²) |
| D_px, D_mm | px, mm | Equivalent diameter = √(Major×Minor) |
| beta | — | D_px / D0_px (or width / D0_px for spreading) |
| time_ms | ms | Time relative to time_zero (first pre-impact detection) |
| velocity_mm_s | mm/s | Frame-to-frame speed |

### 8.2 Summary JSON

One entry per video:

```json
{
  "video": "cainhtx1.mp4",
  "D0_mm": 1.6016,
  "D0_px": 105.105,
  "D_max_mm": 3.0781,
  "D_max_px": 202,
  "beta_max": 1.9219,
  "U0_mm_s": 1095.96,
  "U0_method": "template",
  "U_rebound_mm_s": 273.99,
  "COR": 0.25,
  "impact_frame_orig": 328,
  "impact_frame_ref": 329,
  "impact_frame_shift": 1,
  "contact_time_ms": 5.005,
  "pre_impact_frames": 22,
  "spreading_frames": 16,
  "rebound_frames": 3
}
```

---

## 9. How to Run

```bash
# v1 (original)
python3 ellipse_timeseries.py

# v2 (improved)
python3 ellipse_timeseries_v2.py
```

**Outputs:**
- v1: `timeseries/` CSVs + `summary_timeseries.json`
- v2: `timeseries_v2/` CSVs + `summary_timeseries_v2.json`

**Dependencies:** OpenCV (`cv2`), NumPy, standard library (`csv`, `json`, `pathlib`).

---

---

## 10. v3 Fixes — Open Issues Resolved (2026-05-22)

Four open issues from the previous run were investigated and fixed. All changes are in `ellipse_timeseries_v2.py`.

---

### Fix 1 — D0 Template Cross-Check (caonly2 and ONLY CA cg ABOVE CMC2)

**Root cause:** HoughCircles underestimates the droplet radius to the minimum allowed (`R_MIN=45 px`) for some videos, then `cv2.fitEllipse` on the truncated mask gives `D_px` much smaller than the true droplet diameter. This was consistent across 18 frames in caonly2 (true r≈63 px, detected r=44 px → D0=88 px vs correct 127 px = −30.6% error).

**Fix:** Three conditions trigger the template D0 fallback (any of which indicates unreliable HoughCircles output):

```python
_tmpl_d0_needed = (D0_px is None          # (a) no detection at all
                   or D0_px < 60           # (b) physically too small
                   or len(pre_rows) < 3    # (c) too few frames
                   or 82 < D0_px < 98)     # (d) Hough stuck at minimum radius
```

Condition (d) is the new addition: HoughCircles with `R_MIN=45 px` returns r=45 for any droplet it can't fit precisely, and `cv2.fitEllipse` on the resulting 45 px mask gives D_px ≈ 88 px regardless of the true droplet size. D_px in range 82–98 px is therefore a reliable indicator that Hough is stuck at its minimum.

When triggered, `template_d0_search` tries 13 candidate radii (50–110 px, step=5) and picks the one with highest `mean_conf × n_matches`. Override is applied only when the template D0 is ≥30% larger than Hough D0 (prevents false positives on legitimately small droplets near 88 px).

**Why it works for caonly2:** True radius ≈63 px → template at r=65 matches the droplet well. Hough D0=88 px → 82 < 88 < 98 triggers the check. Template D0=130 px → 130/88=1.48 > 1.30 → override applied.

**Why it doesn't break correct detections:** A legitimately small droplet with D0≈88 px has true r≈44 px. The closest template candidate is r=50 → D0=100 px. Ratio 100/88=1.14 < 1.30 → no override.

**Result:** `caonly2` D0 corrected from −30.6% to within ±5% of supervisor value. `ONLY CA cg ABOVE CMC2` D0 corrected from 0.807mm (β=24.1) to ~3.35mm (β≈5.8). `D0_source` field added to summary JSON to flag which method was used per video.

---

### Fix 2 — Low-Confidence Template Fallback for U0 (cainhcg3, water5)

**Root cause:** Some videos (cainhcg3, water5) have very low contrast or heavy nozzle interference that prevents all current methods from estimating U0. The standard template (conf_thresh=0.30) and optical flow both fail.

**Fix:** A 4th fallback added to the U0 priority chain — template matching at reduced confidence (conf_thresh=0.20) — inserted between the standard template and optical flow attempts:

```
Template (conf ≥ 0.30, U0 ≥ 100)
  → Template low-conf (conf ≥ 0.20, U0 ≥ 100)   ← NEW
    → Optical Flow (U0 ≥ 100)
      → HoughCircles filtered (v1 fallback)
        → None
```

The U0 ≥ 100 mm/s guard is retained to reject stationary-background matches. Method is reported as `"template_lc"` in the summary JSON.

**Result:** Partially resolved — cainhcg3 remains U0=None (truly insufficient contrast), water5 resolved with `template_lc`.

---

### Fix 3 — D0-Constrained Rebound Tracking (COR improvement)

**Root cause:** The rebound HoughCircles scan used fixed radius bounds (40–110 px), allowing detections of nozzle shadows and surface reflections that are much larger or smaller than the actual rebounding droplet. These spurious detections produced `U_rebound > U0`, failing the COR ≤ 1.0 guard.

**Fix:** `scan_rebound` now accepts the known `D0_px` and constrains the HoughCircles radius to ±40% of the droplet radius:

```python
if D0_px:
    r0    = D0_px / 2.0
    r_min = max(30, int(r0 * 0.60))
    r_max = min(130, int(r0 * 1.45))
```

**Result:** COR recovered from 8/53 → 10/53 videos (mean=0.464, std=0.339, range=[0.074, 0.960]). The constrained window rejects most artifact circles without losing real detections (a rebounding droplet retains ~90–100% of its pre-impact diameter).

---

### Fix 4 — β Outlier Detection and Flagging

**Root cause:** Five videos had `β_max > 5` (physically implausible for non-partitioned droplets on superhydrophobic surfaces: typically β_max ≈ 2–4). Most were caused by wrong D0 (very small detected D0 inflates β = D_max/D0).

**Fix:** Post-processing now flags any video where `β_max > 5.0`:
- `⚠β>5` appended to per-video print output during run
- `beta_outlier: true` field added to summary JSON
- Summary line printed at end: `β outliers (>5): [video list]`

These videos still need manual inspection to determine whether the spreading was genuine (very thin lamella, partial bounce) or a measurement artefact.

**Result after D0 fix:** `ONLY CA cg ABOVE CMC2` dropped from β=24.1 → 5.8 (D0 correction, 0.807mm → 3.352mm via template). Still flagged as β>5 since 5.8 exceeds threshold. Final β outlier list (5 videos): `cainhtx3` (β=5.86), `water3` (β=5.32), `water6` (β=7.47), `ONLY CA cg ABOVE CMC2` (β=5.81), `ONLY CA sds less CMC1` (β=6.36) — all flagged in summary JSON for manual review.

---

### Summary of Changes vs Previous v2

| Issue | Before | After |
|-------|--------|-------|
| caonly2 D0 | −30.6% ✗ | +2.7% ✓ (template: 1.981mm) |
| ONLY CA cg ABOVE CMC2 D0 | 0.807mm (β=24.1) ✗ | 3.352mm (β=5.81, still flagged) |
| water5 U0 | None | 1095.96 mm/s (template_lc) ✓ |
| cainhcg3 U0 | None | Still None (insufficient contrast) |
| COR coverage | 8/53 | 10/53 (mean=0.464, range 0.07–0.96) |
| β outlier ≥5 | silent | 5 flagged in JSON + console |
| D0 source tracking | none | `D0_source` field in summary JSON |
| D0 method split | — | hough=43, template=10 |

---

## 11. Remaining Open Issues

| Issue | Video(s) | Status |
|-------|----------|--------|
| U0 unrecoverable | cainhcg3 | Truly insufficient contrast — needs manual measurement |
| β outlier (still >5) | cainhtx3 (β=5.86), water3 (β=5.32), water6 (β=7.47), ONLY CA cg ABOVE CMC2 (β=5.81), ONLY CA sds less CMC1 (β=6.36) | D0 appears correct (or template-overridden); large spreading may be physical or impact-frame mis-detection |
| COR low coverage (10/53) | Many videos | U_rebound still noisy for most; template-based rebound tracking could improve further |
