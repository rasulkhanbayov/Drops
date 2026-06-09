# Droplet Impact Analysis — Full Project Documentation

**Project:** Superhydrophobic surface wetting experiments with cellulose acetate (CA) nanoparticles  
**Last updated:** 2026-06-09  
**Working directory:** `/home/ubuntu/materials/`  
**GitHub:** https://github.com/rasulkhanbayov/Drops

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [Dataset Overview — All Folders](#2-dataset-overview--all-folders)
3. [Camera & Video Properties](#3-camera--video-properties)
4. [Calibration — px/mm & Surface Rows](#4-calibration--pxmm--surface-rows)
5. [Analysis Scripts](#5-analysis-scripts)
6. [ellipse_timeseries_v2.py — Algorithm Detail](#6-ellipse_timeseries_v2py--algorithm-detail)
7. [SAM2 Pipeline — Algorithm Detail](#7-sam2-pipeline--algorithm-detail)
8. [Results Summary & Reliability](#8-results-summary--reliability)
9. [Key Dimensionless Numbers](#9-key-dimensionless-numbers)
10. [Known Issues & Fixes](#10-known-issues--fixes)
11. [AI & Deep Learning Methods](#11-ai--deep-learning-methods)
12. [File Structure](#12-file-structure)

---

## 1. Experimental Setup

| Parameter | Sessions 1–3 (02182026, 03242026, 05052026) | New experiments (05112026, 05122026) |
|---|---|---|
| Drop volume | 4 µL | 4 µL |
| Drop height | 6.5 cm | 13.5 cm (higher velocity) |
| Expected impact velocity | ~1.14 m/s | ~1.63 m/s |
| Surface | Superhydrophobic coating on glass slide | Same |
| Dye | None | Nile Red (tracking only, no effect on physics) |
| Surfactant concentrations | Below AND above CMC | Above CMC only |

**CMC reference values:**
- SDS: 0.225 wt%
- TX-100 (Triton X-100): 0.014 wt%
- Cocoglycoside (CG): 0.005 wt%

---

## 2. Dataset Overview — All Folders

### `02182026/` — CA nanoparticles **with surfactant present** in droplet

| Sample code | Description |
|---|---|
| `cainhcg1–5` | CA + high CG (2× CMC = 0.01 wt%) |
| `cainhsds1–3` | CA + high SDS (2× CMC = 0.45 wt%) |
| `cainhtx1–3` | CA + high TX-100 (2× CMC = 0.028 wt%) |
| `cainlcg1–3` | CA + low CG (0.5× CMC = 0.003 wt%) |
| `cainlsds1–3` | CA + low SDS (0.5× CMC = 0.112 wt%) |
| `cainltx1–3` | CA + low TX-100 (0.5× CMC = 0.007 wt%) |
| `caonly1–3` | CA particles in DI water (no surfactant) |
| `water1–6` | Pure DI water (baseline control) |
| `tx.mp4` | TX-100 surfactant solution only |
| `scale.mp4`, `scale v.mp4` | Calibration rulers |

---

### `03242026_particlesonlypreparedinsurfactant/` — CA particles, **surfactant removed after synthesis**

Particles were washed ×3 by centrifugation. Isolates particle morphology effect only — no residual surfactant in droplet. Same px/mm calibration as 02182026 (same camera setup).

| Sample code | Description |
|---|---|
| `ONLY CA SDS ABOVE CMC 1–3` | Particles made in SDS > CMC, washed |
| `ONLY CA sds less CMC 1–3` | Particles made in SDS < CMC, washed |
| `ONLY CA tx ABOVE CMC 1–4` | Particles made in TX > CMC, washed |
| `ONLY CA tx less CMC 1–3` | Particles made in TX < CMC, washed |
| `ONLY CA cg ABOVE CMC 1–3` | Particles made in CG > CMC, washed |
| `ONLY CA cg less CMC 1–3` | Particles made in CG < CMC, washed |
| `0.001percent cg.mp4` | Pure CG solution (no particles) |
| `0.028percrnt tx.mp4` | Pure TX solution (no particles) |
| `0.45percrnt sds.mp4` | Pure SDS solution (no particles) |
| `ca+TR.mp4` | CA + tracer particles |

---

### `05052026/` — Repeat experiments, different concentrations

| Sample code | Description |
|---|---|
| `0.028tx`, `0.028tx2`, `0.028tx3` | TX-100 surfactant only |
| `0.08cg`, `0.08cg2–4` | CG surfactant only |
| `0.45sds`, `0.45sds2–3` | SDS surfactant only |
| `cainhcg 0.08`, `cainhcg 0.08 b/c/d` | CA + 0.08 wt% CG |
| `cainhg0.02`, `cainhg0.02 2` | CA + 0.02 wt% CG |
| `cainhg0.08 4th` | CA + 0.08 wt% CG (4th replicate) |
| `scale.mp4` | Calibration ruler |

**Note:** `cainhg0.08 4th.mp4` had a bad impact_frame detection (6317 instead of 279) — manually corrected in `feature_table.json`.

---

### `new_experiments/05112026/` — Higher velocity, above-CMC only, Nile Red dye

Drop height: 13.5 cm. Surfactant concentrations are all above CMC.

| Sample code | Description |
|---|---|
| `nr50water`, `nr50water2–4` | Water + Nile Red (control) |
| `water 2`, `water 3` | Water + Nile Red (additional controls) |
| `ca only 2`, `ca only 3` | CA particles only + Nile Red |
| `scale.mp4` | Calibration ruler |

---

### `new_experiments/05122026/` — Day 2, same setup as 05112026

| Sample code | Description |
|---|---|
| `0.028tx1–3` | TX-100 only (0.028 wt%) |
| `0.45sds1–3` | SDS only (0.45 wt%) |
| `cain0.028tx1–3` | CA + TX-100 (0.028 wt%) |
| `cain0.08cg1–3` | CA + CG (0.08 wt%) |
| `cain0.45sds`, `cain0.45sds2–3` | CA + SDS (0.45 wt%) |
| `scale2.mp4` | Calibration ruler |

**Note:** `cain0.028tx3.mp4` is corrupted (0 frames) — skip entirely.

---

## 3. Camera & Video Properties

| Property | Value |
|---|---|
| Resolution | 1280 × 512 px |
| Encoded/playback FPS (from OpenCV) | ~60 fps (MP4 container metadata) |
| **Actual capture FPS** | **2996.766 fps** (confirmed from camera specs) |
| Inter-frame time | 0.3337 ms |
| Format | MP4 |

The MP4 files report ~60 fps — this is the **container playback rate**, not the actual capture rate. All velocity and time calculations must use `FPS_ACTUAL = 2996.766489`.

---

## 4. Calibration — px/mm & Surface Rows

### 4.1 px/mm by folder

| Folder | px/mm | Measured from |
|---|---|---|
| `02182026` | **65.625** | `scale.mp4` — 17 ticks / 16 mm = 65.625 px/mm |
| `03242026` | **65.625** | Same camera setup, same calibration |
| `05052026` | **66.0** | `scale.mp4` in that folder |
| `new_experiments/05112026` | **66.5** | `scale.mp4` — median graduation spacing |
| `new_experiments/05122026` | **56.0** | `scale2.mp4` — camera was positioned ~15% farther |

**Validation:** 4 µL sphere diameter = 2.006 mm. At 65.625 px/mm, HoughCircles gives ~138 px = 2.10 mm ✓ (within 5%).

---

### 4.2 Surface rows — `02182026/`

| Video | Surface row (px) |
|---|---|
| water.mp4 | 433 |
| water2.mp4 | 433 |
| water3.mp4 | 433 |
| water4.mp4 | 417 |
| water5.mp4 | 417 |
| water6.mp4 | 426 |
| cainhcg1.mp4 | 400 |
| cainhcg2.mp4 | 433 |
| cainhcg3.mp4 | 437 |
| cainhcg4.mp4 | 433 |
| cainhcg5.mp4 | 433 |
| cainhsds1.mp4 | 433 |
| cainhsds2.mp4 | 430 |
| cainhsds3.mp4 | 428 |
| cainhtx1.mp4 | 428 |
| cainhtx2.mp4 | 428 |
| cainhtx3.mp4 | 402 |
| cainlcg1.mp4 | 433 |
| cainlcg2.mp4 | 433 |
| cainlcg3.mp4 | 399 |
| cainlsds1.mp4 | 427 |
| cainlsds2.mp4 | 426 |
| cainlsds3.mp4 | 417 |
| cainltx1.mp4 | 433 |
| cainltx2.mp4 | 428 |
| cainltx3.mp4 | 422 |
| caonly1.mp4 | 399 |
| caonly2.mp4 | 405 |
| caonly3.mp4 | 433 |
| tx.mp4 | 417 |

---

### 4.3 Surface rows — `03242026_particlesonlypreparedinsurfactant/`

Surface rows ~40–70 px lower than 02182026 (slide repositioned between sessions).

| Video | Surface row (px) |
|---|---|
| 0.001percent cg.mp4 | 404 |
| 0.028p.mp4 | 404 |
| 0.028percrnt tx.mp4 | 467 |
| 0.45percrnt sds.mp4 | 454 |
| ONLY CA SDS ABOVE CMC.mp4 | 481 |
| ONLY CA SDS ABOVE CMC1.mp4 | 481 |
| ONLY CA SDS ABOVE CMC2.mp4 | 481 |
| ONLY CA cg ABOVE CMC1.mp4 | 485 |
| ONLY CA cg ABOVE CMC2.mp4 | 481 |
| ONLY CA cg ABOVE CMC3.mp4 | 473 |
| ONLY CA cg less CMC1.mp4 | 470 |
| ONLY CA cg less CMC2.mp4 | 465 |
| ONLY CA cg less CMC3.mp4 | 473 |
| ONLY CA sds less CMC1.mp4 | 471 |
| ONLY CA sds less CMC2.mp4 | 470 |
| ONLY CA tx ABOVE CMC1.mp4 | 482 |
| ONLY CA tx ABOVE CMC2.mp4 | 471 |
| ONLY CA tx ABOVE CMC3.mp4 | 470 |
| ONLY CA tx ABOVE CMC4.mp4 | 471 |
| ONLY CA tx less CMC1.mp4 | 465 |
| ONLY CA tx less CMC2.mp4 | 503 |
| ONLY CA tx less CMC3.mp4 | 505 |
| ca+TR.mp4 | 479 |

---

### 4.4 Surface rows — `05052026/`

| Video | Surface row (px) |
|---|---|
| 0.028tx.mp4 | 462 |
| 0.028tx2.mp4 | 470 |
| 0.028tx3.mp4 | 454 |
| 0.08cg.mp4 | 473 |
| 0.08cg2.mp4 | 457 |
| 0.08cg3.mp4 | 454 |
| 0.08cg4.mp4 | 454 |
| 0.45sds.mp4 | 470 |
| 0.45sds2.mp4 | 454 |
| 0.45sds3.mp4 | 454 |
| cainhcg 0.08 b.mp4 | 456 |
| cainhcg 0.08 c.mp4 | 458 |
| cainhcg 0.08 d.mp4 | 454 |
| cainhcg 0.08.mp4 | 454 |
| cainhg0.02 .mp4 | 462 |
| cainhg0.02 2.mp4 | 458 |
| cainhg0.08 4th.mp4 | 449 |

---

### 4.5 Surface rows — `new_experiments/05112026/`

| Video | Surface row (px) |
|---|---|
| ca only 2.mp4 | 304 |
| ca only 3.mp4 | 302 |
| nr50water.mp4 | 356 |
| nr50water2.mp4 | 356 |
| nr50water3.mp4 | 358 |
| nr50water4.mp4 | 307 |
| water 2.mp4 | 305 |
| water 3.mp4 | 304 |

---

### 4.6 Surface rows — `new_experiments/05122026/`

| Video | Surface row (px) |
|---|---|
| 0.028tx1.mp4 | 300 |
| 0.028tx2.mp4 | 303 |
| 0.028tx3.mp4 | 305 |
| 0.45sds1.mp4 | 304 |
| 0.45sds2.mp4 | 312 |
| 0.45sds3.mp4 | 305 |
| cain0.028tx1.mp4 | 302 |
| cain0.028tx2.mp4 | 302 |
| cain0.08cg1.mp4 | 325 |
| cain0.08cg2.mp4 | 303 |
| cain0.08cg3.mp4 | 309 |
| cain0.45sds.mp4 | 304 |
| cain0.45sds2.mp4 | 305 |
| cain0.45sds3.mp4 | 303 |

---

## 5. Analysis Scripts

| Script | Purpose | Output |
|---|---|---|
| `extract_features.py` | Extracts impact_frame, liftoff_frame, D0, β_max, U0 per video using classical CV | `feature_table.json`, `feature_table.csv` |
| `ellipse_timeseries_v2.py` | Full timeseries: pre-impact, spreading, rebound phases with ellipse fitting | Per-video `*_timeseries.csv` + `summary_timeseries_v2.json` |
| `analyze_droplet_sam2.py` | SAM2 video predictor: tracks droplet mask frame-by-frame | Per-video `*_sam2.csv` |
| `run_sam2_03242026.sh` | Batch script for SAM2 on 03242026 folder | `03242026_sam2_results/` |
| `run_sam2_new_experiments.sh` | Batch script for SAM2 on both new_experiment days | `new_experiments/*/sam2_results/` |
| `compare_sam2_opencv.py` | Comparison report between SAM2 and OpenCV results | Markdown report |

### Running the pipeline

```bash
# Feature extraction (needed before timeseries)
python3 extract_features.py --folder 02182026
python3 extract_features.py --folder 03242026
python3 extract_features.py --folder 05052026
python3 extract_features.py --folder 05112026
python3 extract_features.py --folder 05122026

# Timeseries v2 (per folder, saves to dedicated output dir)
python3 ellipse_timeseries_v2.py --folder 02182026 --outdir 02182026_v2_results
python3 ellipse_timeseries_v2.py --folder 03242026 --outdir 03242026_v2_results
python3 ellipse_timeseries_v2.py --folder 05052026 --outdir 05052026_v2_results
python3 ellipse_timeseries_v2.py --folder 05112026 --outdir new_experiments/05112026_v2_results
python3 ellipse_timeseries_v2.py --folder 05122026 --outdir new_experiments/05122026_v2_results

# SAM2 (requires GPU, /data/venv/bin/python for PyTorch)
bash run_sam2_03242026.sh
bash run_sam2_new_experiments.sh
```

**SAM2 Python environment:** Must use `/data/venv/bin/python` (torch 2.5.1+cu121). System `python3` lacks `libtorch_global_deps.so`.  
**SAM2 checkpoint:** `/data/checkpoints/sam2.1_hiera_large.pt`  
**SAM2 model config:** `configs/sam2.1/sam2.1_hiera_l.yaml` (relative to sam2 package, resolved automatically)

---

## 6. ellipse_timeseries_v2.py — Algorithm Detail

### Overview

Three-phase pipeline per video: **pre-impact → spreading → rebound**. All outputs in per-video CSV plus summary JSON.

### Phase 1: Pre-impact (D0 and U0)

**Pass 1 — HoughCircles for D0:**
- Scans backward from `impact_frame` for up to 40 frames
- HoughCircles with radius range 45–110 px, adaptive `param2` (tries 20/15/12/10)
- Takes median of last 5 detections as D0_px
- D0_mm = D0_px / px_per_mm

**Fix 1 — Template D0 cross-check (4 trigger conditions):**  
If any of the following, falls back to template matching for D0:
- `D0_px < 60` — physically too small
- `len(pre_rows) < 3` — too few Hough detections
- `82 < D0_px < 98` — Hough stuck at R_MIN=45 px (diameter = 90–96 px = artifact)
- `D0_px > 155` — Hough latched onto nozzle (~208 px diameter) instead of droplet

Template: synthetic disk (dark body, bright caustic ring) sized to candidate radius. Grid search over radii 28–82 px (step=5), confidence threshold 0.35. Accept override if template is ≥30% larger (small-D case) or ≥30% smaller (nozzle case).

**Pass 2 — Template matching for U0:**
- Slides disk template across search band (above surface, not at frame top) for each pre-impact frame
- Uses `TM_CCOEFF_NORMED`, confidence threshold 0.30
- Falls back to Lucas-Kanade optical flow if <3 template matches
- Further fallback: Theil-Sen velocity from HoughCircles positions

**Fix 2 — Low-confidence template fallback:**
Between standard template (conf ≥0.30) and optical flow, tries `template_lc` at conf ≥0.20.

### Phase 2: Impact frame refinement

Scans ±4 frames around estimated `impact_frame` to find the first frame where contact footprint exceeds 1 mm. Uses background-subtracted diff at `surface_y ± 40 px`.

### Phase 3: Spreading (β_max)

- Scans frames `[impact_frame_ref : liftoff_frame]`
- Background subtracted at surface band → horizontal extent = contact width
- β_max = max(contact_width) / D0_px

**Fix 3 — D0-constrained rebound search:**
HoughCircles radius range for rebound constrained to `[max(30, 0.60×r0), min(130, 1.45×r0)]` to avoid false positives.

**Fix 4 — β outlier flagging:**
Any video with β > 5 is flagged with `"beta_outlier": true` in summary JSON and `⚠β>5` in console output.

### Key constants

```python
FPS_ACTUAL    = 2996.766489   # actual capture rate
PX_PER_MM     = 65.625        # 02182026, 03242026
PX_PER_MM_NEW = 66.0          # 05052026
PX_PER_MM_NEW1 = 66.5         # new_experiments/05112026
PX_PER_MM_NEW2 = 56.0         # new_experiments/05122026
```

### Output CSV fields

```
frame, slice, phase, area, mean, min, max, X, Y, major, minor, angle,
circ, feret, feret_x, feret_y, feret_angle, min_feret,
AR, roundness, solidity, length,
D_px, D_mm, beta, time_ms, Y_dist_px,
dist_travelled_px, velocity_px_s, px_per_mm, velocity_mm_s
```

---

## 7. SAM2 Pipeline — Algorithm Detail

### How it works

1. **Auto-locate reference frame** — OpenCV background subtraction (30-frame background avg) finds first frame where droplet is fully visible. Returns centroid (cx, cy) and area estimate.

2. **Extract frames to disk** — SAM2 video predictor requires JPEG image files. Frames extracted with `frame_step` (1 for short videos, 4 for long videos >5000 frames to avoid GPU OOM).

3. **Point prompt** — centroid (cx, cy) fed as foreground point to `predictor.add_new_points_or_box()` at the reference frame index.

4. **Mask propagation** — `predictor.propagate_in_video()` propagates the mask forward through all frames. SAM2 caches frame embeddings on GPU (memory scales with total frames — hence `frame_step=4` for long videos).

5. **Connected components** — After impact, the propagated mask may split into multiple regions (fragments, satellite droplets). `cv2.connectedComponentsWithStats` separates them, minimum area = max(50, min_area//3).

6. **Reference area** — SAM2 mask area on the reference frame = 100%. All subsequent areas expressed as percentages.

### frame_step policy

| Video length | frame_step | Effective frame rate |
|---|---|---|
| < 5000 frames | 1 | Full resolution |
| ≥ 5000 frames | 4 | Every 4th frame |

Long videos in dataset: `ONLY CA cg ABOVE CMC3` (5520f), `ca+TR` (16377f), `nr50water` (26060f), most 05122026 videos.

### Output CSV fields

```
frame, drop_id, cx, cy, area_px, percentage
```

`frame` is the original video frame number (accounting for frame_step). `drop_id` counts from 1 (largest area = 1). `percentage` = area relative to reference frame area.

### Known failure modes

| Failure | Symptom | Affected videos |
|---|---|---|
| **Rebound exit** | Final cy ≈ 0–4 px (droplet bounces out of top of frame) | Pure water, high-surfactant controls — expected physical behavior |
| **Thin lamella collapse** | Percentage drops to <5% during spreading | Some TX videos — SAM2 loses thin film |
| **Scale video tracked** | Detected ruler ticks as "droplet" | `scale.mp4` videos — exclude from analysis |
| **Short tracking** | <50 frames tracked | Fast-bouncing droplets where reference frame is near end of event |

---

## 8. Results Summary & Reliability

### Completed analyses

| Folder | Videos | SAM2 CSVs | v2 timeseries CSVs | Status |
|---|---|---|---|---|
| `02182026` | 30 | 30 ✅ | 30 ✅ | Complete |
| `03242026` | 23 | 23 ✅ | 23 ✅ | Complete |
| `05052026` | 18 | 18 ✅ | 17 ✅ | Complete (1 skipped) |
| `new_experiments/05112026` | 8 | in progress | in progress | Running |
| `new_experiments/05122026` | 14 | in progress | in progress | Running |

### Reliability assessment (SAM2)

**Reliable (nanoparticle+surfactant combos):** `cainhcg*`, `cainhsds*`, `cainhtx*`, `cainlcg*`, etc. — stable sessile blobs, long tracking, percentages 50–150%.

**Rebound-exit (17 videos total):** Droplet bounces above top of frame. SAM2 tracking stops. Spreading phase data is still valid; post-bounce data is truncated.
- 02182026: `cainhtx2`, `cainltx1`, `cainltx3`, `caonly2`, `water`, `water2`, `water3`, `water4`, `water6`
- 03242026: `ONLY CA cg less CMC1`, `ONLY CA sds less CMC1`, `ONLY CA sds less CMC2`, `ONLY CA tx ABOVE CMC3`, `ONLY CA tx ABOVE CMC4`
- 05052026: `0.028tx3`, `cainhg0.02 2`

**Percentage >200%:** Physically possible (spreading lamella has more area than sphere). Values >300% suggest SAM2 is tracking wetted surface beyond droplet edge.

### v2 timeseries results (02182026, confirmed)

- D0 method split: hough=43/53, template=10/53
- COR computed: 10/53 videos (others rebounded out of frame or no rebound detected)
- β outlier list (β > 5): `cainhtx3` (5.86), `water3` (5.32), `water6` (7.47), `ONLY CA cg ABOVE CMC2` (5.81), `ONLY CA sds less CMC1` (6.36)
- COR range: 0.074–0.960 (mean=0.464, std=0.339)

---

## 9. Key Dimensionless Numbers

| Number | Formula | Physical meaning |
|---|---|---|
| Weber (We) | ρv²D₀/σ | Inertia vs. surface tension |
| Reynolds (Re) | ρvD₀/μ | Inertia vs. viscosity |
| Ohnesorge (Oh) | μ/√(ρσD₀) | Viscosity vs. inertia+surface tension |
| Spreading factor | β_max = D_max/D₀ | Lateral deformation at impact |
| Coefficient of restitution | e = v_rebound/v_impact | Energy retained after bounce |
| Contact time | τ* = τ√(σ/ρR₀³) | Dimensionless contact duration |

For pure water: ρ=998 kg/m³, σ=0.072 N/m, μ=0.001 Pa·s

**Expected ranges at 1.14 m/s, 2 mm droplet (sessions 1–3):**
- We ≈ 36, Re ≈ 2276, Oh ≈ 0.0027
- β_max ≈ 2.0–3.5, e ≈ 0.1–0.8

**Expected ranges at 1.63 m/s, 2 mm droplet (new_experiments):**
- We ≈ 74, Re ≈ 3260, Oh ≈ 0.0027

---

## 10. Known Issues & Fixes

### Confirmed bugs & resolutions

| Issue | Root cause | Fix |
|---|---|---|
| `caonly2` D0 over-estimated | HoughCircles latched onto nozzle (D≈208px) | Fix 1(d): trigger template fallback if D0_px > 155 |
| 05052026 D0 ~3.1mm | Same nozzle detection issue (nozzle r≈104px) | Same fix (D0_px > 155 → accept template if 30% smaller) |
| `cainhg0.08 4th.mp4` wrong impact frame | `find_impact_frame` found false peak in 9000-frame video | Manually corrected: impact=279, liftoff=579 in feature_table.json |
| SAM2 `libtorch_global_deps.so` error | System `python3` lacks PyTorch | Use `/data/venv/bin/python` for all SAM2 runs |
| SAM2 checkpoint not found | Wrong path in batch script | Correct path: `/data/checkpoints/sam2.1_hiera_large.pt` |
| nr50water surface_row detected at 496 | Averaging first 30 frames included post-impact water puddle edges | Fixed: restrict Sobel search to rows 25%–85% of frame height |
| 0.028tx.mp4 and cainhcg 0.08.mp4 impact_frame=1 | `find_impact_frame` triggered on first frame | Known issue; D0=None for these videos |

### Ground truth comparison notes

- ODS ground truth for 02182026 uses **px/mm=62.39**, not 65.625. This produces ~5% systematic offset but all measurements remain within ±10%.
- `water2` D0 in GT measures spreading lamella (not pre-impact sphere) — convention difference, not a bug.
- `caonly2` D0 corrected: GT=1.981mm, v2 reports 1.981mm (template override) ✓

### SAM2 percentage > 100% explanation

`percentage` is area relative to the SAM2 mask at the **reference frame** (first clearly visible droplet frame, before impact). After impact:
- During spreading, the lamella is thin but wide → large pixel area → >100% is physically correct
- Values >100% are expected and valid
- Values >300% suggest SAM2 may be tracking wetted area beyond the droplet edge

---

## 11. AI & Deep Learning Methods

### What has been implemented

| Method | Script | Status |
|---|---|---|
| Classical CV (HoughCircles + optical flow) | `ellipse_timeseries_v2.py` | ✅ Complete, all folders |
| SAM2 video predictor (Meta) | `analyze_droplet_sam2.py` | ✅ Complete, all folders |
| Qwen2.5-VL fine-tuned classifier | `finetune_data/` | ✅ Model trained, benchmark done |
| Claude Sonnet-4.5 zero-shot | `benchmark_eval.py` | ✅ Benchmarked |
| GPT-4o | `benchmark_eval.py` | ✅ Benchmarked |
| Gemini 2.0 Flash | `benchmark_eval.py` | ✅ Benchmarked |

### SAM2 — Segment Anything Model 2

- **Model:** SAM2.1 Hiera Large (`sam2.1_hiera_large.pt`)
- **Mode:** Video predictor (not image predictor)
- **Advantage:** Propagates masks coherently through deformation, splitting, and partial occlusion
- **vs. OpenCV:** SAM2 gives pixel-accurate segmentation; HoughCircles gives circle approximation only

### Potential improvements

1. **Pre-impact D0 as percentage reference** — currently uses the first SAM2 reference frame; using the pre-impact sphere area would make percentages physically meaningful
2. **VLM for reference frame selection** — VLM (Claude, GPT-4o) could identify the optimal prompt frame more reliably than background subtraction
3. **Finer template grid** — current r_step=5 in template D0 search; r_step=2 would improve accuracy for borderline cases like `0.45sds3` (D0 +16% error)

---

## 12. File Structure

```
materials/
├── 02182026/                          # Session 1 videos (30 MP4, not in git)
├── 03242026_particlesonlypreparedinsurfactant/  # Session 2 videos (23 MP4, not in git)
├── 05052026/                          # Session 3 videos (17 MP4, not in git)
├── new_experiments/
│   ├── 05112026/                      # Session 4 day 1 (8 videos + scale)
│   ├── 05122026/                      # Session 4 day 2 (14 videos + scale)
│   └── experiment details.txt
│
├── 02182026_sam2_results/             # 30 SAM2 CSVs
├── 03242026_sam2_results/             # 23 SAM2 CSVs
├── 05052026_sam2_results/             # 18 SAM2 CSVs
├── new_experiments/05112026_sam2_results/  # in progress
├── new_experiments/05122026_sam2_results/  # in progress
│
├── 02182026_v2_results/               # 30 timeseries CSVs
├── 03242026_v2_results/               # 23 timeseries CSVs
├── 05052026_v2_results/               # 17 timeseries CSVs
├── new_experiments/05112026_v2_results/  # in progress
├── new_experiments/05122026_v2_results/  # in progress
│
├── results_drops/                     # Mirror of results (for git — videos excluded)
│
├── feature_table.json                 # Master feature table (all folders)
├── feature_table.csv
├── summary_timeseries_v2.json         # Global v2 summary
│
├── analyze_droplet_sam2.py            # SAM2 analysis script
├── ellipse_timeseries_v2.py           # v2 timeseries pipeline
├── extract_features.py                # Feature extraction
├── compare_sam2_opencv.py             # Comparison tool
├── run_sam2_03242026.sh               # SAM2 batch for 03242026
├── run_sam2_new_experiments.sh        # SAM2 batch for new_experiments
│
├── benchmark/                         # VLM benchmark results
├── finetune_data/                     # Qwen2.5-VL LoRA adapter (weights not in git)
│
├── DOCUMENTATION.md                   # This file
├── README_sam2.md                     # SAM2 setup guide
├── README_opencv.md                   # OpenCV pipeline guide
├── ellipse_timeseries_report.md       # Detailed v2 algorithm report
└── .gitignore                         # Excludes *.mp4, *.safetensors, *.bin, finetune_data/
```

---

*Documentation updated: 2026-06-09*  
*Working directory: `/home/ubuntu/materials/`*
