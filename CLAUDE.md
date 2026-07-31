# CLAUDE.md — Project Knowledge Base
**Project:** Superhydrophobic surface droplet impact — CA nanoparticle & surfactant study  
**Working directory:** `/home/ubuntu/materials/`  
**GitHub:** https://github.com/rasulkhanbayov/Drops  
**Last updated:** 2026-07-31

---

## Quick Reference — Critical Constants

```python
FPS_ACTUAL      = 2996.766489   # real capture rate (OpenCV reports ~60fps — ignore it)
PX_PER_MM       = 65.625        # 02182026 and 03242026 folders
PX_PER_MM_05052 = 66.0          # 05052026 folder
PX_PER_MM_N1    = 66.5          # new_experiments/05112026
PX_PER_MM_N2    = 56.0          # new_experiments/05122026 AND 05172026
D0_expected     = ~2 mm = ~130 px  # 4µL droplet
```

**SAM2 environment** (MUST use this, system python3 lacks libtorch):
```bash
/data/venv/bin/python analyze_droplet_sam2.py ...
```
**SAM2 checkpoint:** `/data/checkpoints/sam2.1_hiera_large.pt`  
**SAM2 config:** `configs/sam2.1/sam2.1_hiera_l.yaml`

**Video locations** (ephemeral — not in git, must be on local disk):
```
/ephemeral/videos/02182026/
/ephemeral/videos/03242026_particlesonlypreparedinsurfactant/
/ephemeral/videos/05052026/
/ephemeral/videos/05172026/
/ephemeral/videos/new_videos/05112026/
/ephemeral/videos/new_videos/05122026/
```

---

## Experimental Setup

| Parameter | Sessions 1–3 (02182026, 03242026, 05052026) | New experiments (05112026, 05122026, 05172026) |
|---|---|---|
| Drop volume | 4 µL | 4 µL |
| Drop height | 6.5 cm → U₀ ≈ 1.14 m/s | 13.5 cm → U₀ ≈ 1.63 m/s |
| Surface | Superhydrophobic glass (contact angle > 150°) | Same |
| Dye | None | Nile Red (tracking only, no physics effect) |
| Surfactant conc. | Below AND above CMC | Above CMC only |

**CMC values:** SDS = 0.225 wt%, TX-100 = 0.014 wt%, CG (cocoglycoside) = 0.005 wt%

**Fluid classes for classification (Task 6):**
- **A** — Pure water (DI, no additives)
- **B** — Surfactant solution only (SDS / TX-100 / CG, no particles)
- **C** — CA nanoparticles + surfactant present in droplet
- **D** — CA nanoparticles, washed (surfactant removed after synthesis)

---

## Dataset Folders

### `02182026/` — CA + surfactant present
30 videos: `cainhcg1–5`, `cainhsds1–3`, `cainhtx1–3`, `cainlcg/sds/tx 1–3`, `caonly1–3`, `water1–6`, `tx.mp4`, `scale.mp4`, `scale v.mp4`  
**Note:** `cainhcg3` and `cainhcg5` have all-zero spread_width — contact width detection fails for these videos.

### `03242026_particlesonlypreparedinsurfactant/` — CA washed (surfactant removed)
23 videos: `ONLY CA SDS/tx/cg ABOVE/less CMC 1–4`, `0.001percent cg`, `0.028percrnt tx`, `0.45percrnt sds`, `ca+TR`

### `05052026/` — Repeat experiments
18 videos: `0.028tx/0.08cg/0.45sds (×3 each)`, `cainhcg 0.08 (a–d)`, `cainhg0.02 (×2)`, `cainhg0.08 4th`, `scale`  
**Note:** `cainhg0.08 4th.mp4` impact_frame manually corrected to 279 in `feature_table.json`.  
**Note:** `0.028tx.mp4` impact_frame=272 (previously wrong at 1), `cainhcg 0.08.mp4` impact_frame=418 — both corrected in `feature_table.json`.

### `new_experiments/05112026/` — Higher velocity, Nile Red
8 videos: `nr50water`, `nr50water2–4`, `water 2`, `water 3`, `ca only 2`, `ca only 3`  
**Note:** `nr50water.mp4` has 26060 frames → `frame_step=4` for SAM2.  
**Note:** 6 of 8 videos are corrupted (0 frames). Only `nr50water4` and `water 3` are readable.

### `new_experiments/05122026/` — Higher velocity, Nile Red, day 2
14 videos: `0.028tx1–3`, `0.45sds1–3`, `cain0.028tx1–3`, `cain0.08cg1–3`, `cain0.45sds/2/3`  
**Note:** `cain0.028tx3.mp4` is CORRUPTED (0 frames) — skip entirely.  
**Note:** Camera was ~15% farther back → px/mm = 56.0 (not 66.5).  
**Note:** Most videos are long (>5000 frames) → `frame_step=4`. `spread_width_px` often saturates at 1279 (frame width) — contact width detection unreliable for these.

### `05172026/` — Additional CA+CG experiments (moved to ephemeral)
2 videos: `cain0.08cg5.mp4`, `cain0.08cg6.mp4`  
**Note:** px/mm = 56.0 (same camera position as 05122026).  
**Note:** Videos live at `/ephemeral/videos/05172026/` — NOT in the new_experiments subfolder.

---

## Calibration — Surface Rows (px from top)

### 02182026
`water/2/3`=433, `water4/5`=417, `water6`=426, `cainhcg1`=400, `cainhcg2/4/5`=433, `cainhcg3`=437, `cainhsds1/2`=430, `cainhsds3`=428, `cainhtx1/2`=428, `cainhtx3`=402, `cainlcg1/2`=433, `cainlcg3`=399, `cainlsds1/2`=427, `cainlsds3`=417, `cainltx1`=433, `cainltx2`=428, `cainltx3`=422, `caonly1`=399, `caonly2`=405, `caonly3`=433, `tx`=417

### 03242026
`0.001percent cg`=404, `0.028p`=404, `0.028percrnt tx`=467, `0.45percrnt sds`=454, `ONLY CA SDS ABOVE CMC`=481, `ONLY CA SDS ABOVE CMC1/2`=481, `ONLY CA cg ABOVE CMC1/2`=481/485, `ONLY CA cg ABOVE CMC3`=473, `ONLY CA cg less CMC1`=470, `ONLY CA cg less CMC2`=465, `ONLY CA cg less CMC3`=473, `ONLY CA sds less CMC1/2`=471/470, `ONLY CA tx ABOVE CMC1`=482, `ONLY CA tx ABOVE CMC2/3`=471/470, `ONLY CA tx ABOVE CMC4`=471, `ONLY CA tx less CMC1`=465, `ONLY CA tx less CMC2`=503, `ONLY CA tx less CMC3`=505, `ca+TR`=479

### 05052026
`0.028tx`=462, `0.028tx2`=470, `0.028tx3`=454, `0.08cg`=473, `0.08cg2`=457, `0.08cg3/4`=454, `0.45sds`=470, `0.45sds2/3`=454, `cainhcg 0.08`=454, `cainhcg 0.08 b`=456, `cainhcg 0.08 c`=458, `cainhcg 0.08 d`=454, `cainhg0.02 `=462, `cainhg0.02 2`=458, `cainhg0.08 4th`=449

### new_experiments/05112026
`ca only 2`=304, `ca only 3`=302, `nr50water`=356, `nr50water2/3`=356/358, `nr50water4`=307, `water 2`=305, `water 3`=304

### new_experiments/05122026
`0.028tx1`=300, `0.028tx2`=303, `0.028tx3`=305, `0.45sds1`=304, `0.45sds2`=312, `0.45sds3`=305, `cain0.028tx1/2`=302, `cain0.08cg1`=325, `cain0.08cg2`=303, `cain0.08cg3`=309, `cain0.45sds`=304, `cain0.45sds2`=305, `cain0.45sds3`=303

### 05172026
`cain0.08cg5`=305, `cain0.08cg6`=304

---

## Key Scripts

| Script | What it does | Output |
|---|---|---|
| `ellipse_timeseries_v2.py` | Main CV pipeline: D0, U0, β_max, spreading, rebound timeseries | Per-video `*_timeseries.csv` + `summary_timeseries_v2.json` |
| `extract_features.py` | Extracts scalar features per video (impact_frame, D0, β_max, U0) | `feature_table.json` |
| `analyze_droplet_sam2.py` | SAM2 video predictor — mask tracking with fallback chain | Per-video `*_sam2.csv` |
| `run_sam2_all_v3.sh` | Batch SAM2 for ALL 6 folders (v3, with px_per_mm + impact/liftoff args) | `results_drops/*_sam2_v3_results/` |
| `extract_retraction_rebound.py` | Extracts retraction_velocity and rebound_velocity from v3 timeseries | Updates timeseries CSVs + `feature_table.json` |
| `fix_sam2_velocity_phase.py` | Patches existing SAM2 CSVs: nulls spreading-phase hough/template, spatial consistency check, recomputes velocity | In-place patch of all SAM2 CSVs |
| `add_phase_to_sam2_csvs.py` | Adds phase column to older SAM2 CSVs from feature_table.json | In-place patch |
| `benchmark_build.py` | Builds benchmark.json from timeseries + VLM frames | `benchmark/benchmark.json` |
| `benchmark_eval.py` | Runs VLM inference (zero-shot) via OpenRouter | `benchmark/results/<model>_results.json` |
| `benchmark_prompted_eval.py` | Runs VLM inference with domain-engineered prompts | `benchmark/results/prompted_<model>_results.json` |
| `task6_cv_classifier.py` | Classical kNN classifier on physical features | Accuracy metrics |
| `add_05172026_features.py` | Adds 05172026 entries to feature_table.json | Updates `feature_table.json` |
| `methodology_segmentation_detection.md` | Journal paper methodology section for SAM2 + CV pipeline | (doc only) |

### Running the full v3 pipeline from scratch

```bash
# Videos must be on disk at /ephemeral/videos/ (not in git)

# 1. Feature extraction (populates feature_table.json)
python3 extract_features.py --folder 02182026
python3 extract_features.py --folder 03242026
python3 extract_features.py --folder 05052026
# 05112026, 05122026, 05172026 entries are already in feature_table.json

# 2. v3 Timeseries (per-video CSVs + summary JSON)
python3 ellipse_timeseries_v2.py --folder 02182026 --outdir results_drops/02182026_v3_results
python3 ellipse_timeseries_v2.py --folder 03242026 --outdir results_drops/03242026_v3_results
python3 ellipse_timeseries_v2.py --folder 05052026 --outdir results_drops/05052026_v3_results
python3 ellipse_timeseries_v2.py --folder 05112026 --outdir results_drops/05112026_v3_results
python3 ellipse_timeseries_v2.py --folder 05122026 --outdir results_drops/05122026_v3_results
python3 ellipse_timeseries_v2.py --folder 05172026 --outdir results_drops/05172026_v3_results

# 3. Retraction + rebound velocity extraction
python3 extract_retraction_rebound.py
# Adds retraction_velocity_mm_s / rebound_velocity_mm_s to timeseries CSVs + feature_table.json

# 4. SAM2 (needs GPU + /data/venv)
bash run_sam2_all_v3.sh

# 5. Patch SAM2 CSVs for coordinate consistency + velocity correctness
python3 fix_sam2_velocity_phase.py

# 6. VLM benchmark (needs OPENROUTER_API_KEY)
export OPENROUTER_API_KEY=sk-or-v1-...
python3 benchmark_eval.py --model anthropic/claude-sonnet-4-5 --resume
python3 benchmark_prompted_eval.py --models anthropic/claude-sonnet-4-5 openai/gpt-4o

# 7. CV classifier
python3 task6_cv_classifier.py
```

---

## frame_step Policy (SAM2)

| Video length | frame_step |
|---|---|
| < 5000 frames | 1 (all frames) |
| ≥ 5000 frames | 4 (every 4th) — avoids GPU OOM |

Long videos: `nr50water` (26060f), `ca+TR` (16377f), `ONLY CA cg ABOVE CMC3` (5520f), most 05122026 videos.

---

## ellipse_timeseries_v2 Algorithm (v3)

**Phase 1 — D0 (pre-impact diameter):**
1. HoughCircles backward from impact_frame, up to 40 frames, radius 45–110 px
2. Outlier rejection: anchor on median cx of 5 detections nearest to impact_frame; reject if >200px away
3. Template fallback triggered if: D0_px < 60, detections < 3, D0_px in 82–98 (stuck at R_MIN), D0_px > 155 (nozzle)
4. Template: synthetic disk (dark body + bright caustic ring), radii 28–82 px, confidence ≥ 0.35

**Phase 2 — U0 (impact velocity):**
1. Template matching (TM_CCOEFF_NORMED, conf ≥ 0.30) across pre-impact frames
2. Fallback: low-confidence template (conf ≥ 0.20)
3. Further fallback: Lucas-Kanade optical flow → Theil-Sen velocity

**Phase 3 — Spreading (β_max):**
- Background median from 30 frames before impact (extended from 5)
- Adaptive threshold: `max(8, min(25, int(peak_diff * 0.4)))` — handles low-contrast 05122026 videos
- β_max = max(contact_width) / D0_px

**Phase 4 — Rebound:**
- HoughCircles constrained to `[max(30, 0.60×r0), min(130, 1.45×r0)]`
- β > 5 flagged as `"beta_outlier": true`

**Per-frame fallback chain (v3 addition):**
1. Pre-computed Hough detection (from scan_pre_impact_d0 / scan_rebound)
2. Template matching fallback
3. Last-known position fallback
- **Spatial consistency check on steps 1 & 2:** reject if displacement from last_known > 150 px — prevents nozzle/edge artifacts at cx≈1190–1270 from corrupting detections when actual droplet is at cx≈390–640

**Timeseries CSV columns:**
`frame, phase, cx_px, cy_px, radius_px, spread_width_px, detection_method, confidence, D_mm, beta, velocity_mm_s, time_ms, dist_travelled_px, px_per_mm, state_change, right_edge_px, top_edge_px, retraction_velocity_mm_s, rebound_velocity_mm_s`

- `right_edge_px`: cx + spread_width/2 during spreading; populated for all spreading rows
- `top_edge_px`: cy - radius during rebounding; populated for all rebounding rows
- `retraction_velocity_mm_s`: stamped on β_max frame only (single value per video)
- `rebound_velocity_mm_s`: stamped on first rebounding fit frame only (single value per video)

---

## SAM2 Pipeline Algorithm (v3)

1. Background subtraction (30-frame avg) → find reference frame with visible droplet → centroid (cx, cy)
2. Point prompt: (cx, cy) as foreground at reference frame
3. Propagate mask forward through all frames (with frame_step)
4. Connected components post-impact (min_area = max(50, min_area//3))
5. **Fallback chain when SAM2 mask is lost:**
   - HoughCircles (spatial consistency check: reject if >12×frame_step px from last_known)
   - Template matching (same spatial check)
   - Last-known position
   - Null (drop_id=0)
6. **Phase assignment:** falling / spreading / rebounding based on impact_frame / liftoff_frame from feature_table.json
7. **Velocity computation:** nulled at detection method transitions (cross-method jumps are artifacts, not motion)

**SAM2 CSV columns:**
`frame, phase, drop_id, cx, cy, area_px, percentage, detection_method, distance_px, velocity_px_per_s, velocity_mm_s`

- `phase`: falling / spreading / rebounding (empty if impact_frame/liftoff_frame unknown)
- `drop_id`: 1 = main droplet, 2+ = satellite fragments during spreading/rebound, 0 = null detection
- `detection_method`: sam2 / hough / template / last_known / null
- `percentage`: area relative to reference frame area (>200% valid during spreading; >300% suggests wetted surface)
- velocity is null across method transitions and during spreading phase (for hough/template)

**Known failure modes:**
- Rebound exit: droplet bounces above frame top — spreading data valid, post-bounce truncated
- Thin lamella collapse: SAM2 loses thin film during TX spreading
- 05122026 long videos: SAM2 tracks multiple fragments as separate drop_ids during spreading — physically valid

---

## Retraction and Rebound Velocity

**Retraction velocity** (`retraction_velocity_mm_s`):
- Defined from β_max frame onward
- Right contact edge = `last_pre_impact_cx + spread_width_px / 2`
- Linear regression on first ≤10 frames of strictly decreasing spread (plateau at peak skipped)
- Zero-spread frames skipped (measurement gaps)
- Caps: ≥3 fit points required, velocity ≤ 2000 mm/s, β_max spread ≤ 600 px (artifact filter)
- 57/94 videos have valid retraction velocity

**Rebound velocity** (`rebound_velocity_mm_s`):
- Top edge = `cy_px - radius_px` (image coords: smaller y = higher)
- Linear regression on first ≤10 monotonically rising rebounding frames
- Caps: ≥3 fit points, velocity ≤ 1500 mm/s (must be < impact velocity)
- 19/94 videos have valid rebound velocity (limited by droplets exiting frame before clean trajectory)

**Why retraction is None for some videos:**
- `cainhcg3/5`: all spread_width_px = 0 (contact detection fails)
- 05112026: no pre-impact cx_px (falling phase detection failed for corrupted videos)
- 05122026 most: spread_width saturates at 1279 px (full frame width) — background subtraction unreliable
- Some others: spread never monotonically decreases ≥3 frames (rebounds or plateaus immediately)

---

## Coordinate System — Known Issue and Fix

**Problem (discovered 2026-07-31):**
In `ellipse_timeseries_v2.py` per-frame fallback loop, HoughCircles finds a **nozzle/edge artifact** at cx≈1190–1270 (right edge of 1280px frame) while template matching correctly finds the **actual droplet** at cx≈390–640. Since methods alternate frame-by-frame, cx jumps ~830px per method switch, producing velocity spikes of thousands of mm/s.

**Root cause:** `hough_detect()` returns the uppermost circle by cy — which is often a fixed reflection/shadow artifact, not the droplet. The pre-impact scan has a separate cx anchor filter (median of 5 nearest-to-impact frames) but the per-frame fallback loop did not.

**Fix in `ellipse_timeseries_v2.py`:** `_spatially_ok(cx, cy)` helper — rejects any Hough or template detection >150 px from last_known position. Falls through to last_known instead. Applied to both Step 1 (pre-computed Hough) and Step 2 (template).

**Same fix in `analyze_droplet_sam2.py`:** `max_disp_per_frame = 12.0 × frame_step` px threshold. Applied to HoughCircles and template matching in the fallback chain. Additionally, hough/template detections are skipped entirely during the spreading phase (flat lamella phase — these methods find artifacts, not the droplet).

**Existing CSVs patched by `fix_sam2_velocity_phase.py`** (run 2026-07-31): 127,153 rows patched across 85 SAM2 v3 CSVs.

**Note:** The v3 timeseries CSVs (from `ellipse_timeseries_v2.py`) still contain bad velocity values for affected videos — those need to be regenerated with the patched script.

---

## VLM Benchmark Results

**Dataset:** 1211 annotated frames, 94 videos, 5 folders  
**Tasks:** Task 6 (fluid classification, 3 frames/video) + Task 1/4 (phase + pixel measurement, 1 frame/call)  
**API:** OpenRouter (`OPENROUTER_API_KEY` env var), OpenAI SDK pointed at `https://openrouter.ai/api/v1`

### Task 6 — Fluid Classification (full benchmark, 94 videos)

| Model | Accuracy | vs Chance (25%) | Bias |
|---|:---:|:---:|---|
| Gemini 2.0 Flash | 21.3% | −3.7 pp | Always predicts D |
| Claude Sonnet 4.5 | 18.1% | −6.9 pp | Mixed A/D |
| GPT-4o-mini | 11.7% | −13.3 pp | Always predicts A |
| Qwen2.5-VL (fine-tuned) | 11.7% | −13.3 pp | Always predicts A |
| GPT-4o | 10.6% | −14.4 pp | Always predicts A |
| **Classical CV (kNN-3)** | **66.0%** | **+41 pp** | — |

### Task 1/4 — Phase + Measurement (full benchmark)

| Model | Phase Acc | cx MAE | radius MAE | spread MAE |
|---|:---:|:---:|:---:|:---:|
| Claude Sonnet 4.5 | **57.6%** | 45.9 px | 20.3 px | **52.4 px** |
| Gemini 2.0 Flash | **57.6%** | 101 px | **14.8 px** | 90.9 px |
| GPT-4o | 50.9% | 120 px | 19.6 px | 145.7 px |
| Qwen2.5-VL (fine-tuned) | 50.0% | **24.5 px** | **7.9 px** | 996 px ✗ |
| GPT-4o-mini | 46.6% | 144 px | 24.7 px | 278.4 px |
| Chance baseline | 33.3% | — | — | — |

### Prompted Evaluation (6 representative videos)

| Model | Task 6 Zero-shot | Task 6 Prompted | Δ |
|---|:---:|:---:|:---:|
| Claude Sonnet 4.5 | 17% | 0% | −17 pp ↓ |
| GPT-4o | 17% | 17% | 0 |

**Finding:** Prompting does NOT help fluid classification. Domain gap is representational (shadowgraphy absent from training data), not instructional.

---

## Known Issues

| Issue | Status | Fix |
|---|---|---|
| SAM2 `libtorch_global_deps.so` error | Permanent | Use `/data/venv/bin/python`, not system python3 |
| SAM2 checkpoint not found | Permanent | Path is `/data/checkpoints/sam2.1_hiera_large.pt` |
| `caonly2` D0 over-estimated (nozzle detected) | Fixed in v3 | Template fallback when D0_px > 155 |
| `cainhg0.08 4th` wrong impact_frame (6317→279) | Fixed | Manually set in `feature_table.json` |
| `0.028tx.mp4` wrong impact_frame (1→272) | Fixed | Updated in `feature_table.json` |
| `cainhcg 0.08.mp4` wrong impact_frame (1→418) | Fixed | Updated in `feature_table.json` |
| `cain0.028tx3.mp4` corrupted | Permanent | Skip — 0 frames, never process |
| `nr50water` surface_row wrong (496) | Fixed | Sobel search restricted to rows 25%–85% |
| β > 5 outliers | Known | Flagged in summary JSON: `cainhtx3`, `water3`, `water6`, `ONLY CA cg ABOVE CMC2`, `ONLY CA sds less CMC1` |
| 05122026 spread_width saturates at 1279 px | Known | Background subtraction unreliable for these long high-contrast videos; retraction velocity not extractable |
| HoughCircles/template cx vs SAM2 cx inconsistency | Fixed | Spatial consistency check (150 px threshold) in both scripts; fix_sam2_velocity_phase.py patches existing CSVs |
| `05172026/` video path | Fixed | Videos moved to `/ephemeral/videos/05172026/`; VIDEOS_NEW3 updated in ellipse_timeseries_v2.py |
| v3 timeseries CSVs still have velocity spikes at hough↔template transitions | Pending | Need to re-run `ellipse_timeseries_v2.py` for affected videos after spatial consistency fix |

---

## Dimensionless Numbers

| Symbol | Formula | Sessions 1–3 | New experiments |
|---|---|---|---|
| We | ρU²D₀/σ | ~36 | ~74 |
| Re | ρUD₀/μ | ~2276 | ~3260 |
| Oh | μ/√(ρσD₀) | 0.0027 | 0.0027 |
| β_max | D_spread/D₀ | 2.0–3.5 | higher (more energy) |
| COR (e) | v_rebound/v_impact | 0.074–0.96 (mean 0.46) | — |

Pure water properties: ρ=998 kg/m³, σ=0.072 N/m, μ=0.001 Pa·s

---

## File Structure (git-tracked)

```
materials/
├── CLAUDE.md                              ← this file
├── DOCUMENTATION.md                       ← full technical docs
├── methodology_segmentation_detection.md  ← journal paper methodology section
├── initial_results_preview.md             ← early results summary
├── supervisor_results_preview.md          ← results doc for supervisor meeting
├── .gitignore                             ← excludes *.mp4, *.png, *.safetensors, *.bin, finetune_data/, github_token.txt
│
├── ellipse_timeseries_v2.py               ← main CV pipeline (v3, with spatial consistency fix)
├── analyze_droplet_sam2.py                ← SAM2 tracker (v3, with phase + fallback chain + spatial check)
├── run_sam2_all_v3.sh                     ← batch SAM2 for all 6 folders
├── extract_retraction_rebound.py          ← retraction + rebound velocity extraction
├── fix_sam2_velocity_phase.py             ← patches existing SAM2 CSVs (spreading null + spatial check)
├── add_phase_to_sam2_csvs.py              ← adds phase column to older SAM2 CSVs
├── add_05172026_features.py               ← adds 05172026 to feature_table.json
├── extract_features.py                    ← scalar feature extraction
├── benchmark_build.py                     ← builds benchmark.json
├── benchmark_eval.py                      ← VLM zero-shot evaluation
├── benchmark_prompted_eval.py             ← VLM prompted evaluation
├── task6_cv_classifier.py                 ← classical kNN classifier
├── compare_sam2_opencv.py                 ← SAM2 vs OpenCV comparison
├── dimensionless_analysis.py              ← We/Re/Oh/COR computation
├── finetune_qwen25vl.py                   ← Qwen2.5-VL LoRA fine-tuning
├── build_finetune_dataset.py              ← builds fine-tune training data
│
├── feature_table.json                     ← master scalar features, all 94 videos, all 6 folders
│                                             includes: impact_frame, liftoff_frame, D0, U0, beta_max,
│                                             retraction_velocity_mm_s, rebound_velocity_mm_s
│
├── results_drops/
│   ├── 02182026_v3_results/               ← 30 timeseries CSVs + summary (✅ committed)
│   ├── 02182026_sam2_v3_results/          ← 28 SAM2 CSVs (cainhcg3/5 missing — no readable data)
│   ├── 03242026_v3_results/               ← 23 timeseries CSVs + summary (✅)
│   ├── 03242026_sam2_v3_results/          ← 23 SAM2 CSVs (✅)
│   ├── 05052026_v3_results/               ← 17 timeseries CSVs + summary (✅)
│   ├── 05052026_sam2_v3_results/          ← 17 SAM2 CSVs (✅)
│   ├── 05112026_v3_results/               ← 9 files (8 stub CSVs + summary; only 2 videos readable)
│   ├── 05112026_sam2_v3_results/          ← 1 SAM2 CSV (nr50water4 only)
│   ├── 05122026_v3_results/               ← 14 timeseries CSVs + summary (✅)
│   ├── 05122026_sam2_v3_results/          ← 14 SAM2 CSVs (✅)
│   ├── 05172026_v3_results/               ← 2 timeseries CSVs + summary (✅)
│   └── 05172026_sam2_v3_results/          ← 2 SAM2 CSVs (✅)
│
├── benchmark/
│   ├── benchmark.json                     ← 1211 annotated frames, all entries + prompts
│   ├── frames/                            ← extracted PNGs (NOT in git — *.png gitignored)
│   └── results/                           ← all VLM result + metric JSONs (✅)
│
└── new_experiments/
    ├── experiment details.txt
    ├── 05112026/                          ← videos NOT in git (corrupted; only 2 of 8 readable)
    └── 05122026/                          ← videos NOT in git
```

---

## Reproducibility Checklist

All result CSVs and JSONs ARE in git and pushed to origin/master. Videos (*.mp4), images (*.png), model weights (*.safetensors, *.bin), and `github_token.txt` are gitignored.

To reproduce from scratch (videos required at `/ephemeral/videos/`):

1. **Features:** `python3 extract_features.py --folder <folder>` for 02182026, 03242026, 05052026 (others already in feature_table.json)
2. **v3 Timeseries:** `python3 ellipse_timeseries_v2.py --folder <folder> --outdir results_drops/<folder>_v3_results` for each folder
3. **Retraction/rebound:** `python3 extract_retraction_rebound.py`
4. **SAM2:** `bash run_sam2_all_v3.sh` (GPU + `/data/venv` required)
5. **Patch SAM2 CSVs:** `python3 fix_sam2_velocity_phase.py`
6. **Benchmark build:** `python3 benchmark_build.py`
7. **VLM eval:** `export OPENROUTER_API_KEY=... && python3 benchmark_eval.py --model <model>`
8. **Prompted eval:** `python3 benchmark_prompted_eval.py --models anthropic/claude-sonnet-4-5 openai/gpt-4o`
9. **CV classifier:** `python3 task6_cv_classifier.py`

**What cannot be reproduced without videos:** All timeseries CSVs, SAM2 CSVs, and benchmark frames. These ARE committed to git so reproduction is only needed if you want to rerun with modified scripts.

**What is safe to delete from this machine:** Everything in `/home/ubuntu/materials/` — all scripts, results, and docs are committed and pushed to GitHub. The `/ephemeral/videos/` folder contains only the raw video files (not in git), which are your original experimental data.
