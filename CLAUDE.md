# CLAUDE.md — Project Knowledge Base
**Project:** Superhydrophobic surface droplet impact — CA nanoparticle & surfactant study  
**Working directory:** `/home/ubuntu/materials/`  
**GitHub:** https://github.com/rasulkhanbayov/Drops  
**Last updated:** 2026-06-14

---

## Quick Reference — Critical Constants

```python
FPS_ACTUAL      = 2996.766489   # real capture rate (OpenCV reports ~60fps — ignore it)
PX_PER_MM       = 65.625        # 02182026 and 03242026 folders
PX_PER_MM_05052 = 66.0          # 05052026 folder
PX_PER_MM_N1    = 66.5          # new_experiments/05112026
PX_PER_MM_N2    = 56.0          # new_experiments/05122026 (camera farther back)
D0_expected     = ~2 mm = ~130 px  # 4µL droplet
```

**SAM2 environment** (MUST use this, system python3 lacks libtorch):
```bash
/data/venv/bin/python analyze_droplet_sam2.py ...
```
**SAM2 checkpoint:** `/data/checkpoints/sam2.1_hiera_large.pt`  
**SAM2 config:** `configs/sam2.1/sam2.1_hiera_l.yaml`

---

## Experimental Setup

| Parameter | Sessions 1–3 (02182026, 03242026, 05052026) | New experiments (05112026, 05122026) |
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

### `03242026_particlesonlypreparedinsurfactant/` — CA washed (surfactant removed)
23 videos: `ONLY CA SDS/tx/cg ABOVE/less CMC 1–4`, `0.001percent cg`, `0.028percrnt tx`, `0.45percrnt sds`, `ca+TR`

### `05052026/` — Repeat experiments
18 videos: `0.028tx/0.08cg/0.45sds (×3 each)`, `cainhcg 0.08 (a–d)`, `cainhg0.02 (×2)`, `cainhg0.08 4th`, `scale`  
**Note:** `cainhg0.08 4th.mp4` impact_frame manually corrected to 279 in `feature_table.json`.

### `new_experiments/05112026/` — Higher velocity, Nile Red
8 videos: `nr50water`, `nr50water2–4`, `water 2`, `water 3`, `ca only 2`, `ca only 3`  
**Note:** `nr50water.mp4` has 26060 frames → `frame_step=4` for SAM2.

### `new_experiments/05122026/` — Higher velocity, Nile Red, day 2
14 videos: `0.028tx1–3`, `0.45sds1–3`, `cain0.028tx1–3`, `cain0.08cg1–3`, `cain0.45sds/2/3`  
**Note:** `cain0.028tx3.mp4` is CORRUPTED (0 frames) — skip entirely.  
**Note:** Camera was ~15% farther back → px/mm = 56.0 (not 66.5).

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

---

## Key Scripts

| Script | What it does | Output |
|---|---|---|
| `ellipse_timeseries_v2.py` | Main CV pipeline: D0, U0, β_max, COR, full timeseries | Per-video `*_timeseries.csv` + `summary_timeseries_v2.json` |
| `extract_features.py` | Extracts scalar features per video (impact_frame, D0, β_max, U0) | `feature_table.json` |
| `analyze_droplet_sam2.py` | SAM2 video predictor — mask tracking | Per-video `*_sam2.csv` |
| `benchmark_build.py` | Builds benchmark.json from timeseries + VLM frames | `benchmark/benchmark.json` |
| `benchmark_eval.py` | Runs VLM inference (zero-shot) via OpenRouter | `benchmark/results/<model>_results.json` |
| `benchmark_prompted_eval.py` | Runs VLM inference with domain-engineered prompts | `benchmark/results/prompted_<model>_results.json` |
| `task6_cv_classifier.py` | Classical kNN classifier on physical features | Accuracy metrics |
| `run_sam2_03242026.sh` | Batch SAM2 for 03242026 | `results_drops/03242026_sam2_results/` |
| `run_sam2_new_experiments.sh` | Batch SAM2 for 05112026 + 05122026 | `new_experiments/*/sam2_results/` |

### Running the full pipeline

```bash
# 1. Feature extraction
python3 extract_features.py --folder 02182026
python3 extract_features.py --folder 03242026
# ... etc for each folder

# 2. Timeseries v2
python3 ellipse_timeseries_v2.py --folder 02182026 --outdir results_drops/02182026_v2_results
# ... etc

# 3. SAM2 (needs GPU + /data/venv)
bash run_sam2_03242026.sh
bash run_sam2_new_experiments.sh

# 4. VLM benchmark (needs OPENROUTER_API_KEY)
export OPENROUTER_API_KEY=sk-or-v1-...
python3 benchmark_eval.py --model anthropic/claude-sonnet-4-5 --resume
python3 benchmark_prompted_eval.py --models anthropic/claude-sonnet-4-5 openai/gpt-4o
```

---

## frame_step Policy (SAM2)

| Video length | frame_step |
|---|---|
| < 5000 frames | 1 (all frames) |
| ≥ 5000 frames | 4 (every 4th) — avoids GPU OOM |

Long videos: `nr50water` (26060f), `ca+TR` (16377f), `ONLY CA cg ABOVE CMC3` (5520f), most 05122026 videos.

---

## ellipse_timeseries_v2 Algorithm

**Phase 1 — D0 (pre-impact diameter):**
1. HoughCircles backward from impact_frame, up to 40 frames, radius 45–110 px
2. Template fallback triggered if: D0_px < 60, detections < 3, D0_px in 82–98 (stuck at R_MIN), D0_px > 155 (nozzle)
3. Template: synthetic disk (dark body + bright caustic ring), radii 28–82 px, confidence ≥ 0.35

**Phase 2 — U0 (impact velocity):**
1. Template matching (TM_CCOEFF_NORMED, conf ≥ 0.30) across pre-impact frames
2. Fallback: low-confidence template (conf ≥ 0.20)
3. Further fallback: Lucas-Kanade optical flow → Theil-Sen velocity

**Phase 3 — Spreading (β_max):**
- Background-subtracted diff at surface band → horizontal contact width per frame
- β_max = max(contact_width) / D0_px

**Phase 4 — Rebound:**
- HoughCircles constrained to `[max(30, 0.60×r0), min(130, 1.45×r0)]`
- β > 5 flagged as `"beta_outlier": true`

---

## SAM2 Algorithm

1. Background subtraction (30-frame avg) → find reference frame with visible droplet → centroid (cx, cy)
2. Extract frames to disk (JPEG), apply frame_step
3. Point prompt: (cx, cy) as foreground at reference frame
4. Propagate mask forward through all frames
5. Connected components post-impact (min_area = max(50, min_area//3))
6. Output: `frame, drop_id, cx, cy, area_px, percentage` (percentage relative to reference frame area)

**Known failure modes:**
- Rebound exit: droplet bounces above frame top — spreading data valid, post-bounce truncated (17 videos)
- Thin lamella collapse: SAM2 loses thin film during TX spreading
- percentage > 200%: physically valid during spreading; > 300% suggests tracking wetted surface

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

Domain-engineered prompt: physics context (We, drop height), shadowgraphy image cues (caustic ring, dark blob), per-class decision strategy, calibration constants, surface row.

| Model | Task 6 Zero-shot | Task 6 Prompted | Δ |
|---|:---:|:---:|:---:|
| Claude Sonnet 4.5 | 17% | 0% | −17 pp ↓ |
| GPT-4o | 17% | 17% | 0 |

**Finding:** Prompting does NOT help fluid classification. Domain gap is representational (shadowgraphy absent from training data), not instructional. Prompted prompt shifts bias (Claude → always B) without improving accuracy.

---

## Known Issues

| Issue | Fix |
|---|---|
| SAM2 `libtorch_global_deps.so` error | Use `/data/venv/bin/python`, not system python3 |
| SAM2 checkpoint not found | Path is `/data/checkpoints/sam2.1_hiera_large.pt` |
| `caonly2` D0 over-estimated (nozzle detected) | Fix 1(d) in v2: template fallback when D0_px > 155 |
| `cainhg0.08 4th` wrong impact_frame (6317) | Manually set to 279 in `feature_table.json` |
| `nr50water` surface_row wrong (496) | Fixed: Sobel search restricted to rows 25%–85% of frame height |
| `cain0.028tx3.mp4` corrupted | Skip — 0 frames, never process |
| `0.028tx.mp4` impact_frame=1 | Known; D0=None for this video |
| ODS GT uses px/mm=62.39 | v2 uses 65.625 — ~5% systematic offset, within ±10% |
| β > 5 outliers | Flagged in summary JSON: `cainhtx3`, `water3`, `water6`, `ONLY CA cg ABOVE CMC2`, `ONLY CA sds less CMC1` |

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
├── CLAUDE.md                          ← this file
├── DOCUMENTATION.md                   ← full technical docs
├── .gitignore                         ← excludes *.mp4, *.safetensors, *.bin, *.png, finetune_data/
│
├── ellipse_timeseries_v2.py           ← main CV analysis pipeline
├── analyze_droplet_sam2.py            ← SAM2 tracker
├── extract_features.py                ← scalar feature extraction
├── benchmark_build.py                 ← builds benchmark.json
├── benchmark_eval.py                  ← VLM zero-shot evaluation
├── benchmark_prompted_eval.py         ← VLM prompted evaluation
├── task6_cv_classifier.py             ← classical kNN classifier
├── compare_sam2_opencv.py             ← SAM2 vs OpenCV comparison
├── dimensionless_analysis.py          ← We/Re/Oh/COR computation
├── finetune_qwen25vl.py               ← Qwen2.5-VL LoRA fine-tuning
├── build_finetune_dataset.py          ← builds fine-tune training data
├── run_sam2_03242026.sh               ← SAM2 batch script
├── run_sam2_new_experiments.sh        ← SAM2 batch for new_experiments
│
├── results_drops/                     ← SAM2 + v2 CSVs for all 5 folders
│   ├── 02182026_sam2_results/         ← 30 CSVs ✅
│   ├── 02182026_v2_results/           ← 30 CSVs + summary ✅
│   ├── 03242026_sam2_results/         ← 23 CSVs ✅
│   ├── 03242026_v2_results/           ← 23 CSVs + summary ✅
│   ├── 05052026_sam2_results/         ← 18 CSVs ✅
│   ├── 05052026_v2_results/           ← 17 CSVs + summary ✅
│   ├── 05112026_sam2_results/         ← pending
│   ├── 05112026_v2_results/           ← summary ✅
│   ├── 05122026_sam2_results/         ← pending
│   └── 05122026_v2_results/           ← summary ✅
│
├── benchmark/
│   ├── benchmark.json                 ← 1211 annotated frames, all entries + prompts
│   ├── frames/                        ← extracted PNGs (not in git — *.png gitignored)
│   └── results/                       ← all VLM result + metric JSONs
│
├── feature_table.json                 ← master scalar features, all videos
├── summary_timeseries_v2.json         ← global v2 summary
├── supervisor_results_preview.md      ← results doc for supervisor meeting
│
└── new_experiments/
    ├── experiment details.txt
    ├── 05112026/                      ← videos not in git
    └── 05122026/                      ← videos not in git
```

---

## Reproducibility Checklist

To reproduce from scratch (videos required locally, not in git):

1. **Features:** `python3 extract_features.py` for each folder
2. **Timeseries:** `python3 ellipse_timeseries_v2.py` for each folder
3. **SAM2:** `bash run_sam2_03242026.sh` + `bash run_sam2_new_experiments.sh` (GPU required)
4. **Benchmark build:** `python3 benchmark_build.py` (regenerates benchmark.json + frames)
5. **VLM eval:** `export OPENROUTER_API_KEY=... && python3 benchmark_eval.py --model <model>`
6. **Prompted eval:** `python3 benchmark_prompted_eval.py --models anthropic/claude-sonnet-4-5 openai/gpt-4o`
7. **CV classifier:** `python3 task6_cv_classifier.py`

All result CSVs and JSONs ARE in git. Only videos (*.mp4), images (*.png), and model weights (*.safetensors, *.bin) are excluded.
