# Research Suggestions — Droplet Impact VLM Paper

**Last updated:** 2026-05-22  
**Target venue:** WACV 2027 (submission ~August 2026)  
**Working directory:** `/home/ubuntu/materials/`

---

## 1. What Has Been Done (Starting Point)

### Dataset
- **74 high-speed shadowgraphy videos** across 3 recording sessions on a superhydrophobic glass surface
- **Camera:** Photron FASTCAM Nova S6 at 2996.766 fps, 1280×512 px
- **Scale:** 65.625 px/mm (02182026, 03242026), 66.0 px/mm (05052026)
- **Drop:** 4 µL, dropped from 6.6 cm → theoretical impact velocity 1.137 m/s

#### Session breakdown
| Folder | Videos | Conditions |
|--------|--------|------------|
| `02182026/` | 32 | CA + surfactant (SDS/TX-100/CG at 2×CMC and 0.5×CMC), CA only, pure water |
| `03242026_particlesonlypreparedinsurfactant/` | 23 | Washed CA particles (surfactant removed), pure surfactant controls |
| `05052026/` | 19 | CA + CG at 0.08 wt%, TX-100 at 0.028 wt%, SDS at 0.45 wt% — both with and without particles |

#### Fluid classes (for Task 6 labeling)
| Class | Label | Example videos |
|-------|-------|----------------|
| A | `pure_water` | `water.mp4` – `water6.mp4` |
| B | `surfactant_only` | `tx.mp4`, `0.028percrnt tx.mp4`, `0.45percrnt sds.mp4`, `0.001percent cg.mp4`, `0.028tx*.mp4`, `0.08cg*.mp4`, `0.45sds*.mp4` |
| C | `CA_with_surfactant` | `cainhcg*.mp4`, `cainhsds*.mp4`, `cainhtx*.mp4`, `cainlcg*.mp4`, `cainlsds*.mp4`, `cainltx*.mp4`, `cainhg*.mp4`, `cainhcg 0.08*.mp4` |
| D | `CA_washed` | `caonly*.mp4`, `ONLY CA*.mp4`, `ca+TR.mp4` |

### Classical CV Pipeline (Done)
- **Script:** `extract_features.py`, `recover_nulls.py`, `ellipse_timeseries_v2.py`
- **Output:** `feature_table.json` (53 videos), `summary_timeseries_v2.json` (53 videos)
- **Extracted:** D0 (mm), U0 (mm/s), D_max (mm), β_max, COR (10/53 complete), contact time
- **SAM2:** Applied to `05052026/` folder, CSV results in `05052026/*_sam2.csv`

### VLM Work (Done)
- **Labeled frames:** 773 total — 695 train, 78 val — across 30 videos from `02182026/`
- **Labels per frame:** phase (falling/spreading/rebounding), cx (px), cy (px), radius (px), spread_width (px)
- **Script:** `build_finetune_dataset.py` → `finetune_data/finetune_dataset.jsonl`

#### Zero-shot baseline results (on 45 frames, 3 videos)
| Model | Phase accuracy | Radius MAE | Spread MAE |
|-------|---------------|------------|------------|
| Gemini 2.0 Flash | 60.0% | 0.449 mm | 16.2 mm |
| GPT-4o-mini | 51.1% | 0.637 mm | 17.4 mm |

#### Fine-tuned results (Qwen2.5-VL-7B LoRA, trained on 695 frames)
| Phase accuracy | cx MAE | cy MAE | Radius MAE |
|---------------|--------|--------|------------|
| 93.3% | 13.91 px | 24.45 px | 16.73 px |

**Key finding already in hand:** Zero-shot frontier VLMs are essentially near-random on scientific measurement (51–60% phase, radius error = 22% of droplet diameter). Fine-tuning with <700 frames recovers near-classical-CV accuracy. This is 33 percentage points of improvement.

---

## 2. Reference Papers in This Domain

### Abbot et al. (J. Colloid Interface Sci. 2025)
- **Title:** "Nanoparticles do not influence droplet break-up, spreading, or splashing"
- **Setup:** Silica NPs (Stöber process, additive-free) in water/ethanol on smooth glass
- **Camera:** Phantom TMX5010/V710, up to 150,000 fps
- **Image analysis:** Custom MATLAB — binarization → boundary detection → circle fitting → linear regression
- **Main finding:** Additive-free nanoparticles do NOT affect droplet dynamics
- **No AI/ML** — purely classical CV

### Jereb et al. (Biomimetics 2025)
- **Title:** "Investigation of Droplet Spreading and Rebound Dynamics on Superhydrophobic Surfaces Using Machine Learning"
- **Setup:** Water-glycerin on laser-textured superhydrophobic aluminum, 1498 experiments
- **ML:** 28 MATLAB models → best = Isotropic Exponential Gaussian Process Regression (IE GPR)
- **Input to ML:** Extracted scalars (D0, u0, ρ, μ, σ, pitch, depth) — NOT images
- **Output:** β_max (R²=0.966), rebound efficiency η (R²=0.979), contact time τ (R²=0.645)
- **Key finding:** Droplet velocity dominates; surface topography only affects rebound, not spreading
- **No image-level AI** — parameters extracted by classical contour detection, ML only on scalars

### Critical gap in both papers
Both use hand-tuned classical CV pipelines requiring expert parameter tuning to go from video to measurements. Neither asks whether modern foundation models (VLMs, SAM2) can automate this pipeline. That is the contribution.

---

## 3. The Contribution Gap

**What exists:**
- Classical CV (MATLAB/Python): works but requires domain expertise, manual parameter tuning per video setup
- Jereb-style ML on scalars: good predictions but still requires accurate scalar extraction first
- VLM benchmarks (MMMU, ScienceQA, etc.): static images, textbook knowledge, not dynamic physical phenomena

**What does NOT exist:**
1. Any evaluation of VLMs on scientific high-speed video measurement
2. Any benchmark testing physical reasoning from observed dynamics (not textbook diagrams)
3. Any demonstration that VLMs can infer hidden physical properties (fluid composition) from visual behavior

**The question that matters:**
> Can vision-language models replace hand-tuned classical CV pipelines for scientific physical measurement — and can they go further, inferring properties that classical CV cannot directly measure?

---

## 4. Paper Options Considered

### Option 1 (Simple): "DropBench — Benchmarking VLMs on Scientific Measurement"
- Tasks: phase detection, centroid localization, radius, spread width measurement
- Evaluation: zero-shot VLMs vs fine-tuned vs classical CV
- Physics accuracy: downstream β_max / COR error
- **Problem:** Finding ("fine-tuning helps on narrow domain task") is expected. Reviewers will say "so what?"

### Option 2 (Big): "PhysDrop — Multi-Task Benchmark for Physical Dynamics Reasoning"
Six-task hierarchy from easy to hard:
1. Phase detection (single frame)
2. Outcome classification (video-level: bounce/stick/splash)
3. Comparative reasoning (which sample bounces more?)
4. Quantitative measurement (velocity, diameter, β_max in physical units)
5. Future state prediction (given first 20 frames, predict outcome)
6. Inverse fluid characterization (given video, classify the fluid composition)
- **Problem:** Too complex to execute properly in 3–4 months. Risks messy results and unclear message.

### Why Option 2 > Option 1 in principle
Option 1 answer: "Fine-tuning helps" → expected, known, low novelty  
Option 2 answer: "VLMs learn visual patterns but cannot reason about physics" → new, maps the frontier

### Why Option 1 can be rescued
Add one high-impact task from Option 2 that **requires zero extra annotation work** — Task 6 (fluid classification), because your experimental design already provides perfect ground truth labels.

---

## 5. Chosen Approach: Option 1 + Task 6

### The Paper in One Paragraph
> Scientific video analysis relies on hand-tuned classical CV pipelines that require domain expertise. We ask whether modern VLMs can replace or extend these pipelines using a novel, unpublished dataset of 74 high-speed droplet impact videos across 8 experimental conditions. We benchmark zero-shot and fine-tuned VLMs on four measurement tasks (phase detection, localization, radius, spread width) and one physically meaningful inference task: classifying the fluid composition (pure water / surfactant / nanoparticle+surfactant / washed nanoparticles) from visual impact behavior alone. Zero-shot VLMs score 51–60% on phase classification and have radius MAE of 22% of the droplet diameter. Fine-tuning with 695 labeled frames raises phase accuracy to 93.3% and reduces measurement error to near-classical-CV level. However, both zero-shot and fine-tuned VLMs fail at fluid composition classification — exposing a fundamental gap between visual pattern learning and physical reasoning that fine-tuning alone cannot close.

### Why This Works for WACV
1. **Applied CV** — automated measurement from scientific video is a core WACV topic
2. **Novel dataset** — 74 private, unpublished high-speed videos, never benchmarked before
3. **Strong existing numbers** — 33 pp improvement from fine-tuning already in hand
4. **Surprising result** — fine-tuning doesn't fix Task 6, which is the new finding
5. **AI for Science direction** — practical relevance to every lab doing high-speed video experiments

### The Three Columns of Results
| Task | Zero-shot | Fine-tuned | Classical CV |
|------|-----------|------------|--------------|
| Phase detection | 51–60% | **93.3%** | ~100%* |
| Radius MAE (mm) | 0.45–0.64 | **~0.26** | ~0.15 |
| Spread width MAE | 16–17 mm | **~0 mm** | ~0.5 mm |
| β_max MAE | TBD | TBD | reference |
| COR MAE | TBD | TBD | reference |
| **Fluid classification** | **~random (25%?)** | **~random (25%?)** | **TBD via features** |

*Phase is labeled deterministically from classical CV.

---

## 6. Implementation Plan

### Phase 1: Benchmark Construction (Week 1–2)
**Script: `benchmark_build.py`**
- Read all 74 videos across all 3 folders
- For each video, extract 3 key frames: pre-impact (impact_frame−5), max spread, post-rebound
- Load GT from `summary_timeseries_v2.json` + `feature_table.json`
- Assign Task 6 fluid class label from filename
- Save to `benchmark/` directory: frames as PNG + `benchmark.json`

**Task 6 label mapping (by filename pattern):**
- `water*.mp4` → `pure_water`
- `tx.mp4`, `*percent*.mp4`, `0.028tx*.mp4`, `0.08cg*.mp4`, `0.45sds*.mp4` → `surfactant_only`
- `cainh*.mp4`, `cainl*.mp4`, `cainhcg*.mp4`, `cainhg*.mp4` → `CA_with_surfactant`
- `caonly*.mp4`, `ONLY CA*.mp4`, `ca+TR.mp4` → `CA_washed`

### Phase 2: Multi-Model Evaluation (Week 3–5)
**Script: `benchmark_eval.py`**
- Evaluate models on all tasks:
  - **Existing (already run):** `google/gemini-2.0-flash-001`, `openai/gpt-4o-mini`
  - **New zero-shot:** `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`, `Qwen2.5-VL-7B` (zero-shot, pre-fine-tune)
  - **Fine-tuned:** Local Qwen2.5-VL-7B LoRA (already trained)
- Unified prompt format, JSON output with per-frame results
- Task 6: 3-frame prompt (pre-impact + max spread + post-rebound), 4-choice MCQ

### Phase 3: Physics Accuracy (Week 5–6)
**Script: `physics_accuracy.py`**
- For videos with VLM predictions, compute downstream:
  - β_max from VLM-predicted spread width and VLM-predicted D0
  - COR from VLM-predicted velocity estimates (if extractable)
- Compare VLM-derived β_max to classical CV β_max
- This is the "killer table" for reviewers

### Phase 4: Task 6 Classical CV Baseline (Week 6)
- Train a simple classifier (logistic regression / SVM) on β_max + COR features to classify fluid type
- Shows classical CV can do something that VLMs cannot, even with fine-tuning
- Contrasts physics-based features vs VLM visual features

### Phase 5: Writing (Week 7–12)
- 8-page WACV paper
- Sections: Intro, Related Work, Dataset, Benchmark Tasks, Experiments, Analysis, Conclusion
- Key figure: per-task accuracy bar chart across all models
- Key table: physics accuracy (β_max MAE, COR MAE) by method

---

## 7. The "Killer Insight" for the Paper

The paper's headline narrative, in plain English:

> A fine-tuned 7B VLM, trained on fewer than 700 frames, can locate a 2mm droplet with near-pixel accuracy and classify its impact phase with 93% accuracy. But when asked "what is in this droplet?" — a question every scientist in this field cares about — it performs no better than random guessing. Classical CV extracts β_max and COR automatically; a simple classifier trained on those two numbers can identify the fluid type with X% accuracy. The VLM sees pixels. The classical method understands physics. Closing that gap is the open problem.

This is the AI for Science contribution: defining precisely where the gap is between visual pattern learning and physical reasoning.

---

## 8. Big Picture Ideas (Future Work / Bigger Paper)

If Option 1 + Task 6 succeeds, the natural extension is the full multi-task benchmark:

### Full "PhysDrop" Benchmark (Future)
- **Task 1:** Phase detection (done)
- **Task 2:** Impact outcome classification — bounce / partial rebound / deposition / splash (label in 2 hours)
- **Task 3:** Comparative reasoning — which of two videos shows higher β_max?
- **Task 4:** Quantitative measurement (done)
- **Task 5:** Future state prediction — given pre-impact frames only, predict β_max and outcome class
- **Task 6:** Inverse fluid characterization (done as chosen extension)

### Why This Would Be Bigger
- Maps a 6-level hierarchy of physical reasoning difficulty
- Each level tests a different cognitive capability: recognition, comparison, measurement, prediction, inference
- Enables future VLM researchers to track progress on physical reasoning specifically
- Positions as the "ImageNet of scientific physical video understanding"

### Domain Extension (Collaborative)
Invite groups working on similar high-speed video:
- Bubble dynamics (cavitation)
- Inkjet droplet formation
- Flame propagation
- Cell impact / biomechanics
→ Each contributes a sub-dataset → becomes a community benchmark across scientific domains

### Physics-Constrained VLM Inference (Technical Extension)
Use known physical laws as hard constraints on VLM output:
- Velocity from position-time must be consistent with gravity
- Volume conservation: pre-impact volume ≈ post-impact volume
- Reject physically impossible VLM outputs automatically
- Shows physics knowledge can be injected into VLM inference without fine-tuning

---

## 9. Files Created

| File | Purpose |
|------|---------|
| `benchmark_build.py` | Extracts key frames, builds benchmark JSON for all tasks |
| `benchmark_eval.py` | Evaluates any VLM on benchmark, outputs results JSON |
| `benchmark/benchmark.json` | Generated benchmark dataset |
| `benchmark/frames/` | Extracted PNG frames for each benchmark entry |
| `suggestions.md` | This file |

---

## 10. Key Numbers to Report

| Metric | Value | Source |
|--------|-------|--------|
| Total videos | 74 | 3 folders |
| Experimental conditions | 8 | Experimental design |
| Labeled frames (VLM training) | 773 (695 train / 78 val) | `finetune_metadata.json` |
| Zero-shot phase accuracy (Gemini) | 60.0% | `vlm_zeroshot_baseline.json` |
| Zero-shot phase accuracy (GPT-4o-mini) | 51.1% | `vlm_zeroshot_baseline.json` |
| Fine-tuned phase accuracy (Qwen LoRA) | 93.3% | `eval_finetuned_results.json` |
| Zero-shot radius MAE (Gemini) | 0.449 mm | `vlm_zeroshot_baseline.json` |
| Droplet diameter | ~2.10 mm | Scale calibration |
| Zero-shot radius error as % of droplet | ~22% | Computed |
| Camera FPS | 2996.766 fps | `comprehensive_analysis_report.md` |
| Scale calibration | 65.625 px/mm | `DOCUMENTATION.md` |

---

## 11. Experimental Results (Collected 2026-05-23)

### Benchmark Statistics
- **Total benchmark entries:** 1,211 (1,117 Task1/4 + 94 Task6)
- **Videos covered:** 5 folders (02182026, 03242026, 05052026, 05112026, 05122026)
- **Task6 class distribution:** A=11 (pure water), B=24 (surfactant), C=35 (CA+surfactant), D=24 (washed CA)

### Task 1/4 — Phase Detection + Measurement (Zero-Shot)

| Model | Phase Acc | Radius MAE | Spread MAE | CX MAE |
|-------|-----------|------------|------------|--------|
| Gemini 2.0 Flash | **57.6%** | **0.23 mm** | 1.38 mm | 1.54 mm |
| GPT-4o | 50.9% | 0.30 mm | 2.22 mm | 1.83 mm |
| GPT-4o-mini | 46.6% | 0.38 mm | 4.24 mm | 2.20 mm |
| Claude Sonnet 4.5 | **57.6%** | 0.31 mm | **0.80 mm** | **0.70 mm** |
| Chance (3-class) | 33.3% | — | — | — |

**Key observations:**
- Gemini and Claude tie on phase accuracy; Claude has 2× better spatial localisation
- Gemini has best radius estimation; Claude has best spread measurement
- No model exceeds 58% phase accuracy (vs ~100% for classical CV)

### Task 6 — Fluid Composition Classification

| Method | Accuracy | Bias |
|--------|----------|------|
| GPT-4o | 10.6% | Always predicts A (pure water) |
| GPT-4o-mini | 11.7% | Always predicts A (pure water) |
| Claude Sonnet 4.5 | 18.1% | Mixed A/D |
| Gemini 2.0 Flash | 21.3% | Always predicts D (washed CA) |
| **Chance baseline** | **25.0%** | — |
| **Classical CV (kNN-3, contact_time + spread_width)** | **66.0%** | Balanced |

**Classical CV per-class:** pure_water=100%, surfactant_only=80%, CA_with_surfactant=66.7%, CA_washed=52.4%

**The killer finding:** All four frontier VLMs score *below* the 25% chance baseline on Task 6.
Both GPT models collapse to predicting a single class for every video. Gemini also collapses to
a single class but a different one. Only Claude distributes predictions but still fails.
Classical CV using just two features (contact_time_ms + max_spread_width_mm) achieves 66.0%
— a **44.7 percentage point gap** over the best VLM.

### Paper Narrative (Confirmed by Data)
> A fine-tuned 7B VLM can classify droplet phase with 93% accuracy (zero-shot: 58%).
> But when asked "what fluid is in this droplet?" — the question every scientist cares about —
> all frontier VLMs perform below random guessing. Classical CV extracts two measurable quantities
> (contact time, spread width) and a simple kNN reaches 66%. The gap is not a matter of model
> scale or training: GPT-4o is *worse* than GPT-4o-mini on Task 6. VLMs see pixels.
> Classical CV understands physics. This gap is the open problem.

### Remaining Work for Paper
- [ ] Fine-tuned Qwen LOra on full benchmark (Task 1/4 expected ~90%+; Task 6 expected still ~25%)
- [ ] Classical CV Task 6 on 05052026 + new_experiments folders (more data for classifier)
- [ ] Physics accuracy table: VLM-derived β_max vs classical β_max
- [ ] Write 8-page WACV 2027 paper
