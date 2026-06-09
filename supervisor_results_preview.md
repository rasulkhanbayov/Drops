# VLM Benchmark — Initial Results for Supervisor
**Date:** 2026-06-09  
**Dataset:** Superhydrophobic surface droplet impact — 5 recording sessions, ~100 videos, 1211 annotated frames  
**Models evaluated:** Claude Sonnet 4.5, GPT-4o, GPT-4o-mini, Gemini 2.0 Flash, Qwen2.5-VL (fine-tuned)  
**Evaluation:** Zero-shot (no training), frames extracted from high-speed video (2997 fps, 1280×512 px)

---

## Task 6 — Fluid Classification

Each video is shown to the model as **3 frames** (falling → maximum spreading → rebound). The model must identify the fluid type from visual behaviour alone.

**Classes:**
- **A** = Pure water (control)
- **B** = Surfactant solution only (SDS / TX-100 / CG)
- **C** = CA nanoparticles + surfactant in droplet
- **D** = CA nanoparticles, washed (surfactant removed after synthesis)

### Results on 6 representative videos

| Video | Ground Truth | Claude | GPT-4o | Gemini | Qwen (ft) |
|---|---|:---:|:---:|:---:|:---:|
| water2.mp4 | A — Pure water | A ✓ | A ✓ | D ✗ | A ✓ |
| cainhcg1.mp4 | C — CA + surfactant | D ✗ | A ✗ | D ✗ | A ✗ |
| ONLY CA sds less CMC1.mp4 | D — CA washed | A ✗ | A ✗ | D ✓ | A ✗ |
| 0.45percrnt sds.mp4 | B — Surfactant only | A ✗ | A ✗ | D ✗ | A ✗ |
| cainhcg 0.08.mp4 | C — CA + surfactant | A ✗ | A ✗ | D ✗ | A ✗ |
| 0.028tx.mp4 | B — Surfactant only | D ✗ | A ✗ | D ✗ | A ✗ |
| **Accuracy (these 6)** | | **17%** | **17%** | **17%** | **17%** |

### Full benchmark accuracy (94 videos)

| Model | Task 6 Accuracy | Bias |
|---|:---:|---|
| Gemini 2.0 Flash | **21.3%** | Always predicts D |
| Claude Sonnet 4.5 | 18.1% | Mixed A/D |
| GPT-4o-mini | 11.7% | Always predicts A |
| Qwen2.5-VL (fine-tuned) | 11.7% | Always predicts A |
| GPT-4o | 10.6% | Always predicts A |
| **Chance baseline** | **25.0%** | — |
| **Classical CV (kNN-3)** | **66.0%** | — |

> **Key finding:** All VLMs perform **at or below chance** on fluid classification. The best model (Gemini, 21.3%) is 44.7 percentage points behind classical CV using only contact time + spreading width. VLMs cannot visually distinguish CA nanoparticle effects from surfactant effects — the differences are subtle and not present in typical training data.

---

## Task 1/4 — Phase Classification + Droplet Measurement

Each frame is shown individually. The model must:
1. Classify the phase: `falling` / `spreading` / `rebound`
2. Estimate centroid (cx, cy) in pixels
3. Estimate droplet radius (px) or spreading width (px)

Ground truth from `ellipse_timeseries_v2` pipeline (HoughCircles + template matching, validated against manual GT).

### Per-video measurement errors

**water2.mp4** (Pure water, 02182026)

| Model | Phase Acc | cx error | radius error | spread error |
|---|:---:|:---:|:---:|:---:|
| Claude Sonnet 4.5 | **92%** | 18 px | 18 px | 18 px |
| GPT-4o | 83% | 23 px | 20 px | 125 px |
| Gemini 2.0 Flash | 83% | **15 px** | 18 px | 37 px |
| Qwen2.5-VL (ft) | 50% | **2 px** | **9 px** | 1037 px ✗ |

**cainhcg1.mp4** (CA + surfactant, 02182026)

| Model | Phase Acc | cx error | radius error | spread error |
|---|:---:|:---:|:---:|:---:|
| Claude Sonnet 4.5 | **75%** | 11 px | 24 px | **9 px** |
| GPT-4o | 42% | 17 px | 11 px | — |
| Gemini 2.0 Flash | 67% | 11 px | **8 px** | 43 px |
| Qwen2.5-VL (ft) | 42% | **6 px** | **8 px** | 1108 px ✗ |

**ONLY CA sds less CMC1.mp4** (CA washed, 03242026)

| Model | Phase Acc | cx error | radius error | spread error |
|---|:---:|:---:|:---:|:---:|
| Claude Sonnet 4.5 | **58%** | 133 px | 27 px | 178 px |
| GPT-4o | 42% | 183 px | 27 px | — |
| Gemini 2.0 Flash | **58%** | 190 px | **19 px** | 382 px |
| Qwen2.5-VL (ft) | 33% | **118 px** | **14 px** | 886 px ✗ |

**0.45percrnt sds.mp4** (Surfactant only, 03242026)

| Model | Phase Acc | cx error | radius error | spread error |
|---|:---:|:---:|:---:|:---:|
| Claude Sonnet 4.5 | **50%** | **3 px** | **4 px** | 34 px |
| GPT-4o | 42% | 158 px | 19 px | — |
| Gemini 2.0 Flash | **50%** | 59 px | 13 px | 124 px |
| Qwen2.5-VL (ft) | **50%** | 8 px | 7 px | 953 px ✗ |

---

## Full Benchmark Summary — All 1211 Frames

| Model | Task 6 Acc | Phase Acc | cx MAE | radius MAE | spread MAE | Bias |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Claude Sonnet 4.5 | 18.1% | **57.6%** | 45.9 px | 20.3 px | **52.4 px** | Mixed A/D |
| Gemini 2.0 Flash | **21.3%** | **57.6%** | 101 px | **14.8 px** | 90.9 px | Always D |
| GPT-4o | 10.6% | 50.9% | 120 px | 19.6 px | 145.7 px | Always A |
| Qwen2.5-VL (fine-tuned) | 11.7% | 50.0% | **24.5 px** | **7.9 px** | 996 px ✗ | Always A |
| GPT-4o-mini | 11.7% | 46.6% | 144 px | 24.7 px | 278.4 px | Always A |
| Classical CV (kNN-3) | **66.0%** | — | — | — | — | — |
| Chance baseline | 25.0% | — | — | — | — | — |

---

## Key Observations

1. **Fluid classification is beyond zero-shot VLM capability.** All 5 models score at or below chance (25%). The visual differences between CA nanoparticles with surfactant vs. surfactant alone are too subtle for models trained on natural images.

2. **Phase classification is partially solved.** Claude and Gemini reach ~58% phase accuracy (falling / spreading / rebound). This is above chance (33%) but far from reliable for automated extraction.

3. **Centroid localisation varies widely.** Qwen fine-tuned achieves the best centroid accuracy (24.5 px MAE = 0.37 mm) but only because it was fine-tuned on this data. Claude is best among zero-shot models (45.9 px = 0.70 mm). GPT-4o-mini is worst (144 px = 2.20 mm).

4. **Spreading width is the hardest measurement.** Requires understanding the surface contact region — all zero-shot models fail significantly. Qwen overfits catastrophically (996 px MAE = 15 mm — predicts large spread in all phases).

5. **Classical CV outperforms all VLMs by 44.7 pp on fluid classification.** Using only two physical features (contact time + spreading width), a kNN classifier achieves 66% — confirming that the signal exists in the data, VLMs just cannot extract it visually.

6. **The domain gap is real and significant.** Shadowgraphy images (dark droplet, bright caustic ring, high-speed capture) are absent from VLM training data. Quantitative pixel-level reasoning and physical priors (We, Re, surface tension) are not encoded in current VLMs.

---

## Next Steps (for paper)

- [ ] Design domain-engineered prompt (physics context, calibration, measurement instructions)
- [ ] Re-run inference with comprehensive prompt → measure improvement on same 6 videos + full set
- [ ] Analyse which tasks benefit most from prompting vs. which remain fundamentally limited
- [ ] Discuss implications: VLMs as pre-processors (frame selection, outcome labelling) vs. measurement tools

---

*Generated from `benchmark/results/` — 1211 frames across 5 folders, 5 models evaluated*
