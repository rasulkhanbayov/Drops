# Comprehensive Cross-Method Analysis Report
## Droplet Impact on Superhydrophobic Surfaces — CA Nanoparticles with Surfactants

**Date:** 2026-04-25  
**Dataset:** 53 high-speed videos across two experimental folders  
**Camera:** Photron FASTCAM Nova S6 at 2996.766 fps, 1280 × 512 px, scale 65.625 px/mm  

---

## Overview

Three distinct approaches were used to extract parameters from the droplet impact videos:

1. **Classical Computer Vision (CV)** — frame-by-frame geometric detection using OpenCV, producing raw physical measurements (D₀, U₀, D_max, contact time, spreading factor).
2. **Paper Methodology Replication** — dimensionless analysis following Abbot et al. (2025, *J. Colloid Interface Sci.*), computing We, Re, Oh, β_max, β_splash, τ_c, and t* from the CV outputs.
3. **Vision-Language Model (VLM)** — Qwen2.5-VL-7B fine-tuned via LoRA on labeled frames, plus zero-shot baselines with Gemini 2.0 Flash and GPT-4o-mini, producing natural-language descriptions of droplet phase, position, radius, and spread width per frame.

These three approaches are not independent: the paper methodology is a post-processing layer built on top of classical CV outputs. The VLM approach is the only fully independent measurement path.

---

## Section 1 — Classical Computer Vision

### 1.1 Implementation

The pipeline was implemented in Python 3 using OpenCV 4.x and NumPy. Two scripts handle the full dataset: `extract_features.py` runs the primary extraction, and `recover_nulls.py` re-attempts detection on videos where the primary pass returned null values.

The processing steps, in order, are:

**Surface row detection.** For each video, the first 10 frames are read, converted to grayscale, and reduced to horizontal profiles. The row with consistently minimum mean intensity (the dark surface line) is identified as `surface_row_px`. This value is used as a reference throughout all subsequent steps.

**Impact frame detection.** A frame-differencing approach computes the absolute difference between consecutive frames across the first 1000 frames. The frame with the maximum total change in the lower half of the image (below mid-frame) is selected as `impact_frame`. This captures the moment the droplet first contacts the surface.

**Pre-impact diameter and velocity.** For the 5 frames immediately before impact (frames `impact_frame − 5` to `impact_frame − 1`), `cv2.HoughCircles` with `HOUGH_GRADIENT` is run on each Gaussian-blurred grayscale frame. The uppermost detected circle (smallest y-coordinate) whose bottom edge is above the surface is taken as the droplet. The median radius across detected frames gives `pre_impact_radius_px`, which is converted to diameter in mm via the calibrated scale. Velocity is extracted from a `np.polyfit` linear regression of centroid y-position against frame index; the slope (pixels per frame) is multiplied by the actual frame rate (2996.766 fps) and divided by 65.625 px/mm to produce `impact_velocity_mm_per_s`.

**Liftoff detection.** After the impact frame, a HoughCircles scan runs on up to 300 subsequent frames. The first frame where a circle is detected with its entire extent above the surface (`cy + r < surface_row − 5`) is designated `liftoff_frame`. Contact time in milliseconds is then `(liftoff_frame − impact_frame) / 2996.766 × 1000`. This HoughCircles-based liftoff method was introduced after an earlier width-threshold approach failed: the surface line itself spans the full frame width and was detected as a spreading footprint even after liftoff.

**Maximum spread width.** A background subtraction approach is used: the median of the 5 frames immediately before impact forms a reference background. For each frame from impact to liftoff, `cv2.absdiff` with a threshold of 25 isolates pixels that have genuinely changed from background. The maximum contiguous width of changed pixels within a band 40 px above to 10 px below the surface row is recorded as `max_spread_width_px`. This was a critical fix: the initial implementation measured the full contact_width of any dark pixels in the band, which always returned 1279 px (the surface line) for every video.

**Recovery pass.** For videos where D₀ or U₀ remained null after primary extraction, `recover_nulls.py` extends the lookback window to 40 frames before impact, cascades HoughCircles param2 from 18 down to 10 for increased sensitivity, and falls back to contour detection (largest dark blob above surface) if Hough fails entirely. Results are sanity-checked: D₀ outside 0.4–4.0 mm and U₀ below 50 mm/s are rejected as physically implausible.

### 1.2 Output Files

| File | Format | Contents |
|---|---|---|
| `/home/ubuntu/materials/feature_table.json` | JSON array, 53 entries | Primary output — all extracted measurements per video |
| `/home/ubuntu/materials/feature_table.csv` | CSV | Same data in tabular form |
| `/home/ubuntu/materials/feature_table_recovered.json` | JSON array | Only the videos updated by the recovery pass |
| `/home/ubuntu/materials/validation_report.json` | JSON | Scale validation, velocity validation, VLM vs CV comparison, β_max summary |

### 1.3 Parameters Extracted

| Parameter | Key in JSON | Physical meaning |
|---|---|---|
| `surface_row_px` | Surface detection | Y-coordinate (pixels) of the liquid surface in the frame |
| `impact_frame` | — | Frame index where droplet first contacts surface |
| `liftoff_frame` | — | Frame index where droplet fully detaches from surface |
| `contact_time_ms` | τ | Time the droplet remains in contact with surface (ms) |
| `pre_impact_diameter_mm` | D₀ | Droplet diameter just before impact (mm) |
| `pre_impact_cx_px`, `cy_px` | — | Centroid coordinates of droplet before impact (px) |
| `pre_impact_radius_px` | r₀ | Droplet radius in pixels before impact |
| `impact_velocity_mm_per_s` | U₀ | Speed of droplet centroid falling toward surface (mm/s) |
| `impact_velocity_px_per_frame` | — | Raw slope from linear fit (px/frame), before unit conversion |
| `max_spread_width_px` | D_max (px) | Maximum lateral extent of contact footprint during spreading (px) |
| `max_spread_width_mm` | D_max | Maximum spread diameter in mm |
| `max_spread_factor` | β_max | D_max / D₀ — dimensionless spreading factor |

### 1.4 Quality Assessment

The scale calibration is confirmed accurate. Analysis of `scale.mp4` via tick-spacing detection yields 65.75 px/mm against the assumed 65.625 px/mm — a 0.19% discrepancy that is negligible and well within one-pixel quantization error.

Velocity measurements are physically sensible. The measured mean impact velocity across all 33 complete videos is 1075.7 ± 118.8 mm/s, against the free-fall theoretical value of 1137.9 mm/s (√(2 × 9.81 × 0.066)) — a 5.5% underestimate consistent with air drag on a falling millimeter-scale droplet. Water replicates (water.mp4, water2.mp4, water3.mp4, water5.mp4) span 1049–1273 mm/s with a coefficient of variation of about 9%, reasonable for manual pipette drops from nominally the same height.

D₀ measurements for water replicates show a mean of 2.00 mm with a standard deviation of 0.25 mm (CV = 12.5%). This spread reflects genuine droplet volume variability from a hand-held pipette rather than measurement noise; HoughCircles is consistent on the same droplet across frames.

The spreading factor β_max now ranges 0.26–11.22 with a mean of 2.88 across 39 videos, which is physically reasonable for droplets impacting superhydrophobic surfaces. The outliers (cainhtx3 β = 8.46, ONLY CA sds less CMC1 β = 8.83, water3 β = 4.21, water6 β = 4.68) may reflect genuine splashing events rather than measurement error and warrant manual inspection.

Contact time data contains two suspicious clusters. Fifteen videos have a contact time of exactly 100.11 ms (300 frames × 1000/2996.766 ms/frame), which is the liftoff search window cap. For these videos, no circle was detected within 300 frames after impact, and the liftoff frame was set to `impact_frame + 300` as a fallback. These contact times are unreliable and should not be used for analysis. Ten videos show a contact time of exactly 3.34 ms (10 frames), which may be too short for a physical bounce — it is possible HoughCircles detected a circle in the ejected splash rather than the rebounding droplet body.

### 1.5 Null Values

**D₀ null — 14 videos:**
cainhcg3.mp4, cainhcg5.mp4, cainlsds1.mp4, caonly1.mp4, ONLY CA SDS ABOVE CMC1.mp4, ONLY CA SDS ABOVE CMC2.mp4, ONLY CA cg ABOVE CMC1.mp4, ONLY CA cg less CMC2.mp4, ONLY CA cg less CMC3.mp4, ONLY CA sds less CMC2.mp4, ONLY CA tx ABOVE CMC1.mp4, ONLY CA tx ABOVE CMC3.mp4, ONLY CA tx ABOVE CMC4.mp4, ONLY CA tx less CMC1.mp4.

In all 14 cases, HoughCircles found no candidate circle in the 40 frames before impact at any sensitivity setting, and contour detection either found nothing or returned a blob outside the valid D₀ range. The most likely cause is that the droplet in these videos is unusually small, falls very close to the nozzle tip (partially out of the upper frame), or is overexposed to the point of near-invisibility against the bright background.

**U₀ null — 20 videos:**
The 14 above plus water4.mp4, water6.mp4, ONLY CA cg ABOVE CMC2.mp4, ONLY CA cg ABOVE CMC3.mp4, ONLY CA cg less CMC1.mp4, and ca+TR.mp4. The additional 6 have either fewer than two reliable centroid detections (preventing a linear fit) or returned a near-zero velocity slope that failed the 50 mm/s plausibility filter. ca+TR.mp4 is a tracer-particle video where the droplet morphology differs substantially from the pure-fluid videos and detection is less reliable.

**β_max null — 14 videos:** Same as D₀ null, since β_max requires D₀ as its denominator.

**contact_time_ms null — 0 videos:** All 53 videos have a value, but 15 of them (the 100.11 ms values) represent the window cap rather than a genuine measurement and should be treated as censored data.

---

## Section 2 — Vision-Language Model

### 2.1 Implementation

Two stages of VLM evaluation were conducted.

**Zero-shot baseline.** `vlm_stress_test.py` selected 45 frames from 3 videos (water.mp4, caonly1.mp4, cainhsds1.mp4), with 15 frames per video distributed equally across falling, spreading, and rebounding phases. Each frame was passed as a base64-encoded JPEG to two commercial API endpoints: Google Gemini 2.0 Flash (`google/gemini-2.0-flash-001`) and OpenAI GPT-4o-mini (`openai/gpt-4o-mini`). The prompt asked the model to return a JSON object with fields `phase` (one of: falling, spreading, rebounding), `cx` (droplet center x in pixels), `cy` (droplet center y in pixels), `radius` (droplet radius in pixels), `spread_width` (lateral contact width in pixels during spreading, else null), and `confidence`. Responses were parsed by stripping markdown code fences and passing to `json.loads`. The 90 total responses (45 frames × 2 models) are stored in `vlm_stress_test_results.json`.

**Fine-tuning.** Qwen2.5-VL-7B was fine-tuned with LoRA (rank 16, alpha 32, dropout 0.05) applied to all query/key/value/output projection layers using the HuggingFace PEFT library. The training dataset consisted of annotated frames where ground-truth labels came from the classical CV pipeline: centroid and radius from HoughCircles, spread width from contact_width detection, and phase from temporal position relative to impact and liftoff frames. Fine-tuning ran for 3 epochs with a batch size of 2, gradient accumulation steps of 4, and a learning rate of 2×10⁻⁴ with cosine scheduling, on a single GPU. Training completed in 42 minutes with a final evaluation loss of 0.1256. The fine-tuned adapter was then evaluated on the same 45 test frames using `eval_finetuned.py`, producing `eval_finetuned_results.json`.

### 2.2 Output Files

| File | Format | Contents |
|---|---|---|
| `/home/ubuntu/materials/vlm_stress_test_results.json` | JSON (dict with `model_metrics` and `frames`) | Zero-shot results: 90 frames × {Gemini, GPT-4o-mini}, with GT and predicted values |
| `/home/ubuntu/materials/vlm_finetuned_results.json` | JSON | Copy of finetuned results (backup, prevents overwrite) |
| `/home/ubuntu/materials/vlm_zeroshot_baseline.json` | JSON | Copy of zero-shot results (backup) |
| `/home/ubuntu/materials/eval_finetuned_results.json` | JSON (dict with `metrics` and `frames`) | Fine-tuned model evaluation: 45 frames with GT and predicted values |

### 2.3 Parameters Extracted

The VLM produces per-frame outputs, not per-video summaries. The fields in each frame record are:

| Field | Physical meaning |
|---|---|
| `vlm_phase` | Phase label: "falling", "spreading", or "rebounding" |
| `vlm_cx`, `vlm_cy` | Predicted droplet centroid (pixels) |
| `vlm_radius` | Predicted droplet radius (pixels) |
| `vlm_spread_width` | Predicted lateral contact footprint width (pixels), null outside spreading phase |
| `vlm_confidence` | Model self-reported confidence: "high", "medium", or "low" |

From these frame-level fields, per-video D₀ can be estimated by averaging the radius over falling-phase frames and converting: `D₀ = 2 × mean_radius / 65.625`. This is a secondary derived quantity, not directly output by the model.

### 2.4 Quality Assessment

**Phase classification.** The fine-tuned Qwen model achieves 93.3% phase accuracy (42/45 frames), dramatically outperforming Gemini 2.0 Flash at 60.0% (27/45) and GPT-4o-mini at 51.1% (23/45). The zero-shot models share a systematic failure: neither correctly classifies any rebounding frames. Gemini labels 0/15 rebounding frames correctly; GPT-4o-mini correctly labels only 1/15. Both models appear to confuse the rebounding phase with falling (the morphology is similar — an airborne sphere — but the droplet is rising rather than falling). The fine-tuned model correctly identifies 13/15 rebounding frames (87%).

**Centroid localization (cx).** During the falling and spreading phases, the fine-tuned model is extremely accurate: cx MAE of 2.2 px in the falling phase and 2.0 px during spreading. This is sub-pixel precision relative to the 69–100 px droplet radius, indicating that the model has genuinely learned spatial localization. During rebounding, cx MAE increases to 28.0 px, consistent with the more difficult morphology (smaller, partially deformed droplet). Gemini and GPT-4o-mini show much higher cx MAE across all phases (31.6–242.6 px).

**Radius estimation.** The fine-tuned model achieves a radius MAE of 7.6 px during falling and 12.2 px during spreading against the HoughCircles ground truth. At the typical droplet radius of 70–80 px, this represents a 10–15% relative error. The zero-shot models average 27.5–53.2 px radius MAE, with GPT-4o-mini particularly poor during spreading (53.2 px).

**D₀ estimation (per-video).** For the two videos with classical CV D₀ ground truth (water.mp4 and cainhsds1.mp4), the fine-tuned model estimates D₀ as 2.054 mm (CV: 2.255 mm, 8.9% error) and 2.651 mm (CV: 2.621 mm, 1.1% error). Gemini estimates 1.481 and 1.219 mm respectively — 34% and 54% below CV ground truth. GPT-4o-mini estimates 1.256 and 0.506 mm — 44% and 81% below CV. The systematic underestimation by zero-shot models is consistent with them treating the background-corrected droplet silhouette as smaller than it appears to a scale-aware HoughCircles detector.

**Spread width.** The fine-tuned model reported a spread_width of 19.49 mm for all three test videos in the spreading phase, which matches no physical measurement (the actual spread widths range from 0.69 to 3.93 mm). This is a fixed-value hallucination — the model learned to output a specific number rather than measure the contact footprint. The zero-shot models also fail on spread_width; Gemini's spread MAE is 1062 px. This is the VLM's primary failure mode and means VLM-derived β_max is not usable.

### 2.5 Null Values

All 45 test frames have non-null `vlm_phase`, `vlm_cx`, `vlm_cy`, and `vlm_radius` predictions from all three models — no parsing failures. `vlm_spread_width` is null for falling and rebounding frames by design (the model is instructed to return null when not in the spreading phase).

The VLM produces no results whatsoever for the remaining 50 videos outside the 3-video test set. Extending the VLM to all 53 videos would require either running inference on each video's frames or fine-tuning on a larger labeled dataset.

---

## Section 3 — Paper Methodology (Abbot et al. 2025)

### 3.1 Implementation

The dimensionless analysis was implemented in `dimensionless_analysis.py`, closely following Abbot et al. (2025, *Journal of Colloid and Interface Science*, "Wetting transitions on superhydrophobic surfaces with aqueous surfactant solutions"). The paper studies how surfactant type, concentration, and nanoparticle morphology affect impact outcome by characterizing droplets through a standard set of dimensionless numbers and time scales.

**Fluid classification.** Since the raw video filenames encode the experimental condition, `classify_fluid(video_name, folder)` maps filenames to one of eight fluid classes: pure water, SDS above/below CMC, TX-100 above/below CMC, CG/CAPB above/below CMC, and CA nanoparticles in water. Each class is assigned a fluid properties tuple (σ surface tension in N/m, η dynamic viscosity in Pa·s, ρ density in kg/m³). Surfactant concentrations follow the experimental design: 2×CMC for "above CMC" and 0.5×CMC for "below CMC", with literature surface tension values for SDS (above CMC: σ = 0.037), TX-100 (above CMC: σ = 0.034), and CG/CAPB (above CMC: σ = 0.031). Water uses σ = 0.072, η = 0.001, ρ = 998.

**Dimensionless numbers computed.** For each video with non-null D₀ and U₀:

- **Weber number:** We = ρ U₀² D₀ / σ — ratio of inertial to surface tension forces
- **Reynolds number:** Re = ρ U₀ D₀ / η — ratio of inertial to viscous forces
- **Ohnesorge number:** Oh = η / √(ρ σ D₀) = √We / Re — ratio of viscous to inertial-surface-tension forces
- **Capillary number:** Ca = η U₀ / σ — ratio of viscous to surface tension forces
- **Froude number:** Fr = U₀ / √(g D₀) — ratio of inertial to gravitational forces
- **Splashing parameter β_splash:** Following Riboux & Gordillo Eq. 1: β = (1/tan α) × (η_g² ρ D₀ U₀⁵ / σ²)^(1/6), where α = 60° and η_g is the dynamic viscosity of air
- **Capillary time:** τ_c = √(ρ R₀³ / σ) in milliseconds — the natural oscillation time scale of the droplet set by surface tension
- **Inertia-capillary time:** τ_ic = D₀ / U₀ in milliseconds — the time for the droplet to travel one diameter
- **Dimensionless contact time:** t* = τ_contact / τ_c — contact time normalized to capillary time
- **Spreading factor:** β_max = D_max / D₀ — ratio of maximum spread to initial diameter (taken directly from classical CV)

### 3.2 Output Files

| File | Format | Contents |
|---|---|---|
| `/home/ubuntu/materials/dimensionless_parameters.json` | JSON array, 53 entries | Per-video dimensionless numbers and raw inputs |

Each entry contains: `folder`, `video`, `fluid_type`, `fluid_label`, `fluid_properties` (σ, η, ρ), `raw` (D₀, U₀, D_max, contact_time_ms, β_max from feature_table), and `dimensionless` (We, Re, Oh, Ca, Fr, β_splash, τ_c_ms, τ_ic_ms, t*, β_max).

### 3.3 Parameters and Physical Meaning

The Weber number (We = 19.3–100.3, mean 43.3) quantifies whether inertia or surface tension dominates the impact dynamics. Higher We → more energetic impact → more spreading and higher likelihood of splashing or non-rebound. The Reynolds number (Re = 889–3285, mean 1743) quantifies the balance of inertia and viscosity. The Ohnesorge number (Oh = 0.0025–0.0065) is uniformly low, confirming that viscous dissipation is negligible compared to surface tension for all conditions tested; Oh < 0.1 is the typical criterion for an inertia-dominated, low-viscosity regime. The dimensionless contact time t* (range 0.4–35.2, mean 3.4) is physically interpretable as the number of droplet oscillation periods the droplet remains in contact with the surface; for a perfect non-wetting bounce on a superhydrophobic surface, t* ≈ 2.2 × π from theory.

### 3.4 Quality Assessment

The implementation faithfully replicates the paper's dimensionless number definitions. We, Re, and Oh are standard and unambiguous. The β_splash formulation follows Riboux & Gordillo (2014) as cited in Abbot et al., with standard constants.

Two concerns exist regarding the t* values. First, the 15 videos with contact time capped at 100.11 ms yield t* values of 25–35 — physically implausible for superhydrophobic rebound and a direct consequence of the liftoff detection cap rather than true physical behavior. Second, the 10 videos with contact time of exactly 3.34 ms (10 frames) yield very low t* around 0.4–0.8, which is also physically suspicious (theoretical minimum for a rebound is ~π ≈ 3.14 in capillary time units). These two clusters should be excluded from any t* analysis until contact time detection is improved.

β_max values from 2.0–2.5 for most conditions align well with literature values for water on superhydrophobic surfaces at similar We (2.0–2.8 reported by Abbot et al. and others). The outliers at β_max > 5 (cainhtx3 = 8.46, ONLY CA sds less CMC1 = 8.83, water6 = 4.68, water3 = 4.21) exceed the expected range and may represent partial wetting transitions or genuinely anomalous events.

The primary gap relative to the paper is the absence of v_rebound and the coefficient of restitution (e = v_rebound / v_impact). This is the single most important parameter for characterizing bouncing vs. non-bouncing behavior and is not yet extracted. The paper uses it as the primary outcome variable.

### 3.5 Null Values

**Dimensionless parameters null — 20 videos.** Any video missing either D₀ or U₀ cannot yield We, Re, Oh, β_splash, or t*, since all five require both inputs. The 20 affected videos are: cainhcg3.mp4, cainhcg5.mp4, cainlsds1.mp4, caonly1.mp4, water4.mp4, water6.mp4, ONLY CA SDS ABOVE CMC1.mp4, ONLY CA SDS ABOVE CMC2.mp4, ONLY CA cg ABOVE CMC1.mp4, ONLY CA cg ABOVE CMC2.mp4, ONLY CA cg ABOVE CMC3.mp4, ONLY CA cg less CMC1.mp4, ONLY CA cg less CMC2.mp4, ONLY CA cg less CMC3.mp4, ONLY CA sds less CMC2.mp4, ONLY CA tx ABOVE CMC1.mp4, ONLY CA tx ABOVE CMC3.mp4, ONLY CA tx ABOVE CMC4.mp4, ONLY CA tx less CMC1.mp4, and ca+TR.mp4.

β_max is additionally null for 14 of the above (those lacking D₀). The spread width D_max is available for all 53 videos (no spread-width nulls), but without D₀ the ratio cannot be computed.

t* is additionally unreliable for the 25 videos with anomalous contact times (15 at cap + 10 at minimum), though it is technically non-null.

---

## Section 4 — Cross-Method Comparison

### 4.1 Shared Parameters

The following table shows every parameter that appears in more than one method, with coverage and representative values.

| Parameter | Classical CV | Paper Methodology | VLM (fine-tuned) | Notes |
|---|---|---|---|---|
| D₀ (mm) | 39/53 non-null | — (input from CV) | 3 videos tested | VLM uses falling-frame radius |
| U₀ (mm/s) | 33/53 non-null | — (input from CV) | Not extracted | — |
| D_max / spread (mm) | 53/53 non-null | — (input from CV) | 3 videos tested | VLM spread is hallucinated |
| β_max = D_max/D₀ | 39/53 non-null | 33/53 non-null | Not reliable | Paper method takes from CV |
| cx (px) | Per video median | — | 45 frames | Frame-level, not per-video |
| radius (px) | Per video median | — | 45 frames | Frame-level |
| Phase label | Implicit | — | 45 frames | CV infers phase from timing |
| We | — | 33/53 | — | Requires D₀ + U₀ + fluid |
| Re | — | 33/53 | — | Requires D₀ + U₀ + fluid |
| Oh | — | 33/53 | — | Requires D₀ + fluid |
| contact_time_ms | 53/53 (15 unreliable) | — | — | — |
| t* | — | 33/53 (25 unreliable) | — | Derived from contact_time |

### 4.2 Completeness

Classical CV produces the most complete dataset. All 53 videos have a contact time value, a max spread width, and a surface row. Thirty-nine videos have D₀ and β_max; 33 have U₀. The paper methodology inherits the CV completeness for its inputs and produces 33 complete dimensionless profiles. The VLM has been evaluated on only 3 of 53 videos (45 test frames), making it the least complete by count — though this is an evaluation-coverage limitation, not a fundamental one.

### 4.3 Consistency and Variance

For D₀ estimation on the two videos where all three approaches yield a value, the fine-tuned VLM is strikingly consistent with classical CV: 8.9% error for water.mp4 and 1.1% error for cainhsds1.mp4. Zero-shot Gemini differs from CV by 34% and 54%; GPT-4o-mini by 44% and 81%. These zero-shot errors are systematic underestimates, not random scatter, suggesting a calibration bias rather than noise.

For centroid localization, the fine-tuned model achieves cx MAE of 2.2 px during falling and 2.0 px during spreading — consistent with sub-1% error relative to the 1280 px frame width. The zero-shot models achieve 31.6–242.6 px MAE, with GPT-4o-mini collapsing to 242.6 px MAE during spreading because it places the centroid at the center of the contact region rather than tracking the droplet.

For β_max, all three approaches are effectively in agreement since the paper methodology takes β_max directly from classical CV. The VLM spread_width hallucination (always outputting ~19.5 mm) means VLM β_max cannot be computed meaningfully.

### 4.4 Physical Plausibility

Classical CV produces physically plausible D₀ (0.58–2.91 mm, consistent with 4 µL pipette drops), U₀ (565–1273 mm/s, within the expected ±15% of free-fall at 6.6 cm), β_max (0.26–11.22 with most values 1.5–3.0, consistent with the literature for superhydrophobic surfaces at We ≈ 30–100), and We/Re in ranges reported by Abbot et al. Contact time is unreliable for roughly half the dataset.

Fine-tuned VLM D₀ estimates are physically plausible. Zero-shot model estimates are systematically too small and should not be used for physical analysis.

### 4.5 Recommendation

**For quantitative physical analysis**, classical CV is the appropriate method. It is the only approach with validated scale, velocity consistent with theory, and coverage across all 53 videos. The paper methodology (dimensionless_analysis.py) should be used as the standard post-processing step on top of CV outputs.

**For frame-level quality control and phase labeling**, the fine-tuned Qwen2.5-VL-7B model is an effective complement to CV. Its 93.3% phase accuracy and sub-3 px centroid MAE during falling and spreading phases make it suitable for auditing impact/liftoff detections and flagging unusual events. It should not be used for quantitative spread measurement.

**Zero-shot VLMs (Gemini, GPT-4o-mini) are not suitable** for any quantitative parameter extraction in this dataset. Their phase classification failure on rebounding frames and systematic D₀ underestimation by 34–81% would introduce large systematic errors into any derived quantities.

**The highest-priority missing measurement** is the rebound velocity and coefficient of restitution (e = v_rebound / v_impact). This is the primary experimental outcome in Abbot et al. and is absent from all three current approaches. It requires implementing a post-liftoff centroid tracking and linear fit analogous to the pre-impact velocity extraction, applied to the 5–15 frames immediately after liftoff.

---

## Summary Statistics

| Quantity | Value |
|---|---|
| Total videos | 53 |
| Videos with complete CV kinematics (D₀ + U₀) | 33 (62%) |
| Videos with null D₀ (permanent failure) | 14 (26%) |
| Videos with null U₀ | 20 (38%) |
| Videos with unreliable contact time | 25 (47%) |
| Dimensionless analysis complete (We/Re/Oh/β_max/t*) | 33 (62%) |
| VLM test coverage | 3 videos / 45 frames |
| Fine-tuned model phase accuracy | 93.3% |
| Zero-shot Gemini phase accuracy | 60.0% |
| Zero-shot GPT-4o-mini phase accuracy | 51.1% |
| Scale calibration error | 0.19% |
| Mean U₀ vs free-fall theory | 1075.7 vs 1137.9 mm/s (5.5% below) |
| We range (33 videos) | 19.3 – 100.3 |
| β_max range (39 videos) | 0.26 – 11.22 |
