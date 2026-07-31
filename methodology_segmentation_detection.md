# Image Segmentation and Droplet Detection Methods

## Overview

Two complementary pipelines were developed to extract quantitative measurements from high-speed shadowgraphy videos: a classical computer vision pipeline (`ellipse_timeseries_v2`) for frame-by-frame diameter, velocity, and spreading measurements, and a deep-learning segmentation pipeline (`analyze_droplet_sam2`) for full mask-based tracking throughout the impact event. Both pipelines operate on the same calibrated coordinate system and output per-frame time-series data in physical units.

All videos were captured at a true frame rate of 2996.766 fps (OpenCV reports the nominal 60 fps encoded in the file header, which was systematically ignored). Spatial calibration was performed from scale-bar recordings acquired in each experimental session, yielding pixel-to-millimetre conversion factors of 65.625 px mm⁻¹ (sessions 02182026 and 03242026), 66.0 px mm⁻¹ (session 05052026), 66.5 px mm⁻¹ (session 05112026), and 56.0 px mm⁻¹ (sessions 05122026 and 05172026). The difference in the final two sessions reflects a camera repositioned approximately 15% further from the impact site.

---

## 1. Classical Computer Vision Pipeline (v3)

The classical pipeline performs a structured four-pass analysis on each video, using the known `impact_frame` and `surface_row` from a pre-computed feature table as anchors.

### 1.1 Image Preprocessing

Every frame is converted to 8-bit greyscale and enhanced with Contrast-Limited Adaptive Histogram Equalization (CLAHE; clip limit 3.0, tile grid 8×8 pixels). CLAHE locally normalises intensity variations caused by the bright caustic ring at the droplet boundary and the dark shadow of the droplet body, making both HoughCircles and template matching more robust across different surfactant concentrations, which alter droplet opacity.

### 1.2 Pre-impact Diameter (D₀) — Two-Pass Scan

**Pass 1 — HoughCircles backward scan.**
Starting from `impact_frame − 2` and scanning back up to 40 frames, the Generalised Hough Transform for circles is applied to each preprocessed frame. The accumulator threshold (`param2`) is progressively relaxed from 20 → 15 → 12 → 10 until at least one circle is returned, making the detector adaptive to contrast variations without requiring per-video tuning. Circles are accepted only if the centre lies strictly above the calibrated surface row and at least 5 pixels below the top of the frame (preventing nozzle artefacts from being accepted). The initial candidate radius window is 45–110 px, corresponding to approximately 1.4–3.4 mm, comfortably bracketing the expected 4 µL drop diameter of ~2 mm. 

For each accepted detection, an ellipse is fitted to the thresholded foreground region within the detected circle to obtain sub-pixel centre coordinates and a shape-averaged diameter. Outlier horizontal positions are removed using the median cx of the five detections closest to impact as an anchor, rejecting any detection more than 200 px away (eliminating nozzle-side fixed artefacts). The final D₀ estimate is the median of the last five valid radius measurements (those closest to impact, where the droplet is fully in frame).

**Pass 1 quality checks — template D₀ fallback.**
Four failure modes trigger a multi-scale template search to replace the HoughCircles D₀ estimate:
- D₀ < 60 px (probable nozzle, not droplet);
- fewer than 3 total detections (insufficient for a reliable median);
- D₀ in the range 82–98 px (detection stuck at the minimum radius limit, not the droplet);
- D₀ > 155 px (nozzle tip detected instead of the droplet, as in `caonly2`).

**Pass 1 template fallback.** A synthetic template is constructed matching the optical signature of a droplet in shadowgraphy: a dark filled disc (pixel value 30) on a neutral grey background (160), with a bright annular ring (230) at the perimeter of width ~1/6 of the radius, representing the caustic. Candidate radii are swept from 50 to 110 px in 5 px steps. At each candidate radius, the template is cross-correlated against the pre-impact search region using normalised cross-correlation (TM_CCOEFF_NORMED). The radius whose matches yield the highest composite score (mean confidence × number of confident matches) is selected. This approach is robust to absolute intensity differences between experimental sessions because the normalised metric is sensitive to pattern shape rather than absolute brightness.

### 1.3 Impact Velocity (U₀) — Pass 2

**Template matching velocity.**
A droplet-shaped template sized to D₀ is cross-correlated against successive pre-impact frames using the same normalised metric. A confidence threshold of 0.30 is required for a position to contribute to the velocity estimate, rising to 0.20 in a secondary relaxed pass for low-contrast videos. The template search band is restricted to the region between 5 px below the frame top and `surface_row − r − 8` px (i.e., it cannot overlap the surface), preventing false matches on reflections. The impact velocity U₀ is computed as the Theil-Sen estimator of the downward displacement rate: the median of all adjacent-frame slopes dy/dt (in px per frame), converted to mm s⁻¹ using FPS_ACTUAL and the calibrated px mm⁻¹ factor. The Theil-Sen estimator is robust to isolated outlier detections.

**Optical flow fallback.**
If template matching yields fewer than 3 confident positions (typical for very low surfactant concentrations where the droplet becomes nearly transparent), Lucas-Kanade sparse optical flow is seeded at the last HoughCircles detection and tracked backward through up to 20 pre-impact frames. LK uses a 31×31 pixel window and a 3-level Gaussian pyramid, tracking the single droplet-centre feature point. The Theil-Sen velocity is then computed on these positions. A final fallback uses the HoughCircles positions directly, filtered to radii within 65%–150% of D₀.

### 1.4 Impact-Frame Refinement

The `impact_frame` entries in the feature table are derived from a frame-difference heuristic and may be off by a few frames. A local refinement scan searches ±4 frames around the stored estimate. A background image is computed as the median of the 8 frames preceding the estimate window. For each candidate frame, the horizontal extent of the foreground region within a 45 px band centred on the surface row is measured (background-subtracted thresholding, threshold 25 grey levels). The first frame at which this contact width exceeds 1 mm (in pixels: `px_per_mm × 1.0`) is declared the true impact frame. This correction eliminates ±5-frame errors that would inflate β by prematurely including early spreading frames.

### 1.5 Spreading Phase and Maximum Spreading Factor (β_max)

During the spreading phase (from `impact_frame` to `liftoff_frame`), the contact footprint diameter is measured per frame by background subtraction. The background is the median of the 5 frames immediately preceding impact. The foreground is thresholded (threshold 25), and the horizontal extent of any non-zero columns within the 50 px surface band (`surface_row − 40` to `surface_row + 10`) is recorded as the contact width. This approach is insensitive to the exact threshold because the lamella, even at its thinnest, produces a grey-level change well above 25 counts relative to the clean surface background.

The maximum spreading factor is:

$$\beta_{\max} = \frac{D_{\max}}{D_0}$$

where D_max is the maximum contact width in pixels divided by `px_per_mm`.

**True liftoff detection.** After the full spreading timeseries is assembled, the true liftoff frame is identified as the first frame after peak spreading at which the contact width drops below 10% of D₀. This is more physically meaningful than fixed offsets or HoughCircles-based rebound detection because it directly observes the droplet detaching from the surface.

### 1.6 Rebound Phase and Coefficient of Restitution (COR)

After liftoff, HoughCircles is run for up to 30 frames with the radius window constrained to [0.60 r₀, 1.45 r₀] where r₀ = D₀/2. This constraint eliminates the majority of false detections from surface reflections and the nozzle shadow, which in v1 inflated U_rebound above U₀ and suppressed COR computation. Tracking stops if three consecutive frames yield no detection, if the droplet centre moves downward for three consecutive frames (indicating re-impact or a secondary satellite drop), or if the droplet exits the top of the frame. The Theil-Sen estimator is applied to the rebound positions to obtain U_rebound. The coefficient of restitution is computed as:

$$e = \frac{U_{\text{rebound}}}{U_0}$$

and is only recorded when e ≤ 1.0 (physically required) and U₀ > 200 mm s⁻¹ (minimum reliability threshold).

### 1.7 Per-frame Output

For each video, the pipeline writes one CSV row per frame from the first visible drop frame through the end of the rebound window. Columns include: `frame`, `phase` (falling / spreading / rebounding), `cx_px`, `cy_px`, `radius_px`, `spread_width_px`, `detection_method`, `confidence`, `D_mm`, `beta`, `velocity_mm_s`, `time_ms`, `dist_travelled_px`, `px_per_mm`, and `state_change`. In the spreading phase, the droplet centre and radius columns are null (the droplet is a flat lamella); `spread_width_px` and `beta` are populated instead. In the falling and rebound phases, `spread_width_px` and `beta` are null. This asymmetry reflects the physical geometry: the droplet cannot simultaneously be tracked as a sphere and measured as a spreading lamella.

State-frame images are saved at four fixed moments per video: the first visible falling frame, the impact frame (spreading onset), the maximum spreading frame, and the true liftoff frame, providing visual quality-control checkpoints for every video without storing the full frame sequence.

---

## 2. SAM2 Segmentation Pipeline

The SAM2 pipeline provides dense, mask-based tracking throughout the entire impact sequence, capturing the full droplet silhouette per frame rather than a single ellipse or contact width measurement. It is particularly valuable during the spreading and breakup phases when the droplet morphology departs significantly from a sphere, and for computing centroid trajectories and projected area timeseries in absolute physical units.

### 2.1 Reference Frame Detection

The video is initialised by computing a background model from the mean of the first 30 frames (pre-drop frames captured before the droplet enters the field of view). Background subtraction (absolute pixel difference, threshold 25 grey levels) is applied to every subsequent frame. Morphological closing (5×5 elliptical kernel) followed by opening removes noise and fills gaps in the foreground mask. Contours in the thresholded foreground are evaluated against two criteria: minimum area of 150 px² (rejecting dust and specular reflections) and minimum circularity of 0.3 (4π·area/perimeter²; rejecting elongated streaks). The first frame that contains a contiguous foreground region satisfying both criteria, with its bounding box at least 3 px inside all frame borders (preventing partial detections at the nozzle exit), is selected as the reference frame. The centroid of that foreground region is computed from image moments and used as the SAM2 prompt point.

### 2.2 SAM2 Propagation

SAM2 (Segment Anything Model 2, version 2.1 Hiera Large) is used as a video object segmentation model. The model receives the reference centroid as a single positive point prompt at the reference frame and propagates a binary mask forward through all remaining frames using its internal memory-attention mechanism. Inference runs under `torch.bfloat16` precision on a GPU to reduce memory footprint. For videos with fewer than 5000 frames, all frames are processed (frame_step = 1); for longer recordings (≥ 5000 frames), every fourth frame is sampled (frame_step = 4) to prevent GPU out-of-memory errors while maintaining 750 effective frames per second temporal resolution — sufficient to resolve all phases of the impact at the experimental time scales.

Frames are extracted to disk as JPEG images (quality 95) before SAM2 ingestion, as required by the SAM2 video predictor API. The temporary directory is deleted after propagation completes.

Connected-component analysis (minimum component area: max(50, min_area / 3) = 50 px²) is applied to each propagated mask to handle post-impact satellite droplets and lamella fragmentation. Components are sorted by area and assigned monotonically increasing `drop_id` values (1 = largest fragment), allowing fragment trajectories to be tracked independently in the output CSV.

### 2.3 Exhaustive Fallback Chain

SAM2 occasionally loses the mask — typically when the droplet becomes a very thin lamella, exits the frame during rebound, or is occluded by a satellite drop. To guarantee one CSV row per original frame without gaps, a three-level fallback chain is applied to every frame at which SAM2 returns an empty mask, and to all inter-sample frames when frame_step > 1:

**Fallback 1 — HoughCircles.** CLAHE-enhanced greyscale is passed to the Generalised Hough Transform with the accumulator threshold relaxed progressively (param2: 20 → 15 → 12 → 10). The radius window is [0.5 r_ref, 1.8 r_ref] where r_ref = √(A_ref / π) is the radius implied by the SAM2 reference area. Only circles whose centre lies above the surface row and at least 5 px below the top of frame are accepted; the uppermost qualifying circle is returned.

**Fallback 2 — Template matching.** If HoughCircles fails, a synthetic shadowgraphy template (dark disc with bright annular caustic ring, sized to r_ref) is cross-correlated with the CLAHE frame using TM_CCOEFF_NORMED. The search region is restricted to between 5 px below frame top and `surface_row − r_ref − 8` px. A minimum correlation coefficient of 0.20 is required. This threshold is intentionally lower than the v3 pipeline's 0.30 because the fallback is only reached when HoughCircles has already failed; the template is sized to match the known droplet radius so false positives from other objects (e.g., nozzle tip of different size) are unlikely to reach the threshold.

**Fallback 3 — Last-known position.** If both detection methods fail, the centroid and radius from the most recent successfully detected frame are carried forward. This ensures continuity over gaps of a few frames (e.g., during the briefest moments of total lamella collapse) while being transparent in the output via the `detection_method` field.

If all three fallbacks are exhausted, the row is written with `drop_id = 0`, `cx = null`, `cy = null`, `area_px = null`, and `detection_method = "null"`.

### 2.4 Area Normalisation (percentage)

For each frame, the projected area of each detected fragment is expressed as a percentage of the reference frame area (the area of the SAM2 mask at the reference frame, or the OpenCV contour area if SAM2 initialisation yields no mask). Values above 100% are physically valid during spreading, as the projected area of the lamella exceeds that of the spherical pre-impact droplet. Values substantially above 300% typically indicate that the mask has adhered to the wetted surface region rather than the droplet alone, and should be treated with caution.

### 2.5 Velocity Computation

Frame-to-frame centroid velocity is computed for each `drop_id` independently. For consecutive frames i and i+1 with valid centroid coordinates (cx, cy):

$$v = \frac{\sqrt{(\Delta c_x)^2 + (\Delta c_y)^2}}{\Delta t} \cdot \frac{1}{\text{px\_per\_mm}}$$

where Δt = (frame_i+1 − frame_i) / FPS_ACTUAL in seconds, and `px_per_mm` is the session-specific calibration constant. Three columns are written: `distance_px` (Euclidean displacement in pixels), `velocity_px_per_s`, and `velocity_mm_s`. The formula accounts for non-unit frame gaps that arise when frame_step > 1 or when individual frames were skipped by the fallback chain. Velocity is undefined (null) for the first detected frame of each drop_id and for any frame with a null centroid.

### 2.6 Output Format

Each video produces one CSV file with columns: `frame`, `drop_id`, `cx`, `cy`, `area_px`, `percentage`, `detection_method`, `distance_px`, `velocity_px_per_s`, `velocity_mm_s`. The `detection_method` field takes values `sam2`, `hough`, `template`, `last_known`, or `null`, enabling post-hoc quality filtering (e.g., excluding last_known rows from velocity analysis). The total number of rows equals `total_frames − reference_frame_index`, guaranteeing a continuous timeseries aligned to a common physical time origin.

---

## 3. Calibration and Coordinate System

The surface row (pixel row at which the superhydrophobic substrate surface appears in the image) was determined per video from the calibrated feature table, which combines automated Sobel-edge detection (horizontal gradient magnitude summed across each row, restricted to 25%–85% of frame height to avoid the nozzle and bottom aperture) with manual verification. All detections that report a centroid below `surface_row − r` are rejected as surface-contact or spurious artefacts, in both the classical and SAM2 pipelines.

Physical time is referenced to the first visible falling frame (classical pipeline) or the reference frame (SAM2 pipeline), with t = 0 defined at that anchor. Frame indices are converted to milliseconds as t_ms = (frame − anchor_frame) / FPS_ACTUAL × 1000.
