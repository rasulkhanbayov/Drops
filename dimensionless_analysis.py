"""
Dimensionless Parameter Analysis — Droplet Impact Dataset
==========================================================
Replicates the methodology of:
  Abbot et al., J. Colloid Interface Sci. 693 (2025) 137570
  "Nanoparticles do not influence droplet break-up, spreading, or splashing"

For each video the script computes:

  Kinematic (from feature_table.json)
  ─────────────────────────────────────
  D0          Initial droplet diameter            [mm, m]
  U0          Impact velocity                     [m/s]

  Dimensionless numbers  (paper Sec. 2–3)
  ─────────────────────────────────────────
  We          Weber number       = ρ D0 U0² / σ
  Re          Reynolds number    = ρ D0 U0 / η
  Oh          Ohnesorge number   = η / sqrt(ρ σ D0)  =  sqrt(We)/Re
  Ca          Capillary number   = η U0 / σ
  Fr          Froude number      = U0 / sqrt(g D0)
  beta_splash Splashing ratio    = (1/tan α) · (ηg² ρ D0 U0⁵ / σ²)^(1/6)
              (Riboux & Gordillo, Eq.1 in the paper; ηg = air viscosity)

  Time scales  (paper Sec. 3.1 / 3.2)
  ──────────────────────────────────────
  tau_c       Capillary time     = sqrt(ρ R0³ / σ)   [ms]
  t_star      Dimensionless contact time = t_contact / tau_c

  Spreading
  ──────────
  beta_max    Maximum spread factor  d_max / D0   (already in feature_table)

Output
──────
  dimensionless_parameters.json  — one entry per video

Fluid properties
────────────────
Properties are inferred from video names (see FLUID_PROPS below).
All viscosities ≈ water (1 mPa·s) because:
  • This paper shows CA nanoparticles have negligible effect on η
  • Dilute surfactants (≤CMC) have negligible effect on η
Surface tension varies with surfactant type and concentration.

Reference values at T = 20°C:
  Water:             σ = 72.4 mN/m,  η = 1.00 mPa·s,  ρ = 997  kg/m³
  SDS above CMC:     σ = 37.0 mN/m,  η = 1.00 mPa·s,  ρ = 999  kg/m³
  SDS below CMC:     σ = 58.0 mN/m,  η = 1.00 mPa·s,  ρ = 998  kg/m³
  TX-100 above CMC:  σ = 30.0 mN/m,  η = 1.02 mPa·s,  ρ = 998  kg/m³
  TX-100 below CMC:  σ = 50.0 mN/m,  η = 1.01 mPa·s,  ρ = 998  kg/m³
  CG/CAPB above CMC: σ = 32.0 mN/m,  η = 1.01 mPa·s,  ρ = 998  kg/m³
  CG/CAPB below CMC: σ = 52.0 mN/m,  η = 1.01 mPa·s,  ρ = 998  kg/m³
  CA nanoparticles only: same as water (per Abbot et al. 2025 finding)
"""

import json
import math
import re
from pathlib import Path

# ── Physical constants ────────────────────────────────────────────────────────
G      = 9.81          # gravitational acceleration [m/s²]
ETA_G  = 1.81e-5       # air dynamic viscosity [Pa·s]  (for splashing ratio)
TAN_ALPHA = 1.0        # lamella angle at spreading ≈ 45°, tan(45°)=1 [dimensionless]
                       # (Riboux & Gordillo; typical value for smooth glass)

# ── Fluid property lookup ─────────────────────────────────────────────────────
# Each entry: (sigma_N_per_m, eta_Pa_s, rho_kg_per_m3, label)
FLUID_PROPS = {
    "water":           (0.0724, 0.001000, 997.0, "Pure water"),
    "ca_only":         (0.0724, 0.001000, 997.0, "CA nanoparticles in water (additive-free)"),
    "sds_above_cmc":   (0.0370, 0.001000, 999.0, "CA + SDS above CMC"),
    "sds_below_cmc":   (0.0580, 0.001000, 998.0, "CA + SDS below CMC"),
    "tx_above_cmc":    (0.0300, 0.001020, 998.0, "CA + TX-100 above CMC"),
    "tx_below_cmc":    (0.0500, 0.001010, 998.0, "CA + TX-100 below CMC"),
    "cg_above_cmc":    (0.0320, 0.001010, 998.0, "CA + CG/CAPB above CMC"),
    "cg_below_cmc":    (0.0520, 0.001010, 998.0, "CA + CG/CAPB below CMC"),
    "tx_only":         (0.0300, 0.001020, 998.0, "TX-100 only above CMC"),
    "unknown":         (0.0724, 0.001000, 997.0, "Unknown (water defaults used)"),
}


def classify_fluid(video_name: str, folder: str) -> str:
    """
    Infer fluid type from video filename and folder.
    Naming conventions from the 02182026 and 03242026 dataset folders.
    """
    v = video_name.lower()

    # ── 02182026 folder naming ────────────────────────────────────────────────
    if v.startswith("water"):
        return "water"
    if v.startswith("caonly"):
        return "ca_only"
    if v == "tx.mp4":
        return "tx_only"

    # cainhsds / cainlsds  — CA + SDS  (h=high, l=low concentration)
    if re.search(r"cainh.*sds", v):
        return "sds_above_cmc"
    if re.search(r"cainl.*sds", v):
        return "sds_below_cmc"

    # cainhtx / cainltx  — CA + TX-100
    if re.search(r"cainh.*tx", v):
        return "tx_above_cmc"
    if re.search(r"cainl.*tx", v):
        return "tx_below_cmc"

    # cainhcg / cainlcg  — CA + CG/CAPB
    if re.search(r"cainh.*cg", v):
        return "cg_above_cmc"
    if re.search(r"cainl.*cg", v):
        return "cg_below_cmc"

    # ── 03242026 folder naming ────────────────────────────────────────────────
    if "sds above cmc" in v or "sds above" in v:
        return "sds_above_cmc"
    if "sds less cmc" in v or "sds below" in v or "0.45percrnt sds" in v:
        return "sds_below_cmc"
    if "tx above cmc" in v or "0.028percrnt tx" in v:
        return "tx_above_cmc"
    if "tx less cmc" in v or "tx below" in v:
        return "tx_below_cmc"
    if "cg above cmc" in v or "0.001percent cg" in v or "0.028p" in v:
        return "cg_above_cmc"
    if "cg less cmc" in v or "cg below" in v:
        return "cg_below_cmc"
    if "ca+tr" in v:
        return "ca_only"

    return "unknown"


# ── Dimensionless number calculators ─────────────────────────────────────────
def weber(rho, D0_m, U0):
    """We = ρ D₀ U₀² / σ  (inertia / surface tension)"""
    return None  # needs sigma — computed in main

def compute_dimensionless(D0_mm, U0_mm_s, contact_ms, beta_max,
                          sigma, eta, rho):
    """
    All inputs in SI except:
      D0_mm       [mm]
      U0_mm_s     [mm/s]
      contact_ms  [ms]
      beta_max    [dimensionless]
    Returns dict of all computed quantities.
    """
    if D0_mm is None or U0_mm_s is None:
        return None

    D0   = D0_mm   * 1e-3          # m
    R0   = D0 / 2                  # m
    U0   = U0_mm_s * 1e-3          # m/s
    t_c  = contact_ms * 1e-3 if contact_ms is not None else None

    We   = rho * D0 * U0**2 / sigma
    Re   = rho * D0 * U0   / eta
    Oh   = eta  / math.sqrt(rho * sigma * D0)
    Ca   = eta  * U0        / sigma
    Fr   = U0   / math.sqrt(G * D0)

    # Splashing ratio β — Riboux & Gordillo (2014), Eq. 1 in paper
    # β = (1/tan α) · (ηg² ρ D₀ U₀⁵ / σ²)^(1/6)
    try:
        beta_splash = (1.0 / TAN_ALPHA) * (
            (ETA_G**2 * rho * D0 * U0**5) / sigma**2
        )**(1.0 / 6.0)
    except Exception:
        beta_splash = None

    # Capillary time scale τ_c = sqrt(ρ R₀³ / σ)  [s → ms]
    tau_c_s  = math.sqrt(rho * R0**3 / sigma)
    tau_c_ms = tau_c_s * 1e3

    # Dimensionless contact time t* = t_contact / τ_c
    t_star = (t_c / tau_c_s) if (t_c is not None) else None

    # Inertia-capillary time τ_ic = D₀ / U₀  [ms]  — time scale of impact
    tau_ic_ms = (D0 / U0) * 1e3 if U0 > 0 else None

    return {
        "We":            round(We,   3),
        "Re":            round(Re,   1),
        "Oh":            round(Oh,   6),
        "Ca":            round(Ca,   6),
        "Fr":            round(Fr,   3),
        "beta_splash":   round(beta_splash, 4) if beta_splash else None,
        "tau_c_ms":      round(tau_c_ms, 4),
        "tau_ic_ms":     round(tau_ic_ms, 4) if tau_ic_ms else None,
        "t_star":        round(t_star, 2) if t_star else None,
        "beta_max":      round(beta_max, 4) if beta_max else None,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    feat_path = Path("/home/ubuntu/materials/feature_table.json")
    out_path  = Path("/home/ubuntu/materials/dimensionless_parameters.json")

    features = json.loads(feat_path.read_text())
    results  = []

    print(f"\n{'Video':<45} {'Fluid type':<28} {'We':>8} {'Re':>7} {'Oh':>9} {'β_max':>7} {'t*':>7}")
    print("─" * 120)

    for f in features:
        video   = f["video"]
        folder  = f["folder"]
        D0_mm   = f.get("pre_impact_diameter_mm")
        U0_mms  = f.get("impact_velocity_mm_per_s")
        contact = f.get("contact_time_ms")
        b_max   = f.get("max_spread_factor")

        fluid_key = classify_fluid(video, folder)
        sigma, eta, rho, fluid_label = FLUID_PROPS[fluid_key]

        dims = compute_dimensionless(D0_mm, U0_mms, contact, b_max, sigma, eta, rho)

        entry = {
            "folder":           folder,
            "video":            video,
            "fluid_type":       fluid_key,
            "fluid_label":      fluid_label,
            "fluid_properties": {
                "sigma_N_per_m":  sigma,
                "eta_Pa_s":       eta,
                "rho_kg_per_m3":  rho,
            },
            "raw": {
                "D0_mm":              D0_mm,
                "U0_mm_per_s":        U0_mms,
                "contact_time_ms":    contact,
                "max_spread_width_mm": f.get("max_spread_width_mm"),
                "beta_max":           b_max,
            },
            "dimensionless": dims,
        }
        results.append(entry)

        # Console summary
        if dims:
            We   = dims["We"]
            Re   = dims["Re"]
            Oh   = dims["Oh"]
            bm   = dims["beta_max"] or "—"
            ts   = dims["t_star"]   or "—"
            print(f"  {video:<43} {fluid_label:<28} {We:>8.1f} {Re:>7.0f} {Oh:>9.5f} {str(bm):>7} {str(ts):>7}")
        else:
            print(f"  {video:<43} {fluid_label:<28} {'[missing D0 or U0]':>40}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDone. {len(results)} videos → {out_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    valid = [r for r in results if r["dimensionless"] is not None]
    We_vals = [r["dimensionless"]["We"]  for r in valid]
    Re_vals = [r["dimensionless"]["Re"]  for r in valid]
    Oh_vals = [r["dimensionless"]["Oh"]  for r in valid]
    bm_vals = [r["dimensionless"]["beta_max"] for r in valid if r["dimensionless"]["beta_max"]]
    ts_vals = [r["dimensionless"]["t_star"]   for r in valid if r["dimensionless"]["t_star"]]

    import numpy as np
    print(f"\n── Dataset summary ({len(valid)} videos with complete kinematics) ──")
    print(f"  We       : {min(We_vals):.1f} – {max(We_vals):.1f}   (mean {np.mean(We_vals):.1f})")
    print(f"  Re       : {min(Re_vals):.0f} – {max(Re_vals):.0f}   (mean {np.mean(Re_vals):.0f})")
    print(f"  Oh       : {min(Oh_vals):.5f} – {max(Oh_vals):.5f}")
    print(f"  β_max    : {min(bm_vals):.2f} – {max(bm_vals):.2f}   (mean {np.mean(bm_vals):.2f})")
    print(f"  t*       : {min(ts_vals):.1f} – {max(ts_vals):.1f}   (mean {np.mean(ts_vals):.1f})")


if __name__ == "__main__":
    main()
