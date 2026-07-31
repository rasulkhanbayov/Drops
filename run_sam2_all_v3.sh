#!/bin/bash
# SAM2 v3 batch — all folders, full fallback chain, per-frame output
# Saves to results_drops/<folder>_sam2_v3_results/

CKPT="/data/checkpoints/sam2.1_hiera_large.pt"
CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
PY="/data/venv/bin/python"

cd /home/ubuntu/materials

run() {
    local video="$1"
    local out_csv="$2"
    local surface_y="$3"
    local step="${4:-1}"
    local px_per_mm="${5:-65.625}"
    local impact="${6:-}"
    local liftoff="${7:-}"
    local stem=$(basename "$video" .mp4)
    if [ ! -f "$video" ]; then
        echo "[skip] $stem — file not found or corrupted"
        return
    fi
    if [ -f "$out_csv" ]; then
        echo "[skip] $stem — already done"
        return
    fi
    echo ">>> $stem  (surface_y=$surface_y  frame_step=$step  px_per_mm=$px_per_mm  impact=$impact  liftoff=$liftoff)"
    local extra_args=""
    [ -n "$impact"  ] && extra_args="$extra_args --impact_frame $impact"
    [ -n "$liftoff" ] && extra_args="$extra_args --liftoff_frame $liftoff"
    $PY analyze_droplet_sam2.py \
        --video "$video" \
        --output "$out_csv" \
        --checkpoint "$CKPT" \
        --model_cfg "$CFG" \
        --surface_y "$surface_y" \
        --frame_step "$step" \
        --px_per_mm "$px_per_mm" \
        $extra_args
}

# ── 02182026 ──────────────────────────────────────────────────────────────────
VDIR="/ephemeral/videos/02182026"
OUT="results_drops/02182026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 02182026 ==="

run "$VDIR/cainhcg1.mp4"    "$OUT/cainhcg1_sam2.csv"    400 1 65.625
run "$VDIR/cainhcg2.mp4"    "$OUT/cainhcg2_sam2.csv"    433 1 65.625
run "$VDIR/cainhcg3.mp4"    "$OUT/cainhcg3_sam2.csv"    437 1 65.625
run "$VDIR/cainhcg4.mp4"    "$OUT/cainhcg4_sam2.csv"    433 1 65.625
run "$VDIR/cainhcg5.mp4"    "$OUT/cainhcg5_sam2.csv"    433 1 65.625
run "$VDIR/cainhsds1.mp4"   "$OUT/cainhsds1_sam2.csv"   433 1 65.625
run "$VDIR/cainhsds2.mp4"   "$OUT/cainhsds2_sam2.csv"   430 1 65.625
run "$VDIR/cainhsds3.mp4"   "$OUT/cainhsds3_sam2.csv"   428 1 65.625
run "$VDIR/cainhtx1.mp4"    "$OUT/cainhtx1_sam2.csv"    428 1 65.625
run "$VDIR/cainhtx2.mp4"    "$OUT/cainhtx2_sam2.csv"    428 1 65.625
run "$VDIR/cainhtx3.mp4"    "$OUT/cainhtx3_sam2.csv"    402 1 65.625
run "$VDIR/cainlcg1.mp4"    "$OUT/cainlcg1_sam2.csv"    433 1 65.625
run "$VDIR/cainlcg2.mp4"    "$OUT/cainlcg2_sam2.csv"    433 1 65.625
run "$VDIR/cainlcg3.mp4"    "$OUT/cainlcg3_sam2.csv"    399 1 65.625
run "$VDIR/cainlsds1.mp4"   "$OUT/cainlsds1_sam2.csv"   427 1 65.625
run "$VDIR/cainlsds2.mp4"   "$OUT/cainlsds2_sam2.csv"   426 1 65.625
run "$VDIR/cainlsds3.mp4"   "$OUT/cainlsds3_sam2.csv"   417 1 65.625
run "$VDIR/cainltx1.mp4"    "$OUT/cainltx1_sam2.csv"    433 1 65.625
run "$VDIR/cainltx2.mp4"    "$OUT/cainltx2_sam2.csv"    428 1 65.625
run "$VDIR/cainltx3.mp4"    "$OUT/cainltx3_sam2.csv"    422 1 65.625
run "$VDIR/caonly1.mp4"     "$OUT/caonly1_sam2.csv"     399 1 65.625
run "$VDIR/caonly2.mp4"     "$OUT/caonly2_sam2.csv"     405 1 65.625
run "$VDIR/caonly3.mp4"     "$OUT/caonly3_sam2.csv"     433 1 65.625
run "$VDIR/tx.mp4"          "$OUT/tx_sam2.csv"          417 1 65.625
run "$VDIR/water.mp4"       "$OUT/water_sam2.csv"       433 1 65.625
run "$VDIR/water2.mp4"      "$OUT/water2_sam2.csv"      433 1 65.625
run "$VDIR/water3.mp4"      "$OUT/water3_sam2.csv"      433 1 65.625
run "$VDIR/water4.mp4"      "$OUT/water4_sam2.csv"      417 1 65.625
run "$VDIR/water5.mp4"      "$OUT/water5_sam2.csv"      417 1 65.625
run "$VDIR/water6.mp4"      "$OUT/water6_sam2.csv"      426 1 65.625

echo "02182026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

# ── 03242026 ──────────────────────────────────────────────────────────────────
VDIR="/ephemeral/videos/03242026_particlesonlypreparedinsurfactant"
OUT="results_drops/03242026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 03242026 ==="

run "$VDIR/0.001percent cg.mp4"        "$OUT/0.001percent cg_sam2.csv"        404 1 65.625
run "$VDIR/0.028p.mp4"                 "$OUT/0.028p_sam2.csv"                 404 1 65.625
run "$VDIR/0.028percrnt tx.mp4"        "$OUT/0.028percrnt tx_sam2.csv"        467 1 65.625
run "$VDIR/0.45percrnt sds.mp4"        "$OUT/0.45percrnt sds_sam2.csv"        454 1 65.625
run "$VDIR/ONLY CA SDS ABOVE CMC.mp4"  "$OUT/ONLY CA SDS ABOVE CMC_sam2.csv"  481 1 65.625
run "$VDIR/ONLY CA SDS ABOVE CMC1.mp4" "$OUT/ONLY CA SDS ABOVE CMC1_sam2.csv" 481 1 65.625
run "$VDIR/ONLY CA SDS ABOVE CMC2.mp4" "$OUT/ONLY CA SDS ABOVE CMC2_sam2.csv" 481 1 65.625
run "$VDIR/ONLY CA cg ABOVE CMC1.mp4"  "$OUT/ONLY CA cg ABOVE CMC1_sam2.csv"  485 1 65.625
run "$VDIR/ONLY CA cg ABOVE CMC2.mp4"  "$OUT/ONLY CA cg ABOVE CMC2_sam2.csv"  481 1 65.625
run "$VDIR/ONLY CA cg less CMC1.mp4"   "$OUT/ONLY CA cg less CMC1_sam2.csv"   470 1 65.625
run "$VDIR/ONLY CA cg less CMC2.mp4"   "$OUT/ONLY CA cg less CMC2_sam2.csv"   465 1 65.625
run "$VDIR/ONLY CA cg less CMC3.mp4"   "$OUT/ONLY CA cg less CMC3_sam2.csv"   473 1 65.625
run "$VDIR/ONLY CA sds less CMC1.mp4"  "$OUT/ONLY CA sds less CMC1_sam2.csv"  471 1 65.625
run "$VDIR/ONLY CA sds less CMC2.mp4"  "$OUT/ONLY CA sds less CMC2_sam2.csv"  470 1 65.625
run "$VDIR/ONLY CA tx ABOVE CMC1.mp4"  "$OUT/ONLY CA tx ABOVE CMC1_sam2.csv"  482 1 65.625
run "$VDIR/ONLY CA tx ABOVE CMC2.mp4"  "$OUT/ONLY CA tx ABOVE CMC2_sam2.csv"  471 1 65.625
run "$VDIR/ONLY CA tx ABOVE CMC3.mp4"  "$OUT/ONLY CA tx ABOVE CMC3_sam2.csv"  470 1 65.625
run "$VDIR/ONLY CA tx ABOVE CMC4.mp4"  "$OUT/ONLY CA tx ABOVE CMC4_sam2.csv"  471 1 65.625
run "$VDIR/ONLY CA tx less CMC1.mp4"   "$OUT/ONLY CA tx less CMC1_sam2.csv"   465 1 65.625
run "$VDIR/ONLY CA tx less CMC2.mp4"   "$OUT/ONLY CA tx less CMC2_sam2.csv"   503 1 65.625
run "$VDIR/ONLY CA tx less CMC3.mp4"   "$OUT/ONLY CA tx less CMC3_sam2.csv"   505 1 65.625
run "$VDIR/ONLY CA cg ABOVE CMC3.mp4"  "$OUT/ONLY CA cg ABOVE CMC3_sam2.csv"  473 4 65.625
run "$VDIR/ca+TR.mp4"                  "$OUT/ca+TR_sam2.csv"                  479 4 65.625

echo "03242026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

# ── 05052026 ──────────────────────────────────────────────────────────────────
VDIR="/ephemeral/videos/05052026"
OUT="results_drops/05052026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 05052026 ==="

run "$VDIR/0.028tx.mp4"         "$OUT/0.028tx_sam2.csv"         462 1 66.0
run "$VDIR/0.028tx2.mp4"        "$OUT/0.028tx2_sam2.csv"        470 1 66.0
run "$VDIR/0.028tx3.mp4"        "$OUT/0.028tx3_sam2.csv"        454 1 66.0
run "$VDIR/0.08cg.mp4"          "$OUT/0.08cg_sam2.csv"          473 1 66.0
run "$VDIR/0.08cg2.mp4"         "$OUT/0.08cg2_sam2.csv"         457 1 66.0
run "$VDIR/0.08cg3.mp4"         "$OUT/0.08cg3_sam2.csv"         454 1 66.0
run "$VDIR/0.08cg4.mp4"         "$OUT/0.08cg4_sam2.csv"         454 1 66.0
run "$VDIR/0.45sds.mp4"         "$OUT/0.45sds_sam2.csv"         470 1 66.0
run "$VDIR/0.45sds2.mp4"        "$OUT/0.45sds2_sam2.csv"        454 1 66.0
run "$VDIR/0.45sds3.mp4"        "$OUT/0.45sds3_sam2.csv"        454 1 66.0
run "$VDIR/cainhcg 0.08 b.mp4"  "$OUT/cainhcg 0.08 b_sam2.csv" 456 1 66.0
run "$VDIR/cainhcg 0.08 c.mp4"  "$OUT/cainhcg 0.08 c_sam2.csv" 458 1 66.0
run "$VDIR/cainhcg 0.08 d.mp4"  "$OUT/cainhcg 0.08 d_sam2.csv" 454 1 66.0
run "$VDIR/cainhcg 0.08.mp4"    "$OUT/cainhcg 0.08_sam2.csv"   454 1 66.0
run "$VDIR/cainhg0.02 .mp4"     "$OUT/cainhg0.02 _sam2.csv"    462 4 66.0
run "$VDIR/cainhg0.02 2.mp4"    "$OUT/cainhg0.02 2_sam2.csv"   458 4 66.0
run "$VDIR/cainhg0.08 4th.mp4"  "$OUT/cainhg0.08 4th_sam2.csv" 449 4 66.0

echo "05052026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

# ── 05112026 — only 2 readable videos ─────────────────────────────────────────
VDIR="/ephemeral/videos/new_videos/05112026"
OUT="results_drops/05112026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 05112026 (2 readable videos) ==="

run "$VDIR/nr50water4.mp4"  "$OUT/nr50water4_sam2.csv"  307 1 66.5
run "$VDIR/water 3.mp4"     "$OUT/water 3_sam2.csv"     304 1 66.5
# remaining 6 videos are corrupted (0 frames) — skipped

echo "05112026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

# ── 05122026 ──────────────────────────────────────────────────────────────────
VDIR="/ephemeral/videos/new_videos/05122026"
OUT="results_drops/05122026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 05122026 ==="

run "$VDIR/0.028tx1.mp4"     "$OUT/0.028tx1_sam2.csv"     300 1 56.0
run "$VDIR/0.45sds1.mp4"     "$OUT/0.45sds1_sam2.csv"     304 1 56.0
run "$VDIR/cain0.45sds.mp4"  "$OUT/cain0.45sds_sam2.csv"  304 1 56.0
run "$VDIR/0.028tx2.mp4"     "$OUT/0.028tx2_sam2.csv"     303 4 56.0
run "$VDIR/0.028tx3.mp4"     "$OUT/0.028tx3_sam2.csv"     305 4 56.0
run "$VDIR/0.45sds2.mp4"     "$OUT/0.45sds2_sam2.csv"     312 4 56.0
run "$VDIR/0.45sds3.mp4"     "$OUT/0.45sds3_sam2.csv"     305 4 56.0
run "$VDIR/cain0.028tx1.mp4" "$OUT/cain0.028tx1_sam2.csv" 302 4 56.0
run "$VDIR/cain0.028tx2.mp4" "$OUT/cain0.028tx2_sam2.csv" 302 4 56.0
run "$VDIR/cain0.08cg1.mp4"  "$OUT/cain0.08cg1_sam2.csv"  325 4 56.0
run "$VDIR/cain0.08cg2.mp4"  "$OUT/cain0.08cg2_sam2.csv"  303 4 56.0
run "$VDIR/cain0.08cg3.mp4"  "$OUT/cain0.08cg3_sam2.csv"  309 4 56.0
run "$VDIR/cain0.45sds2.mp4" "$OUT/cain0.45sds2_sam2.csv" 305 4 56.0
run "$VDIR/cain0.45sds3.mp4" "$OUT/cain0.45sds3_sam2.csv" 303 4 56.0
# cain0.028tx3.mp4 is corrupted — skipped

echo "05122026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

# ── 05172026 ──────────────────────────────────────────────────────────────────
VDIR="/ephemeral/videos/05172026"
OUT="results_drops/05172026_sam2_v3_results"
mkdir -p "$OUT"
echo ""; echo "=== 05172026 ==="

run "$VDIR/cain0.08cg5.mp4"  "$OUT/cain0.08cg5_sam2.csv"  305 1 56.0
run "$VDIR/cain0.08cg6.mp4"  "$OUT/cain0.08cg6_sam2.csv"  304 4 56.0

echo "05172026 done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs"

echo ""
echo "=== ALL SAM2 v3 COMPLETE ==="
echo "Total CSVs produced:"
for d in results_drops/*_sam2_v3_results; do
    echo "  $(basename $d): $(ls $d/*.csv 2>/dev/null | wc -l)"
done
