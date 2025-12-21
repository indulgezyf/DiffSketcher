#!/bin/bash

# Simplified pruning test: Baseline vs Pruned for different path counts
# Usage: bash test_pruning_simple.sh [num_paths]

set -e

# Configuration
PROMPT="a photo of Sydney opera house"
TOKEN_IND=4
SEED=8019
NUM_ITER=500

# Allow num_paths to be passed as argument, default to 128
NUM_PATHS=${1:-128}

echo "======================================================"
echo "DiffSketcher Pruning Test (Simplified)"
echo "======================================================"
echo ""
echo "Configuration:"
echo "  - Prompt: ${PROMPT}"
echo "  - Num Paths: ${NUM_PATHS}"
echo "  - Iterations: ${NUM_ITER}"
echo "  - Seed: ${SEED}"
echo ""

# Create test output directory
TEST_DIR="./workdir/pruning_simple_P${NUM_PATHS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Results will be saved to: $TEST_DIR"
echo ""

# ===================================================================
# Test 1: Baseline (No Pruning)
# ===================================================================
echo ">>> Running BASELINE (${NUM_PATHS} paths, pruning disabled)..."
echo ""

python run_painterly_render.py \
  -c diffsketcher.yaml \
  -eval_step 50 -save_step 50 \
  -update "token_ind=${TOKEN_IND} num_paths=${NUM_PATHS} num_iter=${NUM_ITER} prune_loss_weight=0.0" \
  -pt "${PROMPT}" \
  -respath "${TEST_DIR}/baseline" \
  -d ${SEED}

echo ""
echo "✓ Baseline completed"
echo ""

# ===================================================================
# Test 2: With Pruning
# ===================================================================
echo ">>> Running PRUNED VERSION (${NUM_PATHS} paths, pruning enabled)..."
echo ""

python run_painterly_render.py \
  -c diffsketcher.yaml \
  -eval_step 50 -save_step 50 \
  -update "token_ind=${TOKEN_IND} num_paths=${NUM_PATHS} num_iter=${NUM_ITER} prune_start_step=100 prune_ema_threshold=0.02 prune_loss_weight=10.0" \
  -pt "${PROMPT}" \
  -respath "${TEST_DIR}/pruned" \
  -d ${SEED}

echo ""
echo "✓ Pruned version completed"
echo ""

# ===================================================================
# Analysis
# ===================================================================
echo "======================================================"
echo "ANALYZING RESULTS"
echo "======================================================"
echo ""

# Find SVG files
BASELINE_SVG=$(find "${TEST_DIR}/baseline" -name "final_svg_tmp.svg" -path "*/svg_logs/*" | head -1)
PRUNED_SVG=$(find "${TEST_DIR}/pruned" -name "final_svg_tmp.svg" -path "*/svg_logs/*" | head -1)

# Count strokes
BASELINE_STROKES=$(grep -o '<path' "$BASELINE_SVG" | wc -l | xargs)
PRUNED_STROKES=$(grep -o '<path' "$PRUNED_SVG" | wc -l | xargs)

# File sizes
BASELINE_SIZE=$(du -h "$BASELINE_SVG" | cut -f1)
PRUNED_SIZE=$(du -h "$PRUNED_SVG" | cut -f1)

# Calculate reduction
REDUCTION=$(echo "scale=1; (1 - $PRUNED_STROKES/$BASELINE_STROKES) * 100" | bc)

# Display quick summary
echo "Quick Summary:"
echo "──────────────────────────────────────────────────"
echo "                 Baseline    Pruned    Reduction"
echo "──────────────────────────────────────────────────"
printf "Stroke Count:    %-10s  %-10s  %s%%\n" "${BASELINE_STROKES}" "${PRUNED_STROKES}" "${REDUCTION}"
printf "File Size:       %-10s  %-10s\n" "${BASELINE_SIZE}" "${PRUNED_SIZE}"
echo "──────────────────────────────────────────────────"
echo ""

# Save summary to file
cat << EOF > "${TEST_DIR}/SUMMARY.txt"
====================================================
PRUNING TEST SUMMARY
====================================================

Configuration:
- Prompt: ${PROMPT}
- Num Paths: ${NUM_PATHS}
- Iterations: ${NUM_ITER}
- Seed: ${SEED}

Results:
- Baseline Strokes: ${BASELINE_STROKES}
- Pruned Strokes: ${PRUNED_STROKES}
- Reduction: ${REDUCTION}%

File Sizes:
- Baseline: ${BASELINE_SIZE}
- Pruned: ${PRUNED_SIZE}

SVG Files:
- Baseline: ${BASELINE_SVG}
- Pruned: ${PRUNED_SVG}
====================================================
EOF

# ===================================================================
# Run Python Analysis
# ===================================================================
if [ -f "analyze_pruning.py" ]; then
    echo "Running detailed Python analysis..."
    echo ""

    python analyze_pruning.py \
        "$BASELINE_SVG" \
        "$PRUNED_SVG" \
        -o "${TEST_DIR}/analysis.png"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Analysis visualization saved to:"
        echo "   ${TEST_DIR}/analysis.png"
    else
        echo "⚠️  Python analysis failed (matplotlib may be needed)"
        echo "   Install: pip install matplotlib numpy"
    fi
else
    echo "⚠️  analyze_pruning.py not found, skipping visualization"
fi

echo ""
echo "======================================================"
echo "TEST COMPLETE"
echo "======================================================"
echo ""
echo "Results location: ${TEST_DIR}/"
echo ""
echo "Files to check:"
echo "  - ${TEST_DIR}/SUMMARY.txt (text summary)"
echo "  - ${TEST_DIR}/analysis.png (visualization)"
echo "  - ${TEST_DIR}/baseline/.../visual_best.svg"
echo "  - ${TEST_DIR}/pruned/.../visual_best.svg"
echo ""
echo "======================================================"