#!/bin/bash

# Three-way benchmark: Baseline vs Dynamic Weights vs Dynamic Weights + Curriculum
# Usage: bash benchmark_test_3way.sh

set -e  # Exit on error

# ===== Configuration =====
PROMPT="a photo of Sydney opera house"
NUM_ITER=500
NUM_PATHS=96
SEED=8019
TOKEN_IND=4

# Output directories
BASELINE_DIR="./workdir/baseline_test"
DYNAMIC_WEIGHTS_DIR="./workdir/dynamic_weights_test"
FULL_IMPROVED_DIR="./workdir/full_improved_test"

echo "============================================"
echo "DiffSketcher Three-Way Benchmark Test"
echo "============================================"
echo "Prompt: $PROMPT"
echo "Iterations: $NUM_ITER"
echo "Num Paths: $NUM_PATHS"
echo "Seed: $SEED"
echo "Token Index: $TOKEN_IND"
echo "============================================"
echo ""

# ===== Run Baseline =====
echo "🔹 Running BASELINE version (no improvements)..."
echo ""

time python run_painterly_render.py \
  -c diffsketcher-baseline.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
  -pt "$PROMPT" \
  -respath "$BASELINE_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$BASELINE_DIR/training.log"

echo ""
echo "✅ Baseline complete!"
echo ""

# ===== Run Dynamic Weights Only =====
echo "🔥 Running DYNAMIC WEIGHTS ONLY version..."
echo ""

time python run_painterly_render.py \
  -c diffsketcher-dynamic-weights.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
  -pt "$PROMPT" \
  -respath "$DYNAMIC_WEIGHTS_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$DYNAMIC_WEIGHTS_DIR/training.log"

echo ""
echo "✅ Dynamic weights version complete!"
echo ""

# ===== Run Full Improved Version =====
echo "🚀 Running FULL IMPROVED version (Dynamic Weights + Curriculum)..."
echo ""

time python run_painterly_render.py \
  -c diffsketcher-improved.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
  -pt "$PROMPT" \
  -respath "$FULL_IMPROVED_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$FULL_IMPROVED_DIR/training.log"

echo ""
echo "✅ Full improved version complete!"
echo ""

echo "============================================"
echo "All three experiments completed!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  - Baseline:        $BASELINE_DIR"
echo "  - Dynamic Weights: $DYNAMIC_WEIGHTS_DIR"
echo "  - Full Improved:   $FULL_IMPROVED_DIR"
echo ""
echo "Run analysis with:"
echo "  python analyze_results_3way.py"
echo ""
