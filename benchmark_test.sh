#!/bin/bash

# Benchmark script to compare baseline vs improved DiffSketcher
# Usage: bash benchmark_test.sh

set -e  # Exit on error

# ===== Configuration =====
PROMPT="a photo of Sydney opera house"
NUM_ITER=500
NUM_PATHS=96
SEED=8019
TOKEN_IND=4

# Output directories
BASELINE_DIR="./workdir/baseline_test"
IMPROVED_DIR="./workdir/improved_test"

echo "============================================"
echo "DiffSketcher Benchmark Test"
echo "============================================"
echo "Prompt: $PROMPT"
echo "Iterations: $NUM_ITER"
echo "Num Paths: $NUM_PATHS"
echo "Seed: $SEED"
echo "Token Index: $TOKEN_IND"
echo "============================================"
echo ""

# ===== Run Baseline =====
echo "Running BASELINE version..."

time python run_painterly_render.py \
  -c diffsketcher.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
  -pt "$PROMPT" \
  -respath "$BASELINE_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$BASELINE_DIR/training.log"

echo ""
echo "Baseline complete!"
echo ""

# ===== Run Improved Version =====
echo "Running IMPROVED version (Curriculum + Dynamic Weights)..."

time python run_painterly_render.py \
  -c diffsketcher-improved.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
  -pt "$PROMPT" \
  -respath "$IMPROVED_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$IMPROVED_DIR/training.log"

echo ""
echo "Improved version complete!"
echo ""

