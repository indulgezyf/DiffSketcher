#!/bin/bash

# DSSP Four-Way Comparison Test
# Compares: Baseline vs Directional CLIP vs Saliency Pruning vs Both Combined

set -e  # Exit on error

# ===== Configuration =====
PROMPT="a photo of Sydney opera house"
STYLE_PROMPT="a charcoal sketch of Sydney opera house"
NUM_ITER=500
NUM_PATHS=96
SEED=8019
TOKEN_IND=4

# Output directories
BASELINE_DIR="./workdir/baseline_test"
DIRECTIONAL_DIR="./workdir/directional_test"
PRUNING_DIR="./workdir/pruning_test"
DSSP_DIR="./workdir/dssp_test"

echo "============================================"
echo "DSSP Four-Way Comparison Test"
echo "============================================"
echo "Prompt:       $PROMPT"
echo "Style:        $STYLE_PROMPT"
echo "Iterations:   $NUM_ITER"
echo "Num Paths:    $NUM_PATHS"
echo "Seed:         $SEED"
echo "Token Index:  $TOKEN_IND"
echo "============================================"
echo ""

mkdir -p $BASELINE_DIR
mkdir -p $DIRECTIONAL_DIR
mkdir -p $PRUNING_DIR
mkdir -p $DSSP_DIR
# # ===== Test 1: Baseline =====
# echo "🔹 Test 1/4: Running BASELINE version (no improvements)..."
# echo ""

# time python run_painterly_render.py \
#   -c diffsketcher-baseline.yaml \
#   -eval_step 10 \
#   -save_step 50 \
#   -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
#   -pt "$PROMPT" \
#   -respath "$BASELINE_DIR" \
#   -d $SEED \
#   --download \
#   2>&1 | tee "$BASELINE_DIR/training.log"

# echo ""
# echo "✅ Baseline complete!"
# echo ""

# ===== Test 2: Directional CLIP Only =====
echo "🎨 Test 2/4: Running DIRECTIONAL CLIP only..."
echo ""

time python run_painterly_render.py \
  -c diffsketcher-directional.yaml \
  -eval_step 10 \
  -save_step 50 \
  -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER style_prompt=$STYLE_PROMPT" \
  -pt "$PROMPT" \
  -respath "$DIRECTIONAL_DIR" \
  -d $SEED \
  --download \
  2>&1 | tee "$DIRECTIONAL_DIR/training.log"

echo ""
echo "✅ Directional CLIP complete!"
echo ""

# # ===== Test 3: Saliency Pruning Only =====
# echo "✂️  Test 3/4: Running SALIENCY PRUNING only..."
# echo ""

# time python run_painterly_render.py \
#   -c diffsketcher-pruning.yaml \
#   -eval_step 10 \
#   -save_step 50 \
#   -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER" \
#   -pt "$PROMPT" \
#   -respath "$PRUNING_DIR" \
#   -d $SEED \
#   --download \
#   2>&1 | tee "$PRUNING_DIR/training.log"

# echo ""
# echo "✅ Saliency pruning complete!"
# echo ""

# # ===== Test 4: DSSP (Both Combined) =====
# echo "🚀 Test 4/4: Running DSSP (Directional CLIP + Saliency Pruning)..."
# echo ""

# time python run_painterly_render.py \
#   -c diffsketcher-dssp.yaml \
#   -eval_step 10 \
#   -save_step 50 \
#   -update "token_ind=$TOKEN_IND num_paths=$NUM_PATHS num_iter=$NUM_ITER style_prompt=$STYLE_PROMPT" \
#   -pt "$PROMPT" \
#   -respath "$DSSP_DIR" \
#   -d $SEED \
#   --download \
#   2>&1 | tee "$DSSP_DIR/training.log"

# echo ""
# echo "✅ DSSP complete!"
# echo ""
