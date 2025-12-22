#!/bin/bash

# Test script for all optimizers in adversarial_attack.py
# Runs each optimizer with 10 iterations on 3 test samples

set -e  # Exit on error

# Test parameters
NUM_ITERS=10
TEST_SIZE=3
MODEL="cifar10"
OUTPUT_DIR="outputs_grouped/test_optimizers"
POP_SIZE=20  # Small population for quick testing
EPS=0.1
ALPHA=10.0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of all optimizers
OPTIMIZERS=( "info")

echo "================================"
echo "Testing all optimizers"
echo "Model: $MODEL"
echo "Iterations: $NUM_ITERS"
echo "Test size: $TEST_SIZE"
echo "Population: $POP_SIZE"
echo "================================"
echo ""

# Run each optimizer
for optimizer in "${OPTIMIZERS[@]}"; do
    echo "-----------------------------"
    echo "Testing optimizer: $optimizer"
    echo "-----------------------------"
    
    python scripts/adversarial_attack.py \
        --model "$MODEL" \
        --test_size "$TEST_SIZE" \
        --num_iters "$NUM_ITERS" \
        --pop_size "$POP_SIZE" \
        --eps "$EPS" \
        --alpha "$ALPHA" \
        --optimizer "$optimizer" \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ $optimizer completed successfully"
    else
        echo "✗ $optimizer failed"
    fi
    echo ""
done

echo "================================"
echo "All optimizer tests completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
