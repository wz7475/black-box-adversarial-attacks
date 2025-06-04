#!/usr/bin/env bash

for model in mnist cifar10; do
  echo "Starting loop for model: $model"
  for eps in 0.01 0.1 0.2; do
    echo "  Starting loop for eps: $eps"
    for alpha in 1 10 100; do
      echo "    Starting loop for alpha: $alpha"
      for optimizer in gen jade; do
        echo "      Running: model=$model eps=$eps alpha=$alpha optimizer=$optimizer"
        python main.py --model $model --eps $eps --alpha $alpha --optimizer $optimizer --test_size 100
      done
    done
  done
done

