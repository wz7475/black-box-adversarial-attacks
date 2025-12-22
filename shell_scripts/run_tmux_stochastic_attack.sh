#!/bin/bash

SESSION="stochastic_attack"
BASE_IDX=100
INCREMENT=5
NUM_TABS=40

tmux new-session -d -s $SESSION

for i in $(seq 0 $((NUM_TABS-1))); do
    IDX=$((BASE_IDX + i * INCREMENT))
    CMD="uv run scripts/stochastic_growth_attack.py --test_size 5 --start_from_test_idx $IDX --repeat 1000"
    if [ $i -eq 0 ]; then
        tmux send-keys -t $SESSION "$CMD" C-m
    else
        tmux new-window -t $SESSION
        tmux send-keys -t $SESSION:$i "$CMD" C-m
    fi
done

tmux attach -t $SESSION
