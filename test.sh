#!/bin/bash

python main.py \
    -g "/path/to/genotype/file" \
    --num_bin 8 \
    --sigma_list [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] \
    -k 10 \
    -jn 1 \
    --seed 0
