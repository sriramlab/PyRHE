#!/bin/bash

MODEL="genie"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)
        shift
        MODEL="$1"
        shift
        ;;
        *)
        shift
        ;;
    esac
done

CONFIG_FILE="./${MODEL}_config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file for model '$MODEL' not found: $CONFIG_FILE"
    exit 1
fi

python ../run_rhe.py --config "$CONFIG_FILE"
