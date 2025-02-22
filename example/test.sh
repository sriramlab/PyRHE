#!/bin/bash


# Check if the first argument is "--config" and if a second argument exists.
if [ "$1" = "--config" ] && [ -n "$2" ]; then
    CONFIG_FILE="$2"
else
    echo "Usage: $0 --config <path_to_config_file>"
    exit 1
fi

python ../run_rhe.py --config "$CONFIG_FILE"
