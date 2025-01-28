#!/bin/bash

# Default environment name
DEFAULT_ENV="ag20250103"

# Parse command line arguments
if [ $# -lt 6 ]; then
    echo "Usage: $0 input_data_folder tutorial_path output_dir model_id backend config_path [conda_env]"
    exit 1
fi

input_data_folder="$1"
tutorial_path="$2"
output_dir="$3"
model_id="$4"
backend="$5"
config_path="$6"
conda_env="${7:-$DEFAULT_ENV}"  # Use default if not provided

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$conda_env"; then
    echo "Error: Failed to activate conda environment $conda_env"
    exit 1
fi

# Run the agent with code generation and execution
python3 /media/deephome/AutoMLAgent/run_agent.py \
    -i "$input_data_folder" \
    -t "$tutorial_path" \
    -o "$output_dir" \
    -m "$model_id" \
    -b "$backend" \
    -c "$config_path" \
    -n 3 \
    2>&1 | tee "${output_dir}/log.txt"

# Check if the agent run was successful
if [ $? -ne 0 ]; then
    echo "Error: Code generation and execution failed. Please check the log file."
    conda deactivate
    exit 1
fi

echo "Process completed successfully!"
conda deactivate