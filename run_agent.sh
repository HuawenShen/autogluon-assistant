#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ag20241211

input_data_folder="$1"
tutorial_path="$2"
output_dir="$3"
model_id="$4"
backend="$5"

python3 /media/deephome/AutoMLAgent/run_agent.py \
    -i $input_data_folder \
    -t $tutorial_path \
    -w $output_dir \
    -m $model_id \
    -b $backend \
    2>&1 | tee "${output_dir}/log.txt"

# Run the generated code
echo "Running the generated code..."
python "${output_dir}/generated_code.py" 2>&1 | tee "${output_dir}/generated_code_log.txt"
if [ $? -ne 0 ]; then
    echo "Error: Failed to run the generated code. Please check the output file for any issues."
    exit 1
fi
echo "Process completed successfully!"

conda deactivate
