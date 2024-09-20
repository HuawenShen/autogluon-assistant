#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
for cmd in python aws; do
    if ! command_exists "$cmd"; then
        echo "Error: $cmd is not installed or not in PATH. Please install it and try again."
        exit 1
    fi
done

# Check for required Python libraries
for lib in boto3; do
    if ! python -c "import $lib" &>/dev/null; then
        echo "Error: Python library $lib is not installed. Please install it using 'pip install $lib' and try again."
        exit 1
    fi
done

# Check if AWS credentials are set
if [ -z "$BEDROCK_ACCESS_KEY" ] || [ -z "$BEDROCK_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS credentials are not set. Please set BEDROCK_ACCESS_KEY and BEDROCK_SECRET_ACCESS_KEY environment variables."
    exit 1
fi

# Check if all required arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <folder_path> <tutorial_path> <output_path> <model_id>"
    echo "  <input_data_folder>: Path to the folder containing problem files"
    echo "  <tutorial_path>: Path to the Autogluon Tabular tutorial file"
    echo "  <output_prompt_file>: Desired output path for the generated prompt"
    echo "  <output_code_file>: Desired output path for the generated code"
    echo "  <output_result_file>: Desired output for the result"
    echo "  <model_id>: Claude model ID for AWS Bedrock"
    exit 1
fi

input_data_folder="$1"
tutorial_path="$2"
output_prompt_file="$3"
output_code_file="$4"
output_result_file="$5"
model_id="$6"

# Generate prompt
echo "Generating prompt..."
python /media/deephome/AutoMLAgent/agent/src/prompt_generator.py \
    -i "$input_data_folder" \
    -t "$tutorial_path" \
    -o "$output_result_file" \
    -p "$output_prompt_file"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate prompt. Please check your inputs and try again."
    exit 1
fi

# Generate code with AWS Bedrock
echo "Generating code with AWS Bedrock..."
python /media/deephome/AutoMLAgent/agent/src/bedrock_code_generator.py \
    -p "$output_prompt_file" \
    -c "$output_code_file" \
    -m "$model_id"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate code. Please check your AWS Bedrock setup and try again."
    exit 1
fi

# Run the generated code
echo "Running the generated code..."
python "$output_code_file"

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the generated code. Please check the output file for any issues."
    exit 1
fi

echo "Process completed successfully!"
