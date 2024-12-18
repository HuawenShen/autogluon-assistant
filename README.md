# AutoML Code Generation Agent

This repository contains a code generation agent that uses Claude AI to automatically generate AutoGluon-based machine learning code based on input data and tutorials.

## Prerequisites

- Python 3.x
- Conda package manager
- AutoGluon environment (ag20241211)
- Required Python packages (specified in requirements.txt)

## Installation

1. Clone this repository
2. Create and activate the conda environment:
```bash
conda create -n ag20241211
conda activate ag20241211
```

3. Install required dependencies (ensure you have the necessary requirements.txt file)

## File Structure

- `run_agent.py`: Main Python script for code generation
- `run_agent.sh`: Bash script wrapper for executing the agent
- `agent/src/coding_agent.py`: Contains the core code generation logic

## Usage

### Using the Python Script

```bash
python run_agent.py [-h] -i INPUT_DATA_FOLDER [-b BACKEND] [-t TUTORIAL_PATH] 
                    [-l TUTORIAL_LINK] -w RESULT_DIR [-m MODEL_ID]
```

Arguments:
- `-i, --input_data_folder`: (Required) Path to the input data folder
- `-b, --backend`: (Optional) Backend to use (default: "bedrock")
- `-t, --tutorial_path`: (Optional) Path to the AutoGluon Tabular tutorial file
- `-l, --tutorial_link`: (Optional) URL link to the AutoGluon Tabular tutorials
- `-w, --result_dir`: (Required) Path for the output folder
- `-m, --model_id`: (Optional) Claude model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")

### Using the Shell Script

```bash
./run_agent.sh <input_data_folder> <tutorial_path> <output_dir> <model_id> <backend>
```

Example:
```bash
./run_agent.sh /path/to/data /path/to/tutorial /path/to/output anthropic.claude-3-haiku-20240307-v1:0 bedrock
```

## Output Files

The agent generates several files in the specified output directory:
- `prompt.txt`: Contains the generated prompt used for code generation
- `generated_code.py`: The generated AutoGluon code
- `log.txt`: Execution log of the agent
- `generated_code_log.txt`: Execution log of the generated code

## Notes

- Ensure you have appropriate permissions for executing the shell script (`chmod +x run_agent.sh`)
- The backend parameter defaults to "bedrock" if not specified, can also be "agrag"
- All output files are automatically saved in the specified result directory
