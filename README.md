# AutoML Agent

This repository contains an automated machine learning agent that generates and executes code based on input data and tutorials of third party tools.

## Prerequisites

- Python 3.8+
- Conda package manager
- AutoGluon dependencies
- Access to Bedrock/OpenAI API

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AutoMLAgent
```

2. Create and activate the Conda environment:
TBA

3. Configure your environment variables:
Similar to Autogluon Assistant


## Usage

### Using the Shell Script

The easiest way to run the agent is using the provided shell script:

```bash
./run_agent.sh \
  <input_data_folder> \
  <tutorial_path> \
  <output_dir> \
  <model_id> \
  <backend> \
  <config_path>
```

Arguments:
- `input_data_folder`: Path to the folder containing input data
- `tutorial_path`: Path to the folder of (AutoGluon) tutorial files
- `output_dir`: Directory for output files
- `model_id`: LLM model ID
- `backend`: LLM Backend (bedrock or openai)
- `config_path`: Path to configuration file

### Using Python Script Directly

You can also run the Python script directly:

```bash
python run_agent.py \
  -i <input_data_folder> \
  -t <tutorial_path> \
  -o <output_folder> \
  -c <output_code_file> \
  -f <config_path> \
  -p <output_prompt_file> \
  -m <model_id>
```

## Output Files

The agent generates several output files:
- `generated_code.py`: The generated Python script
- `log.txt`: Execution log of the agent
- `generated_code_log.txt`: Execution log of the generated code
- `retrieved_context.txt`: Retrieved context information (if available)
- ... (TBA)

## Example

```bash
./run_agent.sh \
  ./data/input \
  ./tutorials/tabular_prediction.md \
  ./output \
  anthropic.claude-3-haiku-20240307-v1:0 \
  default \
  ./configs/default.yaml
```


## Configuration
TBA
