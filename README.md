# AutoMLAgent

## Prompt Generator (`prompt_generator.py`)

This script generates a prompt for an AutoML LLM Agent to create Python code using Autogluon Tabular. It analyzes files in a specified folder and incorporates information from an Autogluon Tabular tutorial to produce a comprehensive prompt.

### Usage

```
python prompt_generator.py <folder_path> <tutorial_path> <output_path>
```

- `<folder_path>`: Path to the folder containing the problem files
- `<tutorial_path>`: Path to the Autogluon Tabular tutorial file
- `<output_path>`: Desired output path for the generated code

### Functionality

1. Reads the first three lines of each file in the specified folder
2. Incorporates the content of the Autogluon Tabular tutorial
3. Generates a prompt that includes:
   - Absolute path to the folder
   - List of files in the folder
   - First three lines of each file
   - Autogluon Tabular tutorial content
4. Writes the generated prompt to `generated_prompt.txt` in the current directory

### Output

The script produces a text file (`generated_prompt.txt`) containing the generated prompt. This prompt can be used to instruct an AutoML LLM Agent to create Python code that solves the problem presented in the given folder using Autogluon Tabular.

### Error Handling

The script includes error handling for file reading and writing operations, ensuring graceful failure in case of issues accessing the specified files or directories.

### Note

Ensure that you have the necessary permissions to read from the specified folder and tutorial file, and to write to the current directory for the output prompt file.


## Generate Code with AWS Bedrock

This Python script interacts with AWS Bedrock to generate Python code using Claude AI models and extract it from the response.

### Prerequisites

- Python 3.6 or higher
- `boto3` library
- AWS account with Bedrock access
- Claude model available in your Bedrock setup

### Setup

1. Install the required library:
   ```
   pip install boto3
   ```

2. Set up your AWS credentials as environment variables:
   ```
   export BEDROCK_ACCESS_KEY=your_access_key
   export BEDROCK_SECRET_ACCESS_KEY=your_secret_access_key
   ```

### Usage

Run the script with the following command:

```
python script_name.py --prompt_file path/to/prompt.txt --output_file path/to/output_script.py --model_id your_claude_model_id
```

- `-p, --prompt_file`: Path to a text file containing your prompt for Claude
- `-o, --output_file`: Path where the extracted Python script will be saved
- `-m, --model_id`: The ID of the Claude model you want to use (e.g., `anthropic.claude-3-haiku-20240307-v1:0`)

### Example

```
python generate_code_with_bedrock.py -p my_prompt.txt -o generated_script.py
```

This will send the contents of `my_prompt.txt` to the specified Claude model, extract any Python code from the response, and save it to `generated_script.py`.

### Note

Ensure you have the correct permissions and model access in your AWS Bedrock setup. If you encounter any issues with the model ID, check your AWS Bedrock console or contact your AWS account manager for the correct model identifiers available to you.
