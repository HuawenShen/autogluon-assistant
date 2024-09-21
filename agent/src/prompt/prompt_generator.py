import argparse
import os
import sys


def read_first_three_lines(file_path):
    try:
        with open(file_path, "r") as file:
            return "".join(file.readlines()[:3])
    except Exception as e:
        return f"Error reading file: {str(e)}"


def read_tutorial_content(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error reading tutorial file: {str(e)}"


def generate_prompt(folder_path, tutorial_folder_path, output_path):
    # Get absolute path of the folder
    abs_folder_path = os.path.abspath(folder_path)

    # Get list of files in the folder
    files = os.listdir(abs_folder_path)

    # Read first three lines of each file
    file_contents = {
        file: read_first_three_lines(os.path.join(abs_folder_path, file))
        for file in files
    }

    # Read Autogluon Tabular Tutorials
    abs_tutorial_folder_path = os.path.abspath(tutorial_folder_path)
    tutorial_files = [
        f for f in os.listdir(abs_tutorial_folder_path) if f.endswith(".md")
    ]
    tutorials_content = {}

    for tutorial_file in tutorial_files:
        tutorial_path = os.path.join(abs_tutorial_folder_path, tutorial_file)
        tutorials_content[tutorial_file] = read_tutorial_content(tutorial_path)

    # Generate the prompt
    prompt = f"""
As an AutoML Agent, you will be given a folder containing data and description files. Please generate Python code using Autogluon Tabular to train a predictor and make predictions on test data. Follow these specifications:

1. Data preprocessing:
   - Remove training data samples without valid labels

2. Model training:
   - Use Autogluon Tabular with the following parameters:
     - time_limit: 3600 seconds
     - presets: 'best_quality'

3. Prediction:
   - Make predictions on the test data
   - Save the predicted results to {output_path}
   - Ensure the output columns match those in the sample submission file
   - No need to save the model

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose and usage
   - Include comments explaining any complex operations or design decisions

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

Absolute path to the folder: {abs_folder_path}
Files in the folder:
{', '.join(files)}
First three lines of each file:
{'-' * 10}
"""
    for file, content in file_contents.items():
        prompt += f"{file}:\n{content}\n{'-' * 10}\n"

    prompt += "Autogluon Tabular Tutorials:\n"
    for tutorial_file, content in tutorials_content.items():
        prompt += f"{tutorial_file}:\n{content}\n{'-' * 10}\n"

    return prompt


def write_prompt_to_file(prompt, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(prompt)
        print(f"Prompt successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AutoGluon Tabular prompt")
    parser.add_argument(
        "-i", "--input_data_folder", required=True, help="Path to the input folder"
    )
    parser.add_argument(
        "-t",
        "--tutorial_path",
        required=True,
        help="Path to the Autogluon Tabular tutorial file",
    )
    parser.add_argument(
        "-o", "--output_result_file", required=True, help="Path for the output file"
    )
    parser.add_argument(
        "-p",
        "--output_prompt_file",
        required=True,
        help="Path for the generated prompt file",
    )

    args = parser.parse_args()

    prompt = generate_prompt(
        args.input_data_folder, args.tutorial_path, args.output_result_file
    )
    write_prompt_to_file(prompt, args.output_prompt_file)
