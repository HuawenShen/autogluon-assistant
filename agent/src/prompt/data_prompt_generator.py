import argparse
import os
import sys
from collections import defaultdict
import pandas as pd
from autogluon.core.utils.loaders import load_pd
from pathlib import Path


def group_similar_files(files):
    """
    Group files based on their extensions dynamically.
    Parameters:
        files: List of tuples (relative_path, absolute_path)
    """
    patterns = defaultdict(list)

    for rel_path, _ in files:
        dirname = os.path.dirname(rel_path)
        ext = os.path.splitext(rel_path)[1].lower()

        if ext:  # If file has an extension
            pattern = f"*{ext}"
            if dirname:
                pattern = os.path.join(dirname, pattern)
            patterns[pattern].append(rel_path)
        else:  # Files without extensions
            patterns[rel_path].append(rel_path)

    return patterns


def read_first_three_lines(file_path, max_length=100):
    """
    Read first three lines of a text file with maximum length limit.
    Only shows error message for text files, silently returns empty string for binary files.
    """
    # Common text file extensions and types
    text_extensions = {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".conf",
        ".sh",
        ".bat",
        ".log",
    }

    # Check if file might be text based on extension
    ext = os.path.splitext(file_path)[1].lower()

    # Try to detect if file is text by reading first few bytes
    try:
        with open(file_path, "rb") as file:
            is_text = True
            try:
                file.read(1024).decode("utf-8")
            except UnicodeDecodeError:
                is_text = False
    except Exception:
        return ""  # Return empty string for unreadable files

    # If file is likely text, try to read it
    if is_text or ext in text_extensions:
        try:
            content = []
            total_length = 0
            with open(file_path, "r", encoding="utf-8") as file:
                for _ in range(3):
                    line = file.readline()
                    if not line:
                        break
                    if total_length + len(line) > max_length:
                        remaining = max_length - total_length
                        content.append(line[:remaining] + "...")
                        break
                    content.append(line)
                    total_length += len(line)
            return "".join(content)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    else:
        # For binary files, return empty string
        return "Not text file."


def get_tabular_data_info(file_path):
    """
    Load tabular data and return column information and first two rows.
    """
    try:
        df = load_pd.load(file_path)

        # Get column information
        all_columns = df.columns.tolist()
        if len(all_columns) > 10:
            display_columns = all_columns[:5] + all_columns[-5:]
            columns_info = f"First 5 and last 5 columns (total {len(all_columns)}): {display_columns}"
        else:
            columns_info = f"Columns: {all_columns}"

        # Get first two rows
        first_two_rows = df.head(2).to_string()

        return f"{columns_info}\nFirst two rows:\n{first_two_rows}"
    except Exception as e:
        return f"Error reading tabular data: {str(e)}"


def is_tabular_file(file_path):
    """
    Check if file is a tabular data file based on extension.
    """
    tabular_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls"}
    return Path(file_path).suffix.lower() in tabular_extensions


def get_all_files(folder_path):
    """
    Recursively get all files in the folder and its subfolders.
    Returns a list of tuples containing (relative_path, absolute_path).
    """
    all_files = []
    abs_folder_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, abs_folder_path)
            all_files.append((rel_path, abs_path))

    return all_files


def read_full_tutorial(tutorial_path, max_length=None):
    """
    Read the complete content of a markdown tutorial file.

    Args:
        tutorial_path (str): Path to the tutorial markdown file
        max_length (int, optional): Maximum number of characters to read

    Returns:
        str: The complete content of the tutorial file
    """
    with open(tutorial_path, "r", encoding="utf-8") as file:
        if max_length:
            content = file.read(max_length)
        else:
            content = file.read()
    return content


def generate_data_prompt(
    folder_path, tutorial_folder_path, output_folder, max_length=100
):
    # Get absolute path of the folder
    abs_folder_path = os.path.abspath(folder_path)

    # Get list of all files recursively
    all_files = get_all_files(abs_folder_path)

    # Group similar files
    file_groups = group_similar_files(all_files)

    # Create a mapping of relative paths to absolute paths
    path_mapping = dict(all_files)

    # Process files based on their groups and types
    file_contents = {}
    for pattern, group_files in file_groups.items():
        if len(group_files) > 5:
            # For large groups, only show one example
            example_file = group_files[0]
            file_path = path_mapping[example_file]
            group_info = f"Group pattern: {pattern} (total {len(group_files)} files)\nExample file: {example_file}"

            if is_tabular_file(file_path):
                content = get_tabular_data_info(file_path)
            else:
                content = read_first_three_lines(file_path, max_length)

            file_contents[group_info] = content
        else:
            # For small groups, show all files
            for file in group_files:
                file_path = path_mapping[file]
                if is_tabular_file(file_path):
                    file_contents[file] = get_tabular_data_info(file_path)
                else:
                    file_contents[file] = read_first_three_lines(file_path, max_length)

    # Read tutorials content
    tutorials_content = {}
    if tutorial_folder_path:
        abs_tutorial_folder_path = os.path.abspath(tutorial_folder_path)
        tutorial_files = [
            f for f in os.listdir(abs_tutorial_folder_path) if f.endswith(".md")
        ]
        for tutorial_file in tutorial_files:
            tutorial_path = os.path.join(abs_tutorial_folder_path, tutorial_file)
            tutorials_content[tutorial_file] = read_full_tutorial(
                tutorial_path, max_length=None
            )

    # Generate the prompt
    prompt = f"""
As an AutoML Agent, you will be given a folder containing data and description files. Please generate Python code using Autogluon Multimodal to train a predictor and make predictions on test data. Follow these specifications:

1. Data preprocessing:
   - Remove training data samples without valid labels
   - Remove the unneccesary index column (if applicable)

2. Model training:
   - Use Autogluon Multimodal with the following parameters:
     - time_limit: 3600 seconds
     - presets: 'best_quality'
     - tuning_data: only use validation if there is a validation dataset

3. Prediction:
   - Make predictions on the test data
   - Save the predicted results to {output_folder}, result file name should be "results", the extension should be same as the test data file
   - Save the model in a folder under {output_folder}
   - Ensure the output columns match what in the training file, or those in the sample submission file (if any). Do not change any column names.

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose and usage
   - Include comments explaining any complex operations or design decisions

5. Others:
   - To avoid DDP errors, wrap the code in: if __name__ == "__main__":

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

Absolute path to the folder: {abs_folder_path}
Files analysis:
{'-' * 10}
"""
    for file, content in file_contents.items():
        prompt += f"{file}:\n{content}\n{'-' * 10}\n"

    if tutorials_content:
        prompt += "Autogluon Tutorials:\n"
        for tutorial_file, content in tutorials_content.items():
            prompt += f"{tutorial_file}:\n{content}\n{'-' * 10}\n"
    else:
        print(
            "Warning: Tutorial is not provided. Please provide a tutorial or use agrag as backend to retrieve the tutorial from web."
        )

    return prompt


def write_prompt_to_file(prompt, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(prompt)
        print(f"Prompt successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AutoGluon prompt")
    parser.add_argument(
        "-i", "--input_data_folder", required=True, help="Path to the input folder"
    )
    parser.add_argument(
        "-t",
        "--tutorial_path",
        default=None,
        help="Path to the Autogluon Tabular tutorial file",
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, help="Path for the output file"
    )
    parser.add_argument(
        "-p",
        "--output_prompt_file",
        required=True,
        help="Path for the generated prompt file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length for text file preview",
    )

    args = parser.parse_args()

    prompt = generate_data_prompt(
        args.input_data_folder,
        args.tutorial_path,
        args.output_folder,
        args.max_length,
    )
    write_prompt_to_file(prompt, args.output_prompt_file)
