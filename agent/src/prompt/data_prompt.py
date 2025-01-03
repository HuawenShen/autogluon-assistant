import os
from collections import defaultdict
from pathlib import Path

from autogluon.core.utils.loaders import load_pd


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


def is_tabular_file(file_path):
    """
    Check if file is a tabular data file based on extension.
    """
    tabular_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls"}
    return Path(file_path).suffix.lower() in tabular_extensions


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


def get_file_content(file_path, max_chars_per_file):
    if is_tabular_file(file_path):
        content = get_tabular_data_info(file_path)
    else:
        content = read_first_three_lines(
            file_path=file_path, max_length=max_chars_per_file
        )
    return content


def generate_data_prompt(input_data_folder, max_chars_per_file):
    # Get absolute path of the folder
    abs_folder_path = os.path.abspath(input_data_folder)

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
            group_info = f"Group pattern: {os.path.join(abs_folder_path, pattern)} (total {len(group_files)} files)\nExample file: {example_file}"

            file_contents[group_info] = get_file_content(
                file_path=file_path, max_chars_per_file=max_chars_per_file
            )
        else:
            # For small groups, show all files
            for file in group_files:
                file_path = path_mapping[file]

                file_contents[file] = get_file_content(
                    file_path=file_path, max_chars_per_file=max_chars_per_file
                )

    # Generate the prompt
    prompt = f"Absolute path to the folder: {abs_folder_path}\n\nFiles structures:\n\n{'-' * 10}\n\n"
    for file, content in file_contents.items():
        prompt += f"{file}:\n{content}\n{'-' * 10}\n"

    return prompt
