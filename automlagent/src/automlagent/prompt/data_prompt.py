import argparse
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
    Group files based on their folder structure and extensions.
    Files are placed in the same group if they follow the same pattern at each level.
    At each level, if there are 5 or fewer unique folders, the actual folder names are used,
    otherwise a wildcard '*' is used.
    
    Parameters:
    files: List of tuples (relative_path, absolute_path)
    
    Returns: 
    Dict mapping group keys to lists of tuples (relative_path, absolute_path)
    """
    # First, analyze folder counts at each depth level
    depth_folders = defaultdict(set)
    max_depth = 0
    
    # Collect all unique folders at each depth
    for rel_path, _ in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        max_depth = max(max_depth, len(parts) - 1)  # -1 for filename
        
        # Record folders at each depth
        for depth, folder in enumerate(parts[:-1]):  # Exclude filename
            depth_folders[depth].add(folder)
    
    # Create groups
    groups = defaultdict(list)
    for rel_path, abs_path in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        filename = parts[-1]
        folders = parts[:-1]
        
        # Get file extension (if any)
        ext = os.path.splitext(filename)[1].lower()
        
        # Build group key parts
        group_key_parts = []
        
        # Add folder pattern for each depth
        for depth, folder in enumerate(folders):
            unique_folders = depth_folders[depth]
            if len(unique_folders) <= 5:
                # Use actual folder name if 5 or fewer unique folders at this depth
                group_key_parts.append(folder)
            else:
                # Use wildcard if more than 5 unique folders
                group_key_parts.append('*')
        
        # Add extension pattern
        group_key_parts.append(ext if ext else 'NO_EXT')
        
        # Convert key to immutable tuple for dictionary
        group_key = tuple(group_key_parts)
        groups[group_key].append((rel_path, abs_path))
    
    return groups


def pattern_to_path(pattern, base_path):
    """
    Convert a group pattern tuple to an absolute path string.
    Pattern tuple format: (folder1, folder2, ..., extension)
    
    Parameters:
    pattern: Tuple of folder names and extension
    base_path: Base directory path to make the pattern absolute
    """
    # Last element is extension
    folders = pattern[:-1]  # Get all folder patterns
    ext = pattern[-1]
    
    # Create a path-like string from folder patterns
    path_parts = []
    for folder in folders:
        path_parts.append(str(folder))
    
    # Add a placeholder filename with the extension
    if ext == 'NO_EXT':
        path_parts.append('*')
    else:
        path_parts.append(f'*{ext}')
    
    # Join with base path to make it absolute
    relative_pattern = os.path.join(*path_parts)
    return os.path.join(base_path, relative_pattern)


def is_tabular_file(file_path):
    """
    Check if file is a tabular data file based on extension.
    Also checks if it's a gzipped tabular file.
    """
    # TODO: test suppot for arrow, and compressed files
    tabular_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".arrow"}
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    # Direct check for tabular extensions
    if suffix in tabular_extensions:
        return True
    
    # Check for compressed tabular files
    if suffix in [".gz", ".zip"]:
        # Get the extension before compression
        stem = Path(path.stem)  # Get filename without compression
        previous_suffix = stem.suffix.lower()
        return previous_suffix in tabular_extensions
        
    return False


def should_truncate(text, threshold=50):
    """
    Determine if text should be truncated based on criteria:
    - Must be string type
    - Longer than threshold
    - Not a file path or relative path
    """
    if not isinstance(text, str):
        return False
    if len(text) <= threshold:
        return False
    # Check if text matches the pattern xxx/xxx.xxx
    import re
    if re.match(r'^[^/]+(/[^/]+)*\.[^/]+$', text):
        return False
    return True


def truncate_text(text, max_length=50):
    """
    Truncate text to specified length and add ellipsis
    """
    return text[:max_length] + "..."


def format_value(value):
    """
    Format a single value, truncating if necessary
    """
    str_value = str(value)
    if should_truncate(str_value):
        return truncate_text(str_value)
    return str_value


def detect_separator(file_path):
    """
    Detect the separator used in a tabular file.
    Returns the detected separator or comma as default.
    """
    import csv
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            # Read a few lines to detect the dialect
            sample = csvfile.read(1024)
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
    except Exception:
        # Default to comma if detection fails
        return ','


def get_tabular_data_info(file_path, max_chars_per_tabular_to_text):
    """
    Load tabular data and return column information and first two rows.
    Preserves original file separator and truncates long text content that isn't a path.
    """
    try:
        # For CSV files, detect the separator first
        if Path(file_path).suffix.lower() == '.csv':
            separator = detect_separator(file_path)
            df = load_pd.load(file_path, delimiter=separator)
        else:
            # For other formats (parquet, excel etc.), load normally
            df = load_pd.load(file_path)
            # Use the same separator as detected in CSV files for consistency in output
            separator = ',' 
        
        # Get column information
        all_columns = df.columns.tolist()
        if len(all_columns) > 20:
            display_columns = all_columns[:10] + all_columns[-10:]
            columns_info = f"First 10 and last 10 columns (total {len(all_columns)}): {display_columns}"
        else:
            columns_info = f"Columns: {all_columns}"

        # Get first two rows
        first_two = df.head(2)
        
        # Convert to formatted strings with truncation
        formatted_rows = []
        
        # Format column headers
        formatted_rows.append(separator.join(str(col) for col in first_two.columns))
        
        # Format data rows
        for _, row in first_two.iterrows():
            formatted_row = separator.join(format_value(val) for val in row)
            formatted_rows.append(formatted_row)
            
        first_two_rows = "\n".join(formatted_rows)
        
        return f"{columns_info}\nFirst two rows:\n{first_two_rows}"
    
    except Exception as e:
        return read_first_three_lines(
            file_path=file_path, max_length=max_chars_per_tabular_to_text
        )


def read_first_three_lines(file_path, max_length=100):
    """
    Read first three lines of a text file with maximum length limit.
    Only shows error message for text files, silently returns empty string for binary files.
    """
    # Common text file extensions and types
    text_extensions = {
        ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".ini",
        ".cfg", ".conf", ".sh", ".bat", ".log",
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
                for i in range(3 + 1):  # if there are more than 3 lines, add "..."
                    line = file.readline()
                    if not line:
                        break
                    if i == 3:
                        content.append("...")
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
        return ""


def get_file_content(file_path, max_chars_per_file, max_chars_per_tabular_to_text):
    if is_tabular_file(file_path):
        content = get_tabular_data_info(file_path, max_chars_per_tabular_to_text)
    else:
        content = read_first_three_lines(
            file_path=file_path, max_length=max_chars_per_file
        )
    return content


def generate_data_prompt(input_data_folder, max_chars_per_file, max_chars_per_tabular_to_text):
    # Get absolute path of the folder
    abs_folder_path = os.path.abspath(input_data_folder)

    # Get list of all files recursively
    all_files = get_all_files(abs_folder_path)

    # Group similar files
    file_groups = group_similar_files(all_files)

    # Process files based on their groups and types
    file_contents = {}
    for pattern, group_files in file_groups.items():
        if len(group_files) > 5:
            # For large groups, only show one example
            example_rel_path, example_abs_path = group_files[0]
            pattern_path = pattern_to_path(pattern, abs_folder_path)
            group_info = f"Group pattern: {pattern_path} (total {len(group_files)} files)\nExample file:\nAbsolute path: {example_abs_path}"

            file_contents[group_info] = get_file_content(
                file_path=example_abs_path, 
                max_chars_per_file=max_chars_per_file, 
                max_chars_per_tabular_to_text=max_chars_per_tabular_to_text
            )
        else:
            # For small groups, show all files
            for rel_path, abs_path in group_files:
                file_info = f"Absolute path: {abs_path}"
                file_contents[file_info] = get_file_content(
                    file_path=abs_path, 
                    max_chars_per_file=max_chars_per_file, 
                    max_chars_per_tabular_to_text=max_chars_per_tabular_to_text
                )

    # Generate the prompt
    prompt = f"Absolute path to the folder: {abs_folder_path}\n\nFiles structures:\n\n{'-' * 10}\n\n"
    for file_info, content in file_contents.items():
        prompt += f"{file_info}\nContent:\n{content}\n{'-' * 10}\n"

    return prompt


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate data prompt from folder contents')
    
    # Add arguments
    parser.add_argument('folder_path', 
                       type=str,
                       help='Path to the folder to analyze')
    parser.add_argument('--max-chars-per-file', 
                       type=int,
                       default=1000,
                       help='Maximum characters to read from each non-tabular file')
    parser.add_argument('--max-chars-per-tabular', 
                       type=int,
                       default=2000,
                       help='Maximum characters to read from each tabular file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate folder path
    if not os.path.isdir(args.folder_path):
        print(f"Error: '{args.folder_path}' is not a valid directory")
        exit(1)
    
    # Generate and print the data prompt
    prompt = generate_data_prompt(
        input_data_folder=args.folder_path,
        max_chars_per_file=args.max_chars_per_file,
        max_chars_per_tabular_to_text=args.max_chars_per_tabular
    )
    
    print(prompt)
