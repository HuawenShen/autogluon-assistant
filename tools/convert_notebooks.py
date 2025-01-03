import os
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter


def convert_notebook_to_markdown(notebook_path, output_path):
    """
    Convert a single Jupyter notebook to Markdown format.

    Args:
        notebook_path (str): Path to the input notebook
        output_path (str): Path where the markdown file should be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Convert to markdown
    markdown_exporter = MarkdownExporter()
    markdown, _ = markdown_exporter.from_notebook_node(notebook)

    # Write the markdown file
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown)


def batch_convert_notebooks(input_dir, output_dir):
    """
    Convert all Jupyter notebooks in a directory (and its subdirectories) to Markdown.

    Args:
        input_dir (str): Root directory containing the notebooks
        output_dir (str): Directory where markdown files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all notebook files
    notebook_files = input_path.rglob("*.ipynb")

    for notebook_path in notebook_files:
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(notebook_path):
            continue

        # Create equivalent markdown path in output directory
        relative_path = notebook_path.relative_to(input_path)
        markdown_path = output_path / relative_path.with_suffix(".md")

        print(f"Converting: {notebook_path} -> {markdown_path}")
        try:
            convert_notebook_to_markdown(str(notebook_path), str(markdown_path))
        except Exception as e:
            print(f"Error converting {notebook_path}: {str(e)}")


if __name__ == "__main__":
    # Replace with your directory paths
    input_directory = "/media/ag/autogluon/docs/tutorials"
    output_directory = "/media/deephome/AutoMLAgent/AutogluonTutorials"
    batch_convert_notebooks(input_directory, output_directory)
