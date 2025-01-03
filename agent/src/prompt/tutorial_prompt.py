import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..llm import LLMFactory
from .utils import generate_chat_prompt

logger = logging.getLogger(__name__)


def get_all_tutorials(tutorial_folder: str) -> List[Tuple[Path, str]]:
    """Get all tutorial files from the folder (including nested directories).

    Args:
        tutorial_folder: Path to the folder containing tutorial files

    Returns:
        List of (file_path, title) tuples
    """
    tutorial_files = []
    tutorial_dir = Path(tutorial_folder)

    for file_path in tutorial_dir.rglob(
        "*.md"
    ):  # Assuming tutorials are markdown files
        # Extract title from first line of file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                title = first_line.lstrip("#").strip()  # Remove markdown header symbols
                tutorial_files.append((file_path, title))
        except Exception as e:
            logger.warning(f"Error reading tutorial file {file_path}: {e}")
            continue

    return tutorial_files


def select_relevant_tutorials(
    tutorials: List[Tuple[Path, str]],
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    llm_config,
    max_num_tutorials: int,
) -> List[Tuple[Path, str, float]]:
    """Select most relevant tutorials using LLM scoring based only on titles."""

    # Create LLM instance
    llm = LLMFactory.get_chat_model(llm_config)

    # Construct context for relevance scoring
    context = f"""Task: {task_prompt}
    Data: {data_prompt}
    User Question: {user_prompt}
    Error: {error_prompt}"""

    # Create a single prompt for all titles to minimize API calls
    titles_list = "\n".join(
        [f"{i+1}. {title}" for i, (_, title) in enumerate(tutorials)]
    )

    prompt = f"""Given the following context and list of tutorial titles, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial title matches the task, data, user question, and any errors.

    Context:
    {context}
    
    Tutorial Titles:
    {titles_list}
    
    IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
    For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
    DO NOT include any other text, explanation, or formatting in your response."""

    try:
        response = llm.invoke(generate_chat_prompt(prompt=prompt).format_messages())
        # Clean and parse the response
        content = response.content.strip()

        # Extract first line in case of multi-line response
        content = content.split("\n")[0]

        # Remove any non-numeric characters except commas
        content = "".join(char for char in content if char.isdigit() or char == ",")

        if not content:
            logger.warning("No valid indices found in LLM response")
            return [(path, title) for path, title in tutorials[:max_num_tutorials]]

        # Parse the cleaned response to get selected indices
        try:
            selected_indices = [
                int(idx.strip()) - 1 for idx in content.split(",") if idx.strip()
            ]
        except ValueError as e:
            logger.warning(f"Error parsing indices from LLM response: {e}")
            return [(path, title) for path, title in tutorials[:max_num_tutorials]]

        # Get the selected tutorials
        selected_tutorials = []
        for idx in selected_indices:
            if 0 <= idx < len(tutorials):
                file_path, title = tutorials[idx]
                selected_tutorials.append(
                    (file_path, title)
                )  # Using 1.0 as score since these were selected

        if len(selected_tutorials) > max_num_tutorials:
            selected_tutorials = selected_tutorials[:max_num_tutorials]
        return selected_tutorials

    except Exception as e:
        logger.warning(f"Error selecting tutorials: {e}")
        raise e


def format_tutorial_content(file_path: Path, title: str, max_length: int) -> str:
    """Format a single tutorial's content with truncation."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate if needed
        if len(content) > max_length:
            content = content[:max_length] + "\n...(truncated)"

        formatted = f"""### {title}
        
        {content}
        """
        return formatted

    except Exception as e:
        logger.warning(f"Error formatting tutorial {file_path}: {e}")
        return ""


def save_selection_results(
    output_folder: Path,
    selected_tutorials: List[Tuple[Path, str, float]],
    formatted_tutorials: List[str],
    tutorial_prompt: str,
) -> None:
    """Save selection results and prompt to output folder."""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save selected tutorial metadata
        selection_data = [
            {
                "path": str(path),
                "title": title,
            }
            for path, title in selected_tutorials
        ]

        with open(
            output_folder / "selected_tutorials.json", "w", encoding="utf-8"
        ) as f:
            json.dump(selection_data, f, indent=2)

        # Save tutorial contents individually
        contents_folder = output_folder / "tutorial_contents"
        contents_folder.mkdir(exist_ok=True)

        for i, content in enumerate(formatted_tutorials, 1):
            with open(contents_folder / f"tutorial_{i}.md", "w", encoding="utf-8") as f:
                f.write(content)

        # Save final prompt
        with open(output_folder / "tutorial_prompt.txt", "w", encoding="utf-8") as f:
            f.write(tutorial_prompt)

    except Exception as e:
        logger.error(f"Error saving selection results: {e}")


def generate_tutorial_prompt(
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    tutorial_folder: str,
    llm_config,
    output_folder: Optional[str],
    max_num_tutorials: int = 3,
    max_tutorial_length: int = 9999,
) -> str:
    """Generate a tutorial prompt by selecting relevant tutorials based on current context.

    Args:
        task_prompt: Describe the data science task
        data_prompt: Describe the data
        user_prompt: Instructions from the user
        error_prompt: Error from last run
        tutorial_folder: Path to the folder containing tutorial files (could be nested structure)
        max_num_tutorials: Maximum number of tutorials to include
        max_tutorial_length: Maximum length for each tutorial

    Returns:
        str: Formatted tutorial prompt containing selected tutorials
    """
    # Safety check for tutorial folder
    if not Path(tutorial_folder).exists():
        logger.warning(f"Tutorial folder not found: {tutorial_folder}")
        return ""

    # Get all available tutorials
    tutorials = get_all_tutorials(tutorial_folder)
    if not tutorials:
        logger.warning(f"No tutorials found in {tutorial_folder}")
        return ""

    # Select relevant tutorials
    selected_tutorials = select_relevant_tutorials(
        tutorials,
        task_prompt,
        data_prompt,
        user_prompt,
        error_prompt,
        llm_config,
        max_num_tutorials,
    )

    if not selected_tutorials:
        return ""

    # Format selected tutorials
    formatted_tutorials = []
    for file_path, title in selected_tutorials:
        formatted = format_tutorial_content(file_path, title, max_tutorial_length)
        if formatted:
            formatted_tutorials.append(formatted)

    if not formatted_tutorials:
        return ""

    prompt = "RELEVANT TUTORIALS:\n" + "\n\n".join(formatted_tutorials)

    # Save results if output folder is provided
    output_path = Path(output_folder)
    save_selection_results(output_path, selected_tutorials, formatted_tutorials, prompt)

    return prompt
