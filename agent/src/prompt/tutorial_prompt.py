import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..llm import ChatLLMFactory
from ..tools_registry import get_tool_tutorials_folder

logger = logging.getLogger(__name__)


def get_all_tutorials(tool_name: str) -> List[Tuple[Path, str]]:
    """Get all tutorial files of the tool.

    Args:
        tool_name: Name of the ML tool to use in codes

    Returns:
        List of (file_path, title) tuples
    """
    tutorial_dir = get_tool_tutorials_folder(tool_name)

    tutorial_files = []
    for file_path in tutorial_dir.rglob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                title = first_line.lstrip("#").strip()
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

    llm_select_tutorial = ChatLLMFactory.get_chat_model(llm_config)

    context = f"""Task: {task_prompt}
    Data: {data_prompt}
    User Question: {user_prompt}
    Error: {error_prompt}"""

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
        content = llm_select_tutorial.assistant_chat(prompt)
        content = content.split("\n")[0]
        content = "".join(char for char in content if char.isdigit() or char == ",")

        if not content:
            logger.warning("No valid indices found in LLM response")
            return [(path, title) for path, title in tutorials[:max_num_tutorials]]

        try:
            selected_indices = [
                int(idx.strip()) - 1 for idx in content.split(",") if idx.strip()
            ]
        except ValueError as e:
            logger.warning(f"Error parsing indices from LLM response: {e}")
            return [(path, title) for path, title in tutorials[:max_num_tutorials]]

        selected_tutorials = []
        for idx in selected_indices:
            if 0 <= idx < len(tutorials):
                file_path, title = tutorials[idx]
                selected_tutorials.append((file_path, title))

        if len(selected_tutorials) > max_num_tutorials:
            selected_tutorials = selected_tutorials[:max_num_tutorials]
        return selected_tutorials

    except Exception as e:
        logger.warning(f"Error selecting tutorials: {e}")
        raise e


def condense_tutorial_content(
    content: str,
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    llm_config,
) -> str:
    """Condense tutorial content by removing non-essential parts for coding.
    
    Args:
        content: Original tutorial content
        task_prompt: Current task description
        data_prompt: Data description
        user_prompt: User's question/request
        error_prompt: Any error messages
        llm_config: LLM configuration
        
    Returns:
        str: Condensed tutorial content
    """
    llm_condense = ChatLLMFactory.get_chat_model(llm_config)

    context = f"""Task: {task_prompt}
    Data: {data_prompt}
    User Question: {user_prompt}
    Error: {error_prompt}"""

    prompt = f"""Condense the following tutorial by keeping only the parts that are directly useful for coding. Focus on:
    1. Code snippets and their essential explanations
    2. Key implementation details and patterns
    3. Important parameters and their usage
    4. Critical warnings or gotchas that affect code functionality

    Remove:
    1. General introductions and background information
    2. Theory explanations not directly tied to implementation
    3. Alternative approaches not relevant to the current task
    4. Extended examples not related to the current context

    Context of the current task:
    {context}

    Tutorial content:
    {content}

    Provide ONLY the condensed tutorial content, maintaining the markdown formatting for code blocks and headers."""

    try:
        condensed_content = llm_condense.assistant_chat(prompt)
        return condensed_content
    except Exception as e:
        logger.warning(f"Error condensing tutorial content: {e}")
        return content  # Return original content if condensing fails


def format_tutorial_content(
    file_path: Path,
    title: str,
    max_length: int,
    should_condense: bool = False,
    task_prompt: str = "",
    data_prompt: str = "",
    user_prompt: str = "",
    error_prompt: str = "",
    llm_config=None,
) -> str:
    """Format a single tutorial's content with optional condensing and truncation."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if should_condense and llm_config:
            content = condense_tutorial_content(
                content, task_prompt, data_prompt, user_prompt, error_prompt, llm_config
            )

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

        selection_data = [
            {"path": str(path), "title": title} for path, title in selected_tutorials
        ]

        with open(
            output_folder / "selected_tutorials.json", "w", encoding="utf-8"
        ) as f:
            json.dump(selection_data, f, indent=2)

        contents_folder = output_folder / "tutorial_contents"
        contents_folder.mkdir(exist_ok=True)

        for i, content in enumerate(formatted_tutorials, 1):
            with open(contents_folder / f"tutorial_{i}.md", "w", encoding="utf-8") as f:
                f.write(content)

        with open(output_folder / "tutorial_prompt.txt", "w", encoding="utf-8") as f:
            f.write(tutorial_prompt)

    except Exception as e:
        logger.error(f"Error saving selection results: {e}")


def generate_tutorial_prompt(
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    tool_name: str,
    llm_config,
    output_folder: Optional[str],
    max_num_tutorials: int = 3,
    max_tutorial_length: int = 9999,
    condense_tutorials: bool = False,
) -> str:
    """Generate a tutorial prompt by selecting and optionally condensing relevant tutorials.

    Args:
        task_prompt: Describe the data science task
        data_prompt: Describe the data
        user_prompt: Instructions from the user
        error_prompt: Error from last run
        tool_name: Name of the ML tool to use in codes
        llm_config: Configuration for the LLM
        output_folder: Optional folder to save results
        max_num_tutorials: Maximum number of tutorials to include
        max_tutorial_length: Maximum length for each tutorial
        condense_tutorials: Whether to condense tutorials using LLM

    Returns:
        str: Formatted tutorial prompt containing selected tutorials
    """
    tutorials = get_all_tutorials(tool_name)
    if not tutorials:
        logger.warning(f"No tutorials found for {tool_name}")
        return ""

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

    formatted_tutorials = []
    for file_path, title in selected_tutorials:
        formatted = format_tutorial_content(
            file_path,
            title,
            max_tutorial_length,
            should_condense=condense_tutorials,
            task_prompt=task_prompt,
            data_prompt=data_prompt,
            user_prompt=user_prompt,
            error_prompt=error_prompt,
            llm_config=llm_config if condense_tutorials else None,
        )
        if formatted:
            formatted_tutorials.append(formatted)

    if not formatted_tutorials:
        return ""

    prompt = "RELEVANT TUTORIALS:\n" + "\n\n".join(formatted_tutorials)

    if output_folder:
        output_path = Path(output_folder)
        save_selection_results(
            output_path, selected_tutorials, formatted_tutorials, prompt
        )

    return prompt
