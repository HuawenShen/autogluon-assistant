import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..llm import ChatLLMFactory

logger = logging.getLogger(__name__)


def generate_error_prompt(
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    python_code: str,
    bash_script: str,
    tutorial_prompt: str,
    error_message: str,
    llm_config,
    output_folder: Optional[str],
    max_error_message_length: int = 2000,
) -> str:
    """Generate an error prompt by analyzing the error message and providing guidance for code improvement.
    
    Args:
        task_prompt: Description of the data science task
        data_prompt: Description of the data
        user_prompt: Instructions from the user
        python_code: Previous Python code that generated the error
        bash_script: Previous Bash script that generated the error
        error_message: Error message from the last run
        llm_config: Configuration for the LLM
        output_folder: Optional folder to save the results
        max_error_message_length: Maximum length for error message
        
    Returns:
        str: Formatted error prompt with analysis and suggestions
    """
    try:
        # Truncate error message if needed
        if len(error_message) > max_error_message_length:
            error_message = (
                error_message[: max_error_message_length // 2]
                + "\n...(truncated)\n"
                + error_message[-max_error_message_length // 2 :]
            )

        # Create LLM instance
        llm = ChatLLMFactory.get_chat_model(llm_config)

        # Construct context for error analysis
        context = f"""{task_prompt}
{data_prompt}
{user_prompt}

Previous Python Code:
```python
{python_code}
```

Previous Bash Script to Execute the Python Code:
```bash
{bash_script}
```

{tutorial_prompt}

Error Message:
{error_message}"""

        # Prompt for LLM to analyze error
        analysis_prompt = """Analyze the error message and context provided. Generate a clear, concise summary of the error and provide specific suggestions for fixing it. Focus on:
1. The root cause of the error
2. A specific and concise suggestion for how to fix it, no code needed.

Format your response in two parts:
ERROR SUMMARY: (Brief description of the error)
SUGGESTED FIX: (Specific and concise suggestion, no code needed)

Keep the response focused and technical. Do not include general advice or explanations."""

        # Get error analysis from LLM
        error_analysis = llm.assistant_chat(context + "\n\n" + analysis_prompt)

        # Save results if output folder is provided
        if output_folder:
            save_error_analysis(
                Path(output_folder), context, error_analysis, error_message
            )

        return error_analysis

    except Exception as e:
        logger.error(f"Error generating error prompt: {e}")
        # Fallback to basic error message if LLM analysis fails
        return f"Error Summary: {str(error_message)[:max_error_message_length]}"


def save_error_analysis(
    output_folder: Path, context: str, error_analysis: str, original_error: str
) -> None:
    """Save error analysis results to output folder."""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)

        analysis_data = {
            "context": context,
            "error_analysis": error_analysis,
            "original_error": original_error,
            "timestamp": str(datetime.now()),
        }

        with open(output_folder / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving error analysis: {e}")
