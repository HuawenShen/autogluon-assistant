import os


def generate_execution_prompt(
    output_folder,
    python_file_path,
    create_venv=True,
    previous_bash=None,
    previous_python=None,
    error_message=None,
    current_python=None,
):
    """
    Generate a prompt for an LLM to create a bash script for environment setup and code execution.

    Args:
        output_folder (str): Path to the project folder
        python_file_path (str): Absolute path to the Python file that needs to be executed
        create_venv (bool): Whether to create a new virtual environment or use current environment
        previous_bash (str, optional): Previous bash script that caused errors
        previous_python (str, optional): Previous Python code in the bash script
        error_messages (list, optional): Previous error message to help with debugging

    Returns:
        str: Formatted prompt for the LLM
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Base requirements based on whether to create venv or not
    if create_venv:
        execution_prompt = f"""Please create a bash script that will:
1. Create a Python virtual environment in the folder: {output_folder}
2. Activate the virtual environment
3. Install all necessary Python packages
4. Execute the Python script located at: {python_file_path}"""
    else:
        execution_prompt = f"""Please create a bash script that will:
1. Execute the Python script located at: {python_file_path}"""

    # Add current Python code for environment analysis
    if current_python:
        execution_prompt += f"""

Current Python code to be executed:
```python
{current_python}
```

Please analyze this code to determine required packages and environment setup."""

    # Add error context if provided
    if error_message:
        execution_prompt += "\n\nPrevious error encountered: \n\n{error_message}"

        # Only include previous code if there are error messages
        if previous_bash:
            execution_prompt += f"""

Previous bash script that failed:
```bash
{previous_bash}
```"""

        if previous_python:
            execution_prompt += f"""

Previous Python code in the bash script:
```python
{previous_python}
```"""

        execution_prompt += "\n\nIf any of these errors are related to environment setup, package installation, or Python version issues, please ensure the script handles them."

    execution_prompt += "\n\nPlease format the response as a complete, executable bash script that can be run directly."

    return execution_prompt
