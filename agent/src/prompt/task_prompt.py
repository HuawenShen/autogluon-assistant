import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from omegaconf import DictConfig
from pathlib import Path

from ..llm import LLMFactory

logger = logging.getLogger(__name__)

def find_description_files(data_prompt: str, llm) -> Tuple[List[str], str]:
    """
    Step 1: Use LLM to identify potential description files from the data prompt.
    Only identifies files, does not read content.
    
    Args:
        data_prompt: Text string containing data prompt
        llm: Initialized LLM model
        
    Returns:
        Tuple[List[str], str]: (List of identified description filenames, Analysis explanation)
    """
    try:
        find_descriptions_prompt = f"""
        Given this data prompt:

        {data_prompt}

        Please identify any files that appear to contain project descriptions, requirements, or task definitions.
        Look for files like README, documentation files, or task description files.
        
        Format your response as follows:
        Description Files: [list ONLY the filenames, one per line]
        Explanation: [explain why these files were identified as description files]
        """
        
        response = llm.invoke([{"role": "user", "content": find_descriptions_prompt}])
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        # Extract filenames from the response
        description_files = []
        lines = analysis.split('\n')
        in_files_section = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('description files:'):
                in_files_section = True
                continue
            elif line.lower().startswith('explanation:'):
                break
            elif in_files_section and line:
                filename = line.strip('- []').strip()
                if filename:
                    description_files.append(filename)
        
        return description_files, analysis
        
    except Exception as e:
        logger.error(f"Error in finding description files: {e}")
        return [], str(e)

def generate_task_description(
    data_prompt: str,
    description_files: List[str],
    description_analysis: str,
    llm
) -> str:
    """
    Step 2: Read content of identified files and generate task description.
    
    Args:
        data_prompt: Text string containing data prompt
        description_files: List of description filenames from step 1
        description_analysis: Analysis from step 1
        llm: Initialized LLM model
        
    Returns:
        str: Generated task description
    """
    try:
        # Read content of identified description files
        file_contents = []
        for filename in description_files:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                file_contents.append(f"File: {filename}\nContent: {content}\n")
            except Exception as e:
                logger.warning(f"Could not read content of {filename}: {e}")
                continue
        
        description_context = "\n".join(file_contents) if file_contents else "No description file contents could be read."
        
        task_prompt = f"""
        Based on this data prompt and description files:

        Data Prompt:
        {data_prompt}

        Description File Analysis:
        {description_analysis}

        Description File Contents:
        {description_context}

        Please write a comprehensive description of the data science task, including:
        1. Task objective
        2. Input data description
        3. Required analysis or processing steps
        4. Expected outputs
        5. Any specific requirements or constraints
        """
        
        response = llm.invoke([{"role": "user", "content": task_prompt}])
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        logger.error(f"Error in generating task description: {e}")
        return f"Error generating task description: {str(e)}"

def generate_task_prompt(
    data_prompt: str,
    output_folder: str,
    llm_config: Optional[DictConfig] = None,
) -> str:
    """
    Main function to generate task prompt following two-step process.
    
    Args:
        data_prompt: Text string containing data prompt
        output_folder: Path to the output folder
        llm_config: Configuration for the LLM model
        
    Returns:
        str: Generated task prompt
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM
    if not llm_config:
        logger.warning("No LLM config provided. Cannot generate task prompt without LLM.")
        return data_prompt
    
    llm = LLMFactory.get_chat_model(llm_config)
    
    # Step 1: Find description files (just identifies files, doesn't read content)
    description_files, description_analysis = find_description_files(data_prompt, llm)
    logger.info(f"Found {len(description_files)} potential description files: {description_files}")
    
    # Step 2: Generate task description (includes reading file contents)
    task_description = generate_task_description(
        data_prompt,
        description_files,
        description_analysis,
        llm
    )
    
    # Save results in separate files
    # Save description file names
    files_path = os.path.join(output_folder, 'description_files.txt')
    with open(files_path, 'w') as f:
        for filename in description_files:
            f.write(f"{filename}\n")
    logger.info(f"Description files list saved to: {files_path}")
    
    # Save description analysis
    analysis_path = os.path.join(output_folder, 'description_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write(description_analysis)
    logger.info(f"Description analysis saved to: {analysis_path}")
    
    # Save generated task description
    task_path = os.path.join(output_folder, 'task_description.txt')
    with open(task_path, 'w') as f:
        f.write(task_description)
    logger.info(f"Generated task description saved to: {task_path}")
    
    return task_description
