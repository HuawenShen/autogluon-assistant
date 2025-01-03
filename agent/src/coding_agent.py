import argparse
import os
import subprocess
from .coding import generate_code, write_code_script, write_retrieved_context
from .prompt import PromptGenerator, write_prompt_to_file


def execute_bash_script(bash_script):
    """
    Execute the generated bash code and return success status and error message.
    
    Args:
        bash_code (str): Bash code to execute
        
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        # Execute the bash code
        process = subprocess.run(
            bash_script,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            return True, ""
        else:
            return False, process.stderr
            
    except Exception as e:
        return False, str(e)


def generate_code_script(
    input_data_folder,
    tutorial_path,
    tutorial_link,
    output_folder,
    model_id,
    backend,
    config_path,
    max_iterations=5,
    need_user_input=False,
):
    prompt_generator = PromptGenerator(
        input_data_folder=input_data_folder,
        tutorials_folder=tutorial_path,
        output_folder=output_folder,
        config_path=config_path,
    )
    
    iteration = 0
    while iteration < max_iterations:
        # Create iteration subfolder
        iteration_folder = os.path.join(output_folder, f"iteration_{iteration}")
        os.makedirs(iteration_folder, exist_ok=True)
        
        # Get user inputs if needed
        user_input = None
        if need_user_input:
            if iteration > 0:
                print(f"\nPrevious iteration files are in: {os.path.join(output_folder, f'iteration_{iteration-1}')}")
            user_input = input("Enter your inputs for this iteration (press Enter to skip): ")
        
        prompt_generator.step(user_input=user_input)
        
        # Generate and save the coding prompt
        coding_prompt = prompt_generator.get_coding_prompt()
        coding_prompt_path = os.path.join(iteration_folder, "coding_prompt.txt")
        write_prompt_to_file(coding_prompt, coding_prompt_path)
        
        # Generate code
        generated_content = generate_code(coding_prompt, model_id, backend, tutorial_link)
        generated_python_code = generated_content["code_script"]
        
        # Save the python code
        python_file_path = os.path.join(iteration_folder, "generated_code.py")
        write_code_script(generated_python_code, python_file_path)
        
        # Write retrieved context if present
        if "retrieved_context" in generated_content:
            output_context_path = os.path.join(iteration_folder, "retrieved_context.txt")
            write_retrieved_context(
                generated_content["retrieved_context"],
                output_context_path
            )
        
        prompt_generator.update_python_code(python_code=generated_python_code)
        
        # Generate and save the execution prompt
        execution_prompt = prompt_generator.get_execution_prompt(python_file_path=python_file_path)
        execution_prompt_path = os.path.join(iteration_folder, "execution_prompt.txt")
        write_prompt_to_file(execution_prompt, execution_prompt_path)
        
        # Generate bash code
        generated_bash_script = generate_code(execution_prompt, model_id, backend, None)["code_script"]
        
        # Save the bash code
        bash_file_path = os.path.join(iteration_folder, "execution_script.sh")
        write_code_script(generated_bash_script, bash_file_path)
        
        prompt_generator.update_bash_script(bash_script=generated_bash_script)
        
        try:
            # Attempt to execute the generated code
            success, error_message = execute_bash_script(generated_bash_script)
            
            if success:
                print(f"Code generation successful after {iteration + 1} iterations")
                break
                
            # Feed error back into prompt generator
            prompt_generator.update_error_message(error_message=error_message)
            
        except Exception as e:
            error_message = str(e)
            prompt_generator.update_error_message(error_message=error_message)
        
        iteration += 1
        print(f"Starting iteration {iteration + 1}")
    
    if iteration == max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}) without success")
