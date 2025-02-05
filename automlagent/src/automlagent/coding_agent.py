import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .coder import generate_coder, write_code_script, write_retrieved_context
from .llm import ChatLLMFactory
from .prompt import PromptGenerator, write_prompt_to_file


def execute_bash_script(bash_script, stream_output=True):
    """
    Execute bash script with real-time output streaming.
    """
    try:
        process = subprocess.Popen(
            ["bash", "-c", bash_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_chunks = []
        stderr_chunks = []

        import select

        # Set up tracking of both output streams
        streams = [process.stdout, process.stderr]

        while streams:
            # Wait for output on either stream
            readable, _, _ = select.select(streams, [], [])

            for stream in readable:
                line = stream.readline()
                if not line:  # EOF
                    streams.remove(stream)
                    continue

                # Handle stdout
                if stream == process.stdout:
                    stdout_chunks.append(line)
                    if stream_output:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                # Handle stderr
                else:
                    stderr_chunks.append(line)
                    if stream_output:
                        sys.stderr.write(line)
                        sys.stderr.flush()

        process.wait()
        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)
    except Exception as e:
        return False, "", str(e)


def save_iteration_state(iteration_folder, prompt_generator, stdout, stderr):
    """
    Save the current state of the prompt generator and execution outputs to separate files.

    Args:
        iteration_folder (str): Path to the current iteration folder
        prompt_generator (PromptGenerator): Current prompt generator instance
        stdout (str): Standard output from execution
        stderr (str): Standard error from execution
    """
    # Create a states subfolder
    states_folder = os.path.join(iteration_folder, "states")
    os.makedirs(states_folder, exist_ok=True)

    # Save each state component to a separate file
    state_files = {
        "user_input.txt": prompt_generator.user_input or "",
        "python_code.py": prompt_generator.python_code or "",
        "bash_script.sh": prompt_generator.bash_script or "",
        "error_message.txt": prompt_generator.error_message or "",
        "tutorial_prompt.txt": prompt_generator.tutorial_prompt or "",
        "data_prompt.txt": prompt_generator.data_prompt or "",
        "task_prompt.txt": prompt_generator.task_prompt or "",
        "stdout.txt": stdout or "",
        "stderr.txt": stderr or "",
    }

    for filename, content in state_files.items():
        file_path = os.path.join(states_folder, filename)
        with open(file_path, "w") as f:
            f.write(content)


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
    # Load config from YAML and merge with default
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)

    prompt_generator = PromptGenerator(
        input_data_folder=input_data_folder,
        tutorials_folder=tutorial_path,
        output_folder=output_folder,
        config=config,
    )
    python_coder = generate_coder(
        llm_config=config.coder, tutorial_link_for_rag=tutorial_link
    )
    bash_coder = generate_coder(
        llm_config=config.coder, tutorial_link_for_rag=tutorial_link
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
                print(
                    f"\nPrevious iteration files are in: {os.path.join(output_folder, f'iteration_{iteration-1}')}"
                )
            user_input = input(
                "Enter your inputs for this iteration (press Enter to skip): "
            )

        prompt_generator.step(user_input=user_input)

        # Generate and save the coding prompt
        coding_prompt = prompt_generator.get_coding_prompt()
        coding_prompt_path = os.path.join(iteration_folder, "coding_prompt.txt")
        write_prompt_to_file(coding_prompt, coding_prompt_path)

        # Generate code
        generated_content = python_coder(prompt=coding_prompt, language="python")
        generated_python_code = generated_content["code_script"]

        # Save the python code
        python_file_path = os.path.join(iteration_folder, "generated_code.py")
        write_code_script(generated_python_code, python_file_path)

        # Write retrieved context if present
        if "retrieved_context" in generated_content:
            output_context_path = os.path.join(
                iteration_folder, "retrieved_context.txt"
            )
            write_retrieved_context(
                generated_content["retrieved_context"], output_context_path
            )

        prompt_generator.update_python_code(python_code=generated_python_code)

        # Generate and save the execution prompt
        execution_prompt = prompt_generator.get_execution_prompt(
            python_file_path=python_file_path
        )
        execution_prompt_path = os.path.join(iteration_folder, "execution_prompt.txt")
        write_prompt_to_file(execution_prompt, execution_prompt_path)

        # Generate bash code
        generated_bash_script = bash_coder(prompt=execution_prompt, language="bash")[
            "code_script"
        ]

        # Save the bash code
        bash_file_path = os.path.join(iteration_folder, "execution_script.sh")
        write_code_script(generated_bash_script, bash_file_path)

        prompt_generator.update_bash_script(bash_script=generated_bash_script)

        try:
            # Attempt to execute the generated code
            success, stdout, stderr = execute_bash_script(generated_bash_script)

            if success:
                print(f"Code generation successful after {iteration + 1} iterations")
                prompt_generator.update_error_message(error_message="")
                # Save the current state
                save_iteration_state(iteration_folder, prompt_generator, stdout, stderr)
                break

            # Feed error back into prompt generator
            prompt_generator.update_error_message(error_message=stderr)

            # Save the current state
            save_iteration_state(iteration_folder, prompt_generator, stdout, stderr)

        except Exception as e:
            error_message = str(e)
            prompt_generator.update_error_message(error_message=error_message)
            # Save the current state
            save_iteration_state(iteration_folder, prompt_generator, "", error_message)

        iteration += 1
        print(f"Starting iteration {iteration + 1}")

    if iteration == max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}) without success")
    
    print(f"Total token usage:\n{ChatLLMFactory.get_total_token_usage()}")
