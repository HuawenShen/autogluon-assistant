import argparse
import os

from .coding import generate_code, write_code_script, write_retrieved_context
from .prompt import PromptGenerator, write_prompt_to_file


def generate_code_script(
    input_data_folder,
    tutorial_path,
    tutorial_link,
    output_folder,
    output_prompt_file,
    model_id,
    output_code_file,
    backend,
    config_path,
):
    prompt_generator = PromptGenerator(
        input_data_folder=input_data_folder,
        tutorials_folder=tutorial_path,
        output_folder=output_folder,
        config_path=config_path)
    prompt_generator.step(user_inputs="", error_message="")
    prompt = prompt_generator.get_iterative_prompt()

    write_prompt_to_file(prompt, output_prompt_file)

    generated_content = generate_code(prompt, model_id, backend, tutorial_link)
    write_code_script(generated_content["code_script"], output_code_file)

    if "retrieved_context" in generated_content:
        # Create the path for retrieved_context.txt in the same directory
        output_context_path = os.path.join(
            os.path.dirname(output_code_file), "retrieved_context.txt"
        )
        write_retrieved_context(
            generated_content["retrieved_context"], output_context_path
        )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract Python script from Claude response"
    )
    parser.add_argument(
        "-i", "--input_data_folder", required=True, help="Path to the input folder"
    )
    parser.add_argument(
        "-t",
        "--tutorial_path",
        required=True,
        help="Path to the Autogluon Tabular tutorial file",
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, help="Path for the output file"
    )
    parser.add_argument(
        "-c", "--output_code_file", required=True, help="Path to the output code file"
    )
    parser.add_argument(
        "-f", "--config_path", required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-p",
        "--output_prompt_file",
        required=True,
        help="Path for the generated prompt file",
    )
    parser.add_argument(
        "-m",
        "--model_id",
        required=False,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Claude model ID to use",
    )
    args = parser.parse_args()

    generate_code_script(
        input_data_folder=args.input_data_folder,
        tutorial_path=args.tutorial_path,
        output_folder=args.output_folder,
        output_prompt_file=args.output_prompt_file,
        model_id=args.model_id,
        output_code_file=args.output_code_file,
        config_path=args.config_path,
    )
