import argparse

from .prompt import generate_prompt, write_prompt_to_file
from .script import generate_code, write_code_script


def generate_code_script(
    input_data_folder, tutorial_path, tutorial_link, output_result_file, output_prompt_file, model_id, output_code_file, backend
):
    prompt = generate_prompt(
        input_data_folder, tutorial_path, output_result_file
    )
    write_prompt_to_file(prompt, output_prompt_file)
    script = generate_code(prompt, model_id, backend, tutorial_link)
    write_code_script(script, output_code_file)


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
        "-o", "--output_result_file", required=True, help="Path for the output file"
    )
    parser.add_argument(
        "-c", "--output_code_file", required=True, help="Path to the output code file"
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
        output_result_file=args.output_result_file,
        output_prompt_file=args.output_prompt_file,
        model_id=args.model_id,
        output_code_file=args.output_code_file,
    )
