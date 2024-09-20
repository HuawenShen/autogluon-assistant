import argparse

from .prompt import generate_prompt
from .script import use_bedrock_to_generate, write_code_script

def generate_code_script(input_data_folder, tutorial_path, output_result_file, model_id, output_code_file):
    prompt = generate_prompt(args.input_data_folder, args.tutorial_path, args.output_result_file)
    script = use_bedrock_to_generate(prompt, args.model_id)
    write_code_script(script, args.output_code_file)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract Python script from Claude response')
    parser.add_argument('-i', '--input_data_folder', required=True, help="Path to the input folder")
    parser.add_argument('-t', '--tutorial_path', required=True, help="Path to the Autogluon Tabular tutorial file")
    parser.add_argument('-o', '--output_result_file', required=True, help="Path for the output file")
    parser.add_argument('-c', '--output_code_file', required=True, help='Path to the output code file')
    parser.add_argument('-m', '--model_id', required=False, default='anthropic.claude-3-haiku-20240307-v1:0', help='Claude model ID to use')
    args = parser.parse_args()

    generate_code_script(
        input_data_folder=args.input_data_folder,
        tutorial_path=args.tutorial_path,
        output_result_file=args.output_result_file,
        model_id=args.model_id,
        output_code_file=args.output_code_file,
    )
