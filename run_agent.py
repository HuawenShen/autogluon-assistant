import argparse
import os

from agent.src.coding_agent import generate_code_script

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
        "-w", "--result_dir", required=True, help="Path for the output folder"
    )
    parser.add_argument(
        "-m",
        "--model_id",
        required=False,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Claude model ID to use",
    )
    args = parser.parse_args()

    output_result_file = os.path.join(args.result_dir, "result.csv")
    output_prompt_file = os.path.join(args.result_dir, "prompt.txt")
    output_code_file = os.path.join(args.result_dir, "generated_code.py")

    generate_code_script(
        input_data_folder=args.input_data_folder,
        tutorial_path=args.tutorial_path,
        output_result_file=output_result_file,
        output_prompt_file=output_prompt_file,
        model_id=args.model_id,
        output_code_file=output_code_file,
    )
