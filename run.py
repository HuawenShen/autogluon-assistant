import argparse
import logging
from pathlib import Path

from agent.src.coding_agent import generate_code_script

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute code using AutoML Agent"
    )
    parser.add_argument(
        "-i", "--input_data_folder", required=True, help="Path to the input data folder"
    )
    parser.add_argument(
        "-t", "--tutorial_path", required=True, help="Path to the tutorials folder"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-m",
        "--model_id",
        required=True,
        help="LLM Model ID for task understanding and code generation",
    )
    parser.add_argument("-b", "--backend", required=True, help="LLM Backend")
    parser.add_argument(
        "-c", "--config_path", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "-n",
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of iterations for code generation",
    )
    parser.add_argument(
        "--need_user_input",
        action="store_true",
        help="Enable user input between iterations",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate and execute code
        generate_code_script(
            input_data_folder=args.input_data_folder,
            tutorial_path=args.tutorial_path,
            tutorial_link=None,  # TODO: Only needed if we use RAG
            output_folder=str(output_dir),
            model_id=args.model_id,
            backend=args.backend,
            config_path=args.config_path,
            max_iterations=args.max_iterations,
            need_user_input=args.need_user_input,
        )
    except Exception as e:
        logger.error(f"Error during code generation and execution: {e}")
        raise


if __name__ == "__main__":
    main()
