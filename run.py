import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from automlagent.coding_agent import run_agent
from automlagent.utils import extract_archives

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
        "-e", "--extract_archives", action="store_true", help="Extract the archives."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )
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

    if args.extract_archives:
        print(f"Note: we strongly recommend using data without archived files. Extracting archived files under {args.input_data_folder}...")
        extract_archives(args.input_data_folder)

    # Generate and execute code
    run_agent(
        input_data_folder=args.input_data_folder,
        tutorial_link=None,  # TODO: Only needed if we use RAG
        output_folder=str(output_dir),
        config_path=args.config_path,
        max_iterations=args.max_iterations,
        need_user_input=args.need_user_input,
    )


if __name__ == "__main__":
    main()
