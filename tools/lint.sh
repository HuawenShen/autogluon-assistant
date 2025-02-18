#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Formatting Python files in $SCRIPT_DIR..."

# Format with black
echo "Running black..."
black "$SCRIPT_DIR"

# Run ruff with auto-fix
echo "Running ruff..."
ruff check --fix "$SCRIPT_DIR"

# Sort imports
echo "Running isort..."
isort "$SCRIPT_DIR"

echo "Done! All Python files in $SCRIPT_DIR have been formatted."
