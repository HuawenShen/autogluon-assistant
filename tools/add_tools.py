#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import json
import yaml
from omegaconf import OmegaConf

from automlagent.tools_registry import register_tool, list_tools, add_tool_tutorials

def get_user_input(prompt: str, required: bool = True, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("This field is required. Please provide a value.")

def get_list_input(prompt: str, required: bool = False) -> list:
    """Get a list of items from user input."""
    print(f"\n{prompt} (Enter empty line to finish)")
    items = []
    while True:
        item = input("- ").strip()
        if not item:
            if not items and required:
                print("At least one item is required.")
                continue
            break
        items.append(item)
    return items

def get_llm_config() -> dict:
    """Get LLM configuration from YAML config file."""
    config_path = get_user_input("Path to LLM config file (YAML)")

    with open(config_path, 'r') as f:
        config = OmegaConf.load(f)
    return config.llm

def register_tool_interactive():
    """Interactive function to register a new ML tool."""
    print("\n=== ML Tool Registration ===\n")
    
    # Get existing tools and create catalog if it doesn't exist
    try:
        existing_tools = list_tools()
        if existing_tools:
            print("Existing tools:", ", ".join(existing_tools))
            print()
    except FileNotFoundError:
        # Create catalog.json if it doesn't exist
        catalog_path = Path(__file__).parent.parent / "agent" / "src" / "_common" / "catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(catalog_path, "w") as f:
            json.dump({"tools": {}}, f, indent=2)
        existing_tools = []

    # Get basic tool information
    name = get_user_input("Tool name")
    if name in existing_tools:
        print(f"Error: Tool '{name}' already exists.")
        return
        
    version = get_user_input("Version", default="0.1.0")
    description = get_user_input("Description")

    # Get optional information
    print("\nFeatures (e.g., 'classification', 'regression', etc.)")
    features = get_list_input("Enter tool features")
    
    print("\nRequirements (e.g., 'numpy>=1.20.0', 'torch>=1.9.0', etc.)")
    requirements = get_list_input("Enter tool requirements")
    
    print("\nPrompt templates (enter template strings for tool usage)")
    prompt_template = get_list_input("Enter prompt templates")

    # Get tutorials path and condensing options
    tutorials_path = get_user_input("Path to tutorials directory (optional)", required=False)
    tutorials_path = Path(tutorials_path) if tutorials_path else None

    condense_tutorials = False
    llm_config = None
    max_length = 9999

    if tutorials_path:
        condense = input("\nDo you want to create condensed versions of tutorials? (y/N): ").lower()
        if condense == 'y':
            condense_tutorials = True
            llm_config = get_llm_config()
            max_length = int(get_user_input("Maximum length for condensed tutorials", default="9999"))

    # Confirm registration
    print("\nTool Registration Summary:")
    print(f"Name: {name}")
    print(f"Version: {version}")
    print(f"Description: {description}")
    print(f"Features: {', '.join(features) if features else 'None'}")
    print(f"Requirements: {', '.join(requirements) if requirements else 'None'}")
    print(f"Prompt Templates: {len(prompt_template)} templates")
    print(f"Tutorials Path: {tutorials_path or 'None'}")
    if tutorials_path and condense_tutorials:
        print("Creating condensed tutorials: Yes")
        print(f"LLM Config Path: {llm_config}")
        print(f"Max Length: {max_length}")
    
    confirm = input("\nProceed with registration? (y/N): ").lower()
    if confirm != 'y':
        print("Registration cancelled.")
        return

    # Register the tool first
    register_tool(
        name=name,
        version=version,
        description=description,
        features=features,
        requirements=requirements,
        prompt_template=prompt_template
    )

    # Add tutorials separately if provided
    if tutorials_path:
        add_tool_tutorials(
            tool_name=name,
            tutorials_source=tutorials_path,
            condense=condense_tutorials,
            llm_config=llm_config,
            max_length=max_length
        )

    print(f"\nSuccessfully registered tool: {name}")

def main():
    parser = argparse.ArgumentParser(description="Interactive ML Tool Registration")
    args = parser.parse_args()

    try:
        register_tool_interactive()
    except KeyboardInterrupt:
        print("\nRegistration cancelled.")
        sys.exit(1)

if __name__ == "__main__":
    main()
