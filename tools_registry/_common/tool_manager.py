import os
import json
import shutil
from pathlib import Path

class ToolRegistryManager:
    def __init__(self):
        self.common_dir = Path(__file__).parent
        self.base_path = self.common_dir.parent
        self.catalog_path = self.common_dir / "catalog.json"
        self.ensure_base_structure()

    def ensure_base_structure(self):
        """Create catalog if it doesn't exist or is empty."""
        if not self.catalog_path.exists() or os.path.getsize(self.catalog_path) == 0:
            self.save_catalog({"tools": {}})

    def save_catalog(self, catalog_data):
        """Save the catalog to disk."""
        with open(self.catalog_path, 'w') as f:
            json.dump(catalog_data, f, indent=2)

    def load_catalog(self):
        """Load the existing catalog."""
        try:
            with open(self.catalog_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            empty_catalog = {"tools": {}}
            self.save_catalog(empty_catalog)
            return empty_catalog

    def add_tool_interactive(self):
        """Interactive prompt to add a new tool."""
        print("\n=== Adding New Tool to Registry ===\n")
        
        tool_name = input("Tool name: ").strip()
        if not tool_name:
            print("Tool name cannot be empty.")
            return

        tool_dir = self.base_path / tool_name
        if tool_dir.exists():
            print(f"Tool '{tool_name}' already exists!")
            return

        # Gather all tool information
        tool_info = {
            "name": tool_name,
            "version": input("Version: ").strip(),
            "description": input("Description: ").strip(),
            "requirements": [],
            "features": []
        }

        # Gather requirements
        print("\nEnter requirements (one per line, empty line to finish):")
        while True:
            req = input("> ").strip()
            if not req:
                break
            tool_info["requirements"].append(req)

        # Gather features
        print("\nEnter features (one per line, empty line to finish):")
        while True:
            feature = input("> ").strip()
            if not feature:
                break
            tool_info["features"].append(feature)

        # Create directory structure and save tool info
        tool_dir.mkdir(parents=True)
        (tool_dir / "tutorials").mkdir()
        (tool_dir / "examples").mkdir()

        # Save combined tool info
        with open(tool_dir / "tool.json", 'w') as f:
            json.dump(tool_info, f, indent=2)

        # Create basic tutorial template
        with open(tool_dir / "tutorials" / "quickstart.md", 'w') as f:
            f.write(f"# {tool_name} Quickstart Guide\n\n## Installation\n\n## Basic Usage\n\n## Examples")

        # Update catalog
        catalog = self.load_catalog()
        catalog["tools"][tool_name] = {
            "path": str(tool_dir.relative_to(self.base_path)),
            "version": tool_info["version"],
            "description": tool_info["description"]
        }
        self.save_catalog(catalog)

        print(f"\nSuccessfully added {tool_name} to the registry!")

    def remove_tool_interactive(self):
        """Interactive prompt to remove a tool."""
        print("\n=== Removing Tool from Registry ===\n")
        
        catalog = self.load_catalog()
        if not catalog["tools"]:
            print("No tools in registry to remove.")
            return

        # Show available tools
        print("Available tools:")
        for idx, (name, info) in enumerate(catalog["tools"].items(), 1):
            print(f"{idx}. {name} - {info['description']}")

        try:
            choice = int(input("\nEnter tool number to remove (0 to cancel): "))
            if choice == 0:
                return
            if choice < 1 or choice > len(catalog["tools"]):
                print("Invalid choice!")
                return
            
            # Get tool name from choice
            tool_name = list(catalog["tools"].keys())[choice - 1]
            
            # Confirm deletion
            confirm = input(f"\nAre you sure you want to remove '{tool_name}'? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Removal cancelled.")
                return

            # Remove tool directory
            tool_dir = self.base_path / tool_name
            if tool_dir.exists():
                shutil.rmtree(tool_dir)

            # Update catalog
            del catalog["tools"][tool_name]
            self.save_catalog(catalog)

            print(f"\nSuccessfully removed {tool_name} from the registry!")

        except (ValueError, IndexError):
            print("Invalid input!")
            return

if __name__ == "__main__":
    manager = ToolRegistryManager()
    while True:
        print("\n1. Add new tool")
        print("2. Remove tool")
        print("3. Exit")
        choice = input("\nSelect an option: ").strip()
        
        if choice == "1":
            manager.add_tool_interactive()
        elif choice == "2":
            manager.remove_tool_interactive()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")
