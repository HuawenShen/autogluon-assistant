import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolsRegistry:
    def __init__(self):
        self.registry_path = Path(__file__).parent
        self.catalog_path = self.registry_path / "_common" / "catalog.json"
        self._tools_cache: Optional[Dict] = None

    @property
    def tools(self) -> Dict:
        """
        Lazy loading of tools information from catalog and individual tool.json files.
        
        Returns:
            Dict: Dictionary containing comprehensive tool information
        """
        if self._tools_cache is None:
            self._load_tools()
        return self._tools_cache

    def _load_tools(self) -> None:
        """
        Load and cache tools information from catalog.json and individual tool.json files.
        """
        try:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception as e:
            logger.error(f"Error loading catalog.json: {e}")
            raise

        tools_info = {}
        for tool_name, tool_data in catalog["tools"].items():
            tool_path = self.registry_path / tool_data["path"] / "tool.json"
            tool_info = {
                "name": tool_name,
                "path": tool_data["path"],
                "version": tool_data["version"],
                "description": tool_data["description"],
            }

            try:
                with open(tool_path, "r") as f:
                    tool_json = json.load(f)
                    # Update with additional information from tool.json
                    tool_info.update(
                        {
                            "features": tool_json.get("features", []),
                            "requirements": tool_json.get("requirements", []),
                            "prompt_template": tool_json.get(
                                "prompt_template", []
                            ),  # Add prompt_template
                        }
                    )
            except Exception as e:
                logger.warning(f"Error loading tool.json for {tool_name}: {e}")
                tool_info.update(
                    {
                        "features": [],
                        "requirements": [],
                        "prompt_template": [],  # Add empty prompt_template for failed loads
                    }
                )

            tools_info[tool_name] = tool_info

        self._tools_cache = tools_info

    def get_tool(self, tool_name: str) -> Optional[Dict]:
        """
        Get information for a specific tool.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Optional[Dict]: Tool information if found, None otherwise
        """
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List[str]: List of tool names
        """
        return list(self.tools.keys())

    def get_tool_path(self, tool_name: str) -> Optional[Path]:
        """
        Get the filesystem path for a tool's directory.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Optional[Path]: Path to tool directory if found, None otherwise
        """
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return self.registry_path / tool_info["path"]
        return None

    def get_tool_version(self, tool_name: str) -> Optional[str]:
        """
        Get the version of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Optional[str]: Tool version if found, None otherwise
        """
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return tool_info["version"]
        return None

    def get_tool_prompt_template(self, tool_name: str) -> Optional[List[str]]:
        """
        Get the prompt template for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Optional[List[str]]: Tool's prompt template if found, None otherwise
        """
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return tool_info.get("prompt_template")
        return None

    def get_tools_by_feature(self, feature: str) -> List[str]:
        """
        Get list of tools that have a specific feature.
        
        Args:
            feature: Feature to search for
            
        Returns:
            List[str]: List of tool names with the specified feature
        """
        return [
            tool_name
            for tool_name, tool_info in self.tools.items()
            if feature.lower() in [f.lower() for f in tool_info.get("features", [])]
        ]

    def get_tool_tutorials_folder(self, tool_name: str) -> List[Dict[str, str]]:
        """
        Get all tutorials for a specific tool, including those in nested directories.
        
        Args:
            tool_name: Name of the tool
                
        Raises:
            FileNotFoundError: If tool directory or tutorials directory doesn't exist
        """
        tool_path = self.get_tool_path(tool_name)
        if not tool_path:
            raise FileNotFoundError(f"Tool '{tool_name}' not found")

        tutorials_folder = tool_path / "tutorials"
        if not tutorials_folder.exists():
            raise FileNotFoundError(
                f"No tutorials directory found for tool '{tool_name}'"
            )

        return tutorials_folder


# Create singleton instance
registry = ToolsRegistry()

# Export commonly used functions at module level
get_tool = registry.get_tool
list_tools = registry.list_tools
get_tool_path = registry.get_tool_path
get_tool_version = registry.get_tool_version
get_tools_by_feature = registry.get_tools_by_feature
get_tool_tutorials_folder = registry.get_tool_tutorials_folder
get_tool_prompt_template = registry.get_tool_prompt_template
