"""Tool registry — central catalog of available agent tools."""
from __future__ import annotations

from typing import Dict, List, Optional

from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.course_tools import (
    CheckEligibilityTool,
    GetCourseDetailsTool,
    GetFeeStructureTool,
    ScheduleDemoTool,
    SearchCoursesTool,
)
from src.logger import setup_logger

logger = setup_logger(__name__)


class ToolRegistry:
    """Registry of all available tools the agent can call.

    Provides:
    - Registration / deregistration
    - Lookup by name
    - OpenAI-style tool schema for LLM
    - Execution with parameter validation
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default tool set."""
        default_tools: List[Tool] = [
            SearchCoursesTool(),
            GetCourseDetailsTool(),
            CheckEligibilityTool(),
            GetFeeStructureTool(),
            ScheduleDemoTool(),
        ]
        for tool in default_tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """Remove a tool by name.

        Returns:
            True if the tool was registered and removed.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name, or None if not registered."""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools as a list."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def get_tools_schema(self) -> List[Dict]:
        """Return the OpenAI-style tool schema for all registered tools.

        The schema is what the LLM uses to decide when to call each tool.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Validates parameters first; on failure, returns a ToolResult with
        success=False and an error message.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            ToolResult with success/data/error populated.
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {tool_name}",
                tool_name=tool_name,
            )

        valid, error = tool.validate_parameters(arguments)
        if not valid:
            return ToolResult(
                success=False,
                data=None,
                error=error,
                tool_name=tool_name,
            )

        return await tool.execute_with_timeout(**arguments)


# Global default registry — used by the orchestrator
tool_registry = ToolRegistry()
