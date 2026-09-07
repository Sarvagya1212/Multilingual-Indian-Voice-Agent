"""Tools module — base classes and concrete tool implementations."""
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.tool_registry import ToolRegistry, tool_registry
from src.agent.tools.course_tools import (
    SearchCoursesTool,
    GetCourseDetailsTool,
    CheckEligibilityTool,
    GetFeeStructureTool,
    ScheduleDemoTool,
)

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "SearchCoursesTool",
    "GetCourseDetailsTool",
    "CheckEligibilityTool",
    "GetFeeStructureTool",
    "ScheduleDemoTool",
]
