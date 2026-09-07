"""Base class for agent tools."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_name: Optional[str] = None


class Tool(ABC):
    """Abstract base class for agent tools.

    Concrete tools must define:
    - `name`: unique tool identifier (used by the LLM to call the tool)
    - `description`: what the tool does (used by the LLM to choose it)
    - `parameters`: JSON schema describing accepted arguments
    - `async execute(**kwargs) -> ToolResult`
    """

    name: str = "base_tool"
    description: str = "A tool"
    parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        raise NotImplementedError

    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate that all required parameters are present.

        Args:
            params: Tool arguments.

        Returns:
            (is_valid, error_message_or_none)
        """
        required = self.parameters.get("required", [])
        for r in required:
            if r not in params:
                return False, f"Missing required parameter: {r}"
        return True, None

    async def execute_with_timeout(
        self,
        timeout: float = 30.0,
        **kwargs,
    ) -> ToolResult:
        """Execute the tool with a timeout.

        Args:
            timeout: Maximum execution time in seconds.
            **kwargs: Tool arguments.

        Returns:
            ToolResult with success=False on timeout or exception.
        """
        start = datetime.now()
        try:
            # asyncio.wait_for works on Python 3.10; asyncio.timeout requires 3.11+
            return await asyncio.wait_for(self.execute(**kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start).total_seconds() * 1000
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution timed out after {timeout}s",
                execution_time_ms=execution_time,
                tool_name=self.name,
            )
        except Exception as e:
            execution_time = (datetime.now() - start).total_seconds() * 1000
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time,
                tool_name=self.name,
            )
