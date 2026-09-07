"""Agent evaluation metrics — task completion, tool accuracy, latency."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskCompletion(Enum):
    """Task outcome classification."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class AgentMetrics:
    """Metrics for a single agent conversation evaluation."""

    conversation_id: str
    task_completion: TaskCompletion
    tool_call_accuracy: float  # 0-1: fraction of expected tools called
    response_relevance: float  # 0-1: keyword overlap with user goal
    instruction_following: float  # 0-1: combined score
    turns_to_complete: int
    total_latency_ms: float
    tools_called: list[str]
    expected_tools: list[str]
    interruptions: int
    language: str
    final_response: str

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "task_completion": self.task_completion.value,
            "tool_call_accuracy": self.tool_call_accuracy,
            "response_relevance": self.response_relevance,
            "instruction_following": self.instruction_following,
            "turns_to_complete": self.turns_to_complete,
            "total_latency_ms": self.total_latency_ms,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "interruptions": self.interruptions,
            "language": self.language,
        }


class AgentEvaluator:
    """Aggregates agent conversation evaluations."""

    def __init__(self):
        self.results: list[AgentMetrics] = []
        self.conversations: list[dict] = []

    def record_conversation(self, conversation: dict) -> None:
        """Record a full conversation for later inspection."""
        self.conversations.append(conversation)

    def evaluate_conversation(
        self,
        conversation_id: str,
        user_goal: str,
        expected_tools: list[str],
        actual_tools: list[str],
        final_response: str,
        num_turns: int,
        latency_ms: float,
        language: str = "en",
        interruptions: int = 0,
    ) -> AgentMetrics:
        """Evaluate a single agent conversation.

        Args:
            conversation_id: Unique conversation ID.
            user_goal: The user's original goal/question.
            expected_tools: Tools the agent should have called.
            actual_tools: Tools the agent actually called.
            final_response: The agent's final response.
            num_turns: Number of turns in the conversation.
            latency_ms: Total conversation latency.
            language: Language code.
            interruptions: Number of interruptions.

        Returns:
            AgentMetrics for this conversation.
        """
        # Tool call accuracy (recall-based): how many expected tools were called
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        if expected_set:
            tool_accuracy = len(expected_set & actual_set) / len(expected_set)
        else:
            tool_accuracy = 1.0 if not actual_set else 0.5  # No tools needed

        # Response relevance: keyword overlap with user goal
        goal_words = set(_normalize(user_goal).split())
        response_words = set(_normalize(final_response).split())
        if goal_words:
            overlap = goal_words & response_words
            relevance = len(overlap) / len(goal_words)
        else:
            relevance = 0.0

        # Instruction following: combined metric
        instruction_following = min(tool_accuracy + relevance * 0.5, 1.0)

        # Task completion classification
        if tool_accuracy >= 0.8 and relevance >= 0.4:
            completion = TaskCompletion.SUCCESS
        elif tool_accuracy >= 0.5 or relevance >= 0.25:
            completion = TaskCompletion.PARTIAL
        else:
            completion = TaskCompletion.FAILED

        metrics = AgentMetrics(
            conversation_id=conversation_id,
            task_completion=completion,
            tool_call_accuracy=tool_accuracy,
            response_relevance=relevance,
            instruction_following=instruction_following,
            turns_to_complete=num_turns,
            total_latency_ms=latency_ms,
            tools_called=actual_tools,
            expected_tools=expected_tools,
            interruptions=interruptions,
            language=language,
            final_response=final_response,
        )
        self.results.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        """Aggregate agent metrics."""
        if not self.results:
            return {}
        n = len(self.results)
        completion_counts = {
            "success": sum(1 for r in self.results if r.task_completion == TaskCompletion.SUCCESS),
            "partial": sum(1 for r in self.results if r.task_completion == TaskCompletion.PARTIAL),
            "failed": sum(1 for r in self.results if r.task_completion == TaskCompletion.FAILED),
        }
        return {
            "total_conversations": n,
            "task_completion_rate": completion_counts["success"] / n * 100,
            "task_completion_rate_partial": (completion_counts["success"] + completion_counts["partial"]) / n * 100,
            "avg_tool_accuracy": sum(r.tool_call_accuracy for r in self.results) / n * 100,
            "avg_response_relevance": sum(r.response_relevance for r in self.results) / n * 100,
            "avg_instruction_following": sum(r.instruction_following for r in self.results) / n * 100,
            "avg_turns": sum(r.turns_to_complete for r in self.results) / n,
            "avg_latency_ms": sum(r.total_latency_ms for r in self.results) / n,
            "total_interruptions": sum(r.interruptions for r in self.results),
            "completion_breakdown": completion_counts,
        }

    def get_per_language_summary(self) -> dict[str, dict]:
        """Per-language breakdown."""
        by_lang: dict[str, list[AgentMetrics]] = {}
        for r in self.results:
            by_lang.setdefault(r.language, []).append(r)

        result = {}
        for lang, group in by_lang.items():
            n = len(group)
            result[lang] = {
                "count": n,
                "task_completion_rate": sum(
                    1 for r in group if r.task_completion == TaskCompletion.SUCCESS
                ) / n * 100,
                "avg_tool_accuracy": sum(r.tool_call_accuracy for r in group) / n * 100,
                "avg_turns": sum(r.turns_to_complete for r in group) / n,
            }
        return result

    def clear(self) -> None:
        self.results = []
        self.conversations = []


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", text.lower().strip())
