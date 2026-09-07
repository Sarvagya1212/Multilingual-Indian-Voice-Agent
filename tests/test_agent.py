"""Tests for the agent module — state machine, tools, memory, orchestrator."""
from datetime import datetime, timedelta

import pytest

from src.agent.state_machine import AgentStateMachine, StateTransition
from src.agent.memory.short_term import ShortTermMemory, ConversationTurn
from src.agent.memory.long_term import LongTermMemory
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.course_tools import (
    SearchCoursesTool,
    GetCourseDetailsTool,
    CheckEligibilityTool,
    GetFeeStructureTool,
    ScheduleDemoTool,
)
from src.agent.tools.tool_registry import ToolRegistry, tool_registry as default_registry
from src.pipeline.types import AgentState


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestAgentStateMachine:
    def test_initial_state(self):
        """State machine should start at IDLE."""
        sm = AgentStateMachine()
        assert sm.state == AgentState.IDLE

    def test_valid_transition(self):
        """Valid transitions should be allowed."""
        sm = AgentStateMachine()
        assert sm.transition(AgentState.LISTENING) is True
        assert sm.state == AgentState.LISTENING

    def test_invalid_transition(self):
        """Invalid transitions should be rejected."""
        sm = AgentStateMachine()
        # IDLE -> SPEAKING is not valid
        assert sm.transition(AgentState.SPEAKING) is False
        assert sm.state == AgentState.IDLE

    def test_full_pipeline_flow(self):
        """Full happy-path pipeline flow should work."""
        sm = AgentStateMachine()
        assert sm.transition(AgentState.LISTENING) is True
        assert sm.transition(AgentState.TRANSCRIBING) is True
        assert sm.transition(AgentState.THINKING) is True
        assert sm.transition(AgentState.GENERATING) is True
        assert sm.transition(AgentState.SPEAKING) is True
        assert sm.transition(AgentState.LISTENING) is True

    def test_tool_call_flow(self):
        """THINKING -> CALLING_TOOL -> GENERATING should be valid."""
        sm = AgentStateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.THINKING)
        assert sm.transition(AgentState.CALLING_TOOL) is True
        assert sm.transition(AgentState.GENERATING) is True

    def test_can_interrupt(self):
        """SPEAKING/GENERATING/CALLING_TOOL should be interruptible."""
        sm = AgentStateMachine()
        # IDLE is not interruptible
        assert sm.can_interrupt() is False

        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.THINKING)
        sm.transition(AgentState.GENERATING)
        assert sm.can_interrupt() is True

    def test_force_interrupt(self):
        """force_interrupt should transition to INTERRUPTED then LISTENING."""
        sm = AgentStateMachine()
        sm.transition(AgentState.LISTENING)
        sm.transition(AgentState.TRANSCRIBING)
        sm.transition(AgentState.THINKING)
        sm.transition(AgentState.GENERATING)
        assert sm.force_interrupt() is True
        assert sm.state == AgentState.LISTENING
        # INTERRUPTED should be in history
        interrupted_count = sum(
            1 for _, s, _ in sm.state_history if s == AgentState.INTERRUPTED
        )
        assert interrupted_count == 1

    def test_force_interrupt_when_idle(self):
        """force_interrupt should return False when not interruptible."""
        sm = AgentStateMachine()
        assert sm.force_interrupt() is False

    def test_reset(self):
        """reset() should return to IDLE."""
        sm = AgentStateMachine()
        sm.transition(AgentState.LISTENING)
        sm.reset()
        assert sm.state == AgentState.IDLE

    def test_history_tracks_reason(self):
        """History should record the reason for each transition."""
        sm = AgentStateMachine()
        sm.transition(AgentState.LISTENING, "user_connected")
        sm.transition(AgentState.TRANSCRIBING, "speech_detected")
        assert len(sm.state_history) == 2
        ts, from_state, reason = sm.state_history[1]
        assert from_state == AgentState.LISTENING
        assert reason == "speech_detected"

    def test_get_history_format(self):
        """get_history should return ISO-formatted entries."""
        sm = AgentStateMachine()
        sm.transition(AgentState.LISTENING)
        history = sm.get_history()
        assert len(history) == 1
        ts_str, from_state, _ = history[0]
        # Should be parseable as ISO
        datetime.fromisoformat(ts_str)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestShortTermMemory:
    def test_empty_memory(self):
        """Empty memory should have no turns and no context."""
        m = ShortTermMemory()
        assert len(m) == 0
        assert m.user_context == {}

    def test_add_turn(self):
        """add_turn should append to turns list."""
        m = ShortTermMemory()
        m.add_turn("Hi", "Hello!", "en")
        assert len(m) == 1
        assert m.turns[0].user_message == "Hi"

    def test_max_turns_trims(self):
        """Memory should trim to max_turns."""
        m = ShortTermMemory(max_turns=3)
        for i in range(5):
            m.add_turn(f"msg {i}", f"reply {i}", "en")
        assert len(m) == 3
        assert m.turns[-1].user_message == "msg 4"

    def test_language_preference(self):
        """preferred_language should be tracked in context."""
        m = ShortTermMemory()
        m.add_turn("hi", "hi", "hi")
        assert m.user_context["preferred_language"] == "hi"

    def test_class_extraction(self):
        """User's class should be extracted from messages."""
        m = ShortTermMemory()
        m.add_turn("I am in class 12", "ok", "en")
        assert m.user_context["current_class"] == "Class 12"

    def test_interest_extraction_engineering(self):
        """JEE/engineering interest should be tracked."""
        m = ShortTermMemory()
        m.add_turn("I want to do JEE prep", "ok", "en")
        assert m.user_context["interest"] == "engineering"

    def test_interest_extraction_medical(self):
        """NEET/medical interest should be tracked."""
        m = ShortTermMemory()
        m.add_turn("I want to be a doctor", "ok", "en")
        assert m.user_context["interest"] == "medical"

    def test_recent_context_format(self):
        """get_recent_context should return a string with User/Agent lines."""
        m = ShortTermMemory()
        m.add_turn("Q1", "A1", "en")
        m.add_turn("Q2", "A2", "en")
        ctx = m.get_recent_context(num_turns=2)
        assert "User: Q1" in ctx
        assert "Agent: A1" in ctx

    def test_recent_context_empty(self):
        """Empty memory should return the no-context message."""
        m = ShortTermMemory()
        assert m.get_recent_context() == "No previous context."

    def test_expiry(self):
        """is_expired should return True after TTL."""
        m = ShortTermMemory(ttl_minutes=0)
        # ttl=0 means any time after session start
        m.session_start = datetime.now() - timedelta(minutes=1)
        assert m.is_expired() is True

    def test_clear(self):
        """clear should reset everything."""
        m = ShortTermMemory()
        m.add_turn("hi", "hi", "en")
        m.clear()
        assert len(m) == 0
        assert m.user_context == {}

    def test_context_summary(self):
        """get_context_summary should return structured metadata."""
        m = ShortTermMemory()
        m.add_turn("hi", "hi", "en")
        m.add_turn("namaste", "namaste", "hi")
        summary = m.get_context_summary()
        assert summary["turns_count"] == 2
        assert "en" in summary["languages_used"]
        assert "hi" in summary["languages_used"]


class TestLongTermMemory:
    def test_load_missing_user(self):
        """Loading a non-existent user should return empty dict."""
        ltm = LongTermMemory()
        assert ltm.load_profile("unknown") == {}

    def test_save_and_load_profile(self):
        """Saved profile should be retrievable."""
        ltm = LongTermMemory()
        ltm.save_profile("user-1", {"name": "Aarav", "interest": "engineering"})
        profile = ltm.load_profile("user-1")
        assert profile["name"] == "Aarav"
        assert profile["interest"] == "engineering"
        assert "_updated_at" in profile

    def test_list_users(self):
        """list_users should return all known user IDs."""
        ltm = LongTermMemory()
        ltm.save_profile("u1", {})
        ltm.save_profile("u2", {})
        assert "u1" in ltm.list_users()
        assert "u2" in ltm.list_users()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TestToolBase:
    def test_validate_missing_required(self):
        """Tool.validate_parameters should return False on missing required."""
        class MyTool(Tool):
            name = "my_tool"
            description = "test"
            parameters = {"type": "object", "required": ["foo"], "properties": {}}

            async def execute(self, **kwargs):
                return ToolResult(success=True, data=None)

        tool = MyTool()
        valid, err = tool.validate_parameters({})
        assert valid is False
        assert "foo" in err

    def test_validate_all_present(self):
        """Validation should pass when all required params present."""
        class MyTool(Tool):
            name = "my_tool"
            description = "test"
            parameters = {"type": "object", "required": ["foo"], "properties": {}}

            async def execute(self, **kwargs):
                return ToolResult(success=True, data=None)

        tool = MyTool()
        valid, err = tool.validate_parameters({"foo": "bar"})
        assert valid is True
        assert err is None

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        """Successful execution should return success=True."""
        class MyTool(Tool):
            name = "quick"
            description = "test"

            async def execute(self, **kwargs):
                return ToolResult(success=True, data="ok")

        tool = MyTool()
        result = await tool.execute_with_timeout(timeout=1.0)
        assert result.success is True
        assert result.data == "ok"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_failure(self):
        """Failed execution should return success=False with error."""
        class MyTool(Tool):
            name = "fails"
            description = "test"

            async def execute(self, **kwargs):
                raise ValueError("oops")

        tool = MyTool()
        result = await tool.execute_with_timeout(timeout=1.0)
        assert result.success is False
        assert "oops" in result.error


class TestSearchCoursesTool:
    @pytest.mark.asyncio
    async def test_search_jee(self):
        """Search for 'JEE' should return engineering course."""
        tool = SearchCoursesTool()
        result = await tool.execute(query="JEE")
        assert result.success is True
        assert result.data["count"] >= 1
        assert any("JEE" in c["name"] for c in result.data["courses"])

    @pytest.mark.asyncio
    async def test_search_neet(self):
        """Search for 'NEET' should return medical course."""
        tool = SearchCoursesTool()
        result = await tool.execute(query="NEET")
        assert result.success is True
        assert any("NEET" in c["name"] for c in result.data["courses"])

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Search for non-existent topic should return empty list."""
        tool = SearchCoursesTool()
        result = await tool.execute(query="xyz123nonexistent")
        assert result.success is True
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        """limit parameter should cap results."""
        tool = SearchCoursesTool()
        result = await tool.execute(query="course", limit=1)
        assert result.data["count"] <= 1


class TestGetCourseDetailsTool:
    @pytest.mark.asyncio
    async def test_existing_course(self):
        """Get details for existing course."""
        tool = GetCourseDetailsTool()
        result = await tool.execute(course_id="jee-2024")
        assert result.success is True
        assert result.data["id"] == "jee-2024"
        assert "syllabus" in result.data

    @pytest.mark.asyncio
    async def test_missing_course(self):
        """Get details for missing course should fail."""
        tool = GetCourseDetailsTool()
        result = await tool.execute(course_id="unknown")
        assert result.success is False
        assert "not found" in result.error


class TestCheckEligibilityTool:
    @pytest.mark.asyncio
    async def test_jee_class_12_eligible(self):
        """Class 12 student should be eligible for JEE."""
        tool = CheckEligibilityTool()
        result = await tool.execute(course_id="jee-2024", current_class="12")
        assert result.success is True
        assert result.data["eligible"] is True

    @pytest.mark.asyncio
    async def test_jee_class_9_not_eligible(self):
        """Class 9 student should NOT be eligible for JEE."""
        tool = CheckEligibilityTool()
        result = await tool.execute(course_id="jee-2024", current_class="9")
        assert result.success is True
        assert result.data["eligible"] is False
        assert len(result.data["reasons"]) > 0


class TestGetFeeStructureTool:
    @pytest.mark.asyncio
    async def test_jee_fee(self):
        """JEE fee should have installments and scholarship."""
        tool = GetFeeStructureTool()
        result = await tool.execute(course_id="jee-2024")
        assert result.success is True
        assert result.data["total"] == 150000
        assert len(result.data["installments"]) == 2
        assert "scholarship" in result.data


class TestScheduleDemoTool:
    @pytest.mark.asyncio
    async def test_schedule_with_phone(self):
        """Scheduling with phone should succeed."""
        tool = ScheduleDemoTool()
        result = await tool.execute(
            course_id="jee-2024",
            phone="9876543210",
        )
        assert result.success is True
        assert "demo_id" in result.data
        assert result.data["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_schedule_missing_course(self):
        """Scheduling with invalid course should fail."""
        tool = ScheduleDemoTool()
        result = await tool.execute(course_id="invalid", phone="9876543210")
        assert result.success is False


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_default_registry_has_tools(self):
        """Default registry should have the 5 default tools."""
        names = default_registry.list_names()
        assert "search_courses" in names
        assert "get_course_details" in names
        assert "check_eligibility" in names
        assert "get_fee_structure" in names
        assert "schedule_demo" in names

    def test_register_custom(self):
        """A custom tool can be registered."""

        class MyTool(Tool):
            name = "custom"
            description = "test"

            async def execute(self, **kwargs):
                return ToolResult(success=True, data="x")

        r = ToolRegistry()
        r.register(MyTool())
        assert r.get("custom") is not None

    def test_get_all(self):
        """get_all should return a list of Tool instances."""
        r = ToolRegistry()
        tools = r.get_all()
        assert all(isinstance(t, Tool) for t in tools)
        assert len(tools) >= 5

    def test_unregister(self):
        """unregister should remove a tool."""
        r = ToolRegistry()
        r.unregister("search_courses")
        assert r.get("search_courses") is None

    def test_get_tools_schema(self):
        """Schema should be in OpenAI format."""
        r = ToolRegistry()
        schema = r.get_tools_schema()
        assert len(schema) >= 5
        for entry in schema:
            assert entry["type"] == "function"
            assert "name" in entry["function"]
            assert "description" in entry["function"]
            assert "parameters" in entry["function"]

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """execute_tool should run a registered tool."""
        r = ToolRegistry()
        result = await r.execute_tool("search_courses", {"query": "JEE"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Executing an unknown tool should return an error result."""
        r = ToolRegistry()
        result = await r.execute_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_invalid_params(self):
        """Executing with invalid params should return a validation error."""
        r = ToolRegistry()
        result = await r.execute_tool("search_courses", {})  # missing 'query'
        assert result.success is False


# Run with: pytest tests/test_agent.py -v
