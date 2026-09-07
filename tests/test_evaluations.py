"""Tests for the evaluation framework — STT, TTS, RAG, and agent evaluators."""
import pytest
import pytest_asyncio

from evaluations.stt.metrics import (
    STTEvaluator,
    compute_cer,
    compute_wer,
    _levenshtein,
)
from evaluations.tts.metrics import TTSEvaluator
from evaluations.rag.metrics import RAGEvaluator
from evaluations.agent.metrics import (
    AgentEvaluator,
    TaskCompletion,
    _normalize,
)
from evaluations.stt.evaluator import STTTestCase
from evaluations.tts.evaluator import TTSTestCase
from evaluations.rag.evaluator import RAGTestCase


# ---------------------------------------------------------------------------
# Levenshtein helper
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_empty_inputs(self):
        assert _levenshtein([], []) == 0

    def test_one_empty(self):
        assert _levenshtein([], ["a"]) == 1
        assert _levenshtein(["a"], []) == 1

    def test_identical(self):
        assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_one_substitution(self):
        assert _levenshtein(["a", "b"], ["a", "c"]) == 1

    def test_insertions_deletions(self):
        assert _levenshtein(["a", "b"], ["a", "b", "c"]) == 1  # insertion
        assert _levenshtein(["a", "b", "c"], ["a", "b"]) == 1  # deletion

    def test_complex(self):
        # kitten -> sitting = 3 (k->s, e->i, insert g)
        assert _levenshtein(list("kitten"), list("sitting")) == 3


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------

class TestWerCer:
    def test_wer_perfect(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_wer_one_wrong(self):
        # 1 of 2 words wrong
        assert compute_wer("hello world", "hello there") == 0.5

    def test_wer_all_wrong(self):
        assert compute_wer("hello", "goodbye") == 1.0

    def test_wer_empty_reference(self):
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "hello") == 1.0
        assert compute_wer("hello", "") == 1.0

    def test_wer_case_insensitive(self):
        assert compute_wer("Hello World", "hello world") == 0.0

    def test_cer_perfect(self):
        assert compute_cer("hello", "hello") == 0.0

    def test_cer_one_char_off(self):
        # 1 of 5 chars wrong
        assert compute_cer("hello", "hallo") == 0.2

    def test_cer_empty(self):
        assert compute_cer("", "") == 0.0
        assert compute_cer("hello", "") == 1.0


# ---------------------------------------------------------------------------
# STTEvaluator
# ---------------------------------------------------------------------------

class TestSTTEvaluator:
    def test_empty_summary(self):
        ev = STTEvaluator()
        assert ev.get_summary() == {}

    def test_single_evaluation(self):
        ev = STTEvaluator()
        m = ev.evaluate(
            reference="hello world",
            hypothesis="hello there",
            language="en",
            detected_language="en",
            latency_ms=200.0,
            audio_duration=2.0,
        )
        assert m.wer == 0.5
        assert m.cer < 0.5
        assert m.language_correct is True
        assert m.real_time_factor == pytest.approx(0.1, abs=1e-3)

    def test_language_mismatch(self):
        ev = STTEvaluator()
        m = ev.evaluate(
            reference="namaste",
            hypothesis="namaste",
            language="hi",
            detected_language="en",  # Wrong
            latency_ms=200.0,
            audio_duration=2.0,
        )
        assert m.language_correct is False

    def test_summary_aggregates(self):
        ev = STTEvaluator()
        for i in range(3):
            ev.evaluate(
                reference=f"test {i}",
                hypothesis=f"test {i}",
                language="en",
                detected_language="en",
                latency_ms=200.0 + i * 50,
                audio_duration=2.0,
            )
        summary = ev.get_summary()
        assert summary["total_samples"] == 3
        # latencies: 200, 250, 300 → avg = 250
        assert summary["avg_latency_ms"] == 250.0
        assert summary["language_accuracy_pct"] == 100.0

    def test_percentiles(self):
        ev = STTEvaluator()
        for i in range(10):
            ev.evaluate(
                reference="x",
                hypothesis="x",
                language="en",
                detected_language="en",
                latency_ms=float(i * 100),
                audio_duration=1.0,
            )
        # latencies: 0, 100, 200, ..., 900
        summary = ev.get_summary()
        # p50 = (sorted index 4.5) = average of 400 and 500 = 450
        assert summary["p50_latency_ms"] == pytest.approx(450.0, abs=1e-3)
        # p90 = sorted index 8.1 = 800 + 0.1 * (900 - 800) = 810
        assert summary["p90_latency_ms"] == pytest.approx(810.0, abs=1e-3)

    def test_per_language_summary(self):
        ev = STTEvaluator()
        ev.evaluate("hello", "hello", "en", "en", 200.0, 1.0)
        ev.evaluate("hello", "hello", "en", "en", 200.0, 1.0)
        ev.evaluate("namaste", "namaste", "hi", "hi", 300.0, 1.5)
        per_lang = ev.get_per_language_summary()
        assert per_lang["en"]["count"] == 2
        assert per_lang["hi"]["count"] == 1
        assert per_lang["en"]["avg_latency_ms"] == 200.0
        assert per_lang["hi"]["avg_latency_ms"] == 300.0

    def test_clear(self):
        ev = STTEvaluator()
        ev.evaluate("x", "x", "en", "en", 200.0, 1.0)
        ev.clear()
        assert ev.results == []


class TestSTTTestCase:
    def test_create(self):
        case = STTTestCase(
            audio=b"\x00\x01",
            reference="hello",
            language="en",
            audio_duration=2.0,
        )
        assert case.audio == b"\x00\x01"
        assert case.reference == "hello"
        assert case.audio_duration == 2.0


# ---------------------------------------------------------------------------
# TTSEvaluator
# ---------------------------------------------------------------------------

class TestTTSEvaluator:
    def test_empty_summary(self):
        ev = TTSEvaluator()
        assert ev.get_summary() == {}

    def test_evaluate(self):
        ev = TTSEvaluator()
        m = ev.evaluate(
            text="JEE Main preparation",
            audio_bytes=b"\x00" * 1000,
            audio_duration=2.0,
            latency_ms=300.0,
        )
        assert m.text == "JEE Main preparation"
        assert m.audio_duration == 2.0
        # JEE appears once
        assert m.pronunciation_corrections == 1

    def test_multiple_acronyms(self):
        ev = TTSEvaluator()
        m = ev.evaluate(
            text="JEE and NEET coaching for IIT aspirants",
            audio_bytes=b"\x00" * 1000,
            audio_duration=3.0,
            latency_ms=400.0,
        )
        # JEE, NEET, IIT = 3 acronyms
        assert m.pronunciation_corrections == 3

    def test_no_acronyms(self):
        ev = TTSEvaluator()
        m = ev.evaluate(
            text="Hello world",
            audio_bytes=b"\x00" * 1000,
            audio_duration=1.5,
            latency_ms=200.0,
        )
        assert m.pronunciation_corrections == 0

    def test_rtf_calculation(self):
        ev = TTSEvaluator()
        m = ev.evaluate(
            text="test",
            audio_bytes=b"x" * 5000,
            audio_duration=5.0,
            latency_ms=500.0,  # 0.5s
        )
        # RTF = 0.5s / 5.0s = 0.1
        assert m.real_time_factor == pytest.approx(0.1, abs=1e-3)
        # bytes per second = 1000
        assert m.bytes_per_second == pytest.approx(1000.0, abs=1e-3)

    def test_summary(self):
        ev = TTSEvaluator()
        ev.evaluate("JEE", b"x" * 1000, 1.0, 100.0)
        ev.evaluate("NEET", b"x" * 2000, 2.0, 200.0)
        s = ev.get_summary()
        assert s["total_samples"] == 2
        assert s["avg_audio_duration"] == 1.5
        assert s["total_pronunciation_corrections"] == 2

    def test_silence_pad_detection(self):
        ev = TTSEvaluator()
        # Normalized text with single-letter "J E E" pads
        m = ev.evaluate(
            text="JEE",
            audio_bytes=b"x" * 1000,
            audio_duration=1.0,
            latency_ms=100.0,
            normalized_text="J E E",
        )
        assert m.has_silence_pads is True

    def test_clear(self):
        ev = TTSEvaluator()
        ev.evaluate("x", b"x", 1.0, 100.0)
        ev.clear()
        assert ev.results == []


class TestTTSTestCase:
    def test_default_bounds(self):
        case = TTSTestCase(text="hello")
        assert case.language == "en"
        assert case.expected_duration_min == 0.5
        assert case.expected_duration_max == 60.0


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------

class TestRAGEvaluator:
    def test_empty_summary(self):
        ev = RAGEvaluator()
        assert ev.get_summary() == {}

    def test_evaluate_no_ground_truth(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {"text": "JEE preparation", "metadata": {"source": "jee"}})(), "score": 0.8})()
        ]
        m = ev.evaluate(query="JEE", retrieved_chunks=chunks)
        assert m.retrieval_recall == 0.0
        assert m.context_relevance == 0.8

    def test_recall_perfect(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {"text": "...", "metadata": {"source": "jee"}})(), "score": 0.9})(),
            type("RC", (), {"chunk": type("C", (), {"text": "...", "metadata": {"source": "neet"}})(), "score": 0.7})(),
        ]
        m = ev.evaluate(
            query="q",
            retrieved_chunks=chunks,
            ground_truth_chunks=["jee"],
        )
        assert m.retrieval_recall == 1.0

    def test_recall_partial(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {"text": "...", "metadata": {"source": "other"}})(), "score": 0.9})(),
        ]
        m = ev.evaluate(
            query="q",
            retrieved_chunks=chunks,
            ground_truth_chunks=["jee", "neet"],
        )
        # 0 of 2 ground truths retrieved
        assert m.retrieval_recall == 0.0
        assert m.num_relevant == 2

    def test_groundedness_good(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {
                "text": "JEE coaching fee is Rs 150000 with installment facility",
                "metadata": {"source": "jee"},
            })(), "score": 0.9})(),
        ]
        m = ev.evaluate(
            query="fee",
            retrieved_chunks=chunks,
            generated_answer="JEE coaching is Rs 150000 with installments",
        )
        # Most content words appear in context
        assert m.answer_groundedness > 0.7
        assert m.hallucination_score < 0.3
        assert m.has_answer is True

    def test_groundedness_hallucinated(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {
                "text": "JEE coaching fee details",
                "metadata": {"source": "jee"},
            })(), "score": 0.9})(),
        ]
        m = ev.evaluate(
            query="q",
            retrieved_chunks=chunks,
            generated_answer="The astronaut training program costs 50000 dollars in Mars",
        )
        # Most content words NOT in context
        assert m.hallucination_score > 0.5

    def test_no_answer(self):
        ev = RAGEvaluator()
        m = ev.evaluate(
            query="q",
            retrieved_chunks=[],
            generated_answer="",
        )
        assert m.has_answer is False

    def test_no_context_high_hallucination(self):
        ev = RAGEvaluator()
        m = ev.evaluate(
            query="q",
            retrieved_chunks=[],
            generated_answer="I made this up entirely",
        )
        assert m.hallucination_score == 1.0
        assert m.answer_groundedness == 0.0

    def test_relevance_scores_override(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {"text": "...", "metadata": {}})(), "score": 0.5})(),
        ]
        m = ev.evaluate(
            query="q",
            retrieved_chunks=chunks,
            relevance_scores=[0.9],
        )
        assert m.context_relevance == 0.9

    def test_summary(self):
        ev = RAGEvaluator()
        chunks = [
            type("RC", (), {"chunk": type("C", (), {"text": "JEE coaching fee", "metadata": {"source": "jee"}})(), "score": 0.8})(),
        ]
        for i in range(3):
            ev.evaluate(
                query=f"query {i}",
                retrieved_chunks=chunks,
                generated_answer="JEE coaching fee is here",
            )
        s = ev.get_summary()
        assert s["total_queries"] == 3
        assert s["has_answer_rate"] == 100.0

    def test_clear(self):
        ev = RAGEvaluator()
        ev.evaluate("q", [])
        ev.clear()
        assert ev.results == []


class TestRAGTestCase:
    def test_create(self):
        case = RAGTestCase(
            query="test",
            ground_truth_sources=["src1"],
            expected_keywords=["foo"],
            language="en",
        )
        assert case.query == "test"
        assert case.ground_truth_sources == ["src1"]


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------

class TestAgentEvaluator:
    def test_empty_summary(self):
        ev = AgentEvaluator()
        assert ev.get_summary() == {}

    def test_successful_evaluation(self):
        ev = AgentEvaluator()
        m = ev.evaluate_conversation(
            conversation_id="c1",
            user_goal="Find JEE course fee",
            expected_tools=["search_courses", "get_fee_structure"],
            actual_tools=["search_courses", "get_fee_structure"],
            final_response="JEE course fee is Rs 150000",
            num_turns=1,
            latency_ms=1500.0,
        )
        assert m.task_completion == TaskCompletion.SUCCESS
        assert m.tool_call_accuracy == 1.0

    def test_partial_evaluation(self):
        ev = AgentEvaluator()
        m = ev.evaluate_conversation(
            conversation_id="c1",
            user_goal="Find JEE course fee",
            expected_tools=["search_courses", "get_fee_structure"],
            actual_tools=["search_courses"],  # Missing get_fee_structure
            final_response="JEE course is available",
            num_turns=2,
            latency_ms=2000.0,
        )
        assert m.task_completion in (TaskCompletion.PARTIAL, TaskCompletion.FAILED)
        assert m.tool_call_accuracy == 0.5

    def test_failed_evaluation(self):
        ev = AgentEvaluator()
        m = ev.evaluate_conversation(
            conversation_id="c1",
            user_goal="Find JEE course fee",
            expected_tools=["search_courses", "get_fee_structure"],
            actual_tools=[],  # No tools called
            final_response="I don't know",
            num_turns=1,
            latency_ms=1000.0,
        )
        assert m.task_completion == TaskCompletion.FAILED
        assert m.tool_call_accuracy == 0.0

    def test_no_expected_tools_perfect_score(self):
        ev = AgentEvaluator()
        m = ev.evaluate_conversation(
            conversation_id="c1",
            user_goal="Hello",
            expected_tools=[],
            actual_tools=[],
            final_response="Hello!",
            num_turns=1,
            latency_ms=100.0,
        )
        assert m.tool_call_accuracy == 1.0

    def test_record_conversation(self):
        ev = AgentEvaluator()
        ev.record_conversation({"id": "c1", "turns": []})
        assert len(ev.conversations) == 1

    def test_summary(self):
        ev = AgentEvaluator()
        for i in range(3):
            ev.evaluate_conversation(
                conversation_id=f"c{i}",
                user_goal="find JEE course",
                expected_tools=["search_courses"],
                actual_tools=["search_courses"],
                final_response="JEE course details",
                num_turns=1,
                latency_ms=1500.0,
            )
        s = ev.get_summary()
        assert s["total_conversations"] == 3
        assert s["task_completion_rate"] == 100.0
        assert s["avg_tool_accuracy"] == 100.0

    def test_per_language_summary(self):
        ev = AgentEvaluator()
        ev.evaluate_conversation("c1", "find JEE", ["search_courses"], ["search_courses"], "JEE found", 1, 1000, language="en")
        ev.evaluate_conversation("c2", "find NEET", ["search_courses"], ["search_courses"], "NEET found", 1, 1500, language="hi")
        per_lang = ev.get_per_language_summary()
        assert per_lang["en"]["count"] == 1
        assert per_lang["hi"]["count"] == 1

    def test_interruptions_tracked(self):
        ev = AgentEvaluator()
        m = ev.evaluate_conversation(
            conversation_id="c1",
            user_goal="q",
            expected_tools=[],
            actual_tools=[],
            final_response="r",
            num_turns=1,
            latency_ms=100.0,
            interruptions=2,
        )
        assert m.interruptions == 2

    def test_clear(self):
        ev = AgentEvaluator()
        ev.evaluate_conversation("c1", "q", [], [], "r", 1, 100.0)
        ev.clear()
        assert ev.results == []
        assert ev.conversations == []


# ---------------------------------------------------------------------------
# Run module — CLI smoke test
# ---------------------------------------------------------------------------

class TestNormalizeHelper:
    def test_lowercase(self):
        assert _normalize("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize("hello   world\n\t!") == "hello world !"

    def test_strip(self):
        assert _normalize("  hello  ") == "hello"
