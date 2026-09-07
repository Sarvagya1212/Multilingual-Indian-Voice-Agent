"""Master evaluation runner — aggregates STT, TTS, RAG, and agent evaluations.

Usage:
    python -m evaluations.run
    python -m evaluations.run --component rag
    python -m evaluations.run --output reports/my_run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import setup_logger
from evaluations.stt.metrics import STTEvaluator
from evaluations.stt.evaluator import STTTestCase
from evaluations.tts.metrics import TTSEvaluator
from evaluations.tts.evaluator import TTSTestCase
from evaluations.rag.metrics import RAGEvaluator
from evaluations.rag.evaluator import RAGTestCase
from evaluations.agent.metrics import AgentEvaluator

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Test datasets (placeholders — replace with real annotated data)
# ---------------------------------------------------------------------------

_STT_TEST_CASES = [
    # English
    STTTestCase(audio=b"", reference="Hello, I want to join JEE classes", language="en", audio_duration=3.0),
    STTTestCase(audio=b"", reference="What is the fee for NEET coaching", language="en", audio_duration=2.5),
    STTTestCase(audio=b"", reference="Can I get a demo class tomorrow", language="en", audio_duration=2.0),
    # Hindi
    STTTestCase(audio=b"", reference="mujhe JEE ki taiyari karna hai", language="hi", audio_duration=3.0),
    STTTestCase(audio=b"", reference="NEET ka course kitna mahal hai", language="hi", audio_duration=2.5),
    # Hinglish
    STTTestCase(audio=b"", reference="JEE ke liye preparation kaise karein", language="hinglish", audio_duration=2.5),
    STTTestCase(audio=b"", reference="demo class lena hai, phone number do", language="hinglish", audio_duration=2.0),
]

_TTS_TEST_CASES = [
    TTSTestCase(text="Hello, I want to join JEE classes. What are the fees?"),
    TTSTestCase(text="नमस्ते, मुझे NEET कोचिंग चाहिए। क्या फीस है?"),
    TTSTestCase(text="JEE ke liye preparation course hai. Total fee Rs 1,50,000.", language="hinglish"),
    TTSTestCase(text="We offer scholarships up to 50% for meritorious students."),
    TTSTestCase(text="New batches start in June, September, and January."),
    TTSTestCase(text="Call 9876543210 to schedule your demo class today."),
]

_RAG_TEST_CASES = [
    RAGTestCase(
        query="What is the fee for JEE coaching?",
        ground_truth_sources=["jee_course"],
        expected_keywords=["150000", "150000", "Rs", "installment", "scholarship"],
    ),
    RAGTestCase(
        query="Do you offer NEET preparation?",
        ground_truth_sources=["neet_course"],
        expected_keywords=["NEET", "Biology", "Physics", "Chemistry", "NCERT"],
    ),
    RAGTestCase(
        query="How do I take admission?",
        ground_truth_sources=["faq_admissions"],
        expected_keywords=["admission", "marksheets", "counseling"],
    ),
    RAGTestCase(
        query="Is hostel facility available?",
        ground_truth_sources=["faq_admissions"],
        expected_keywords=["hostel", "boys", "girls", "mess"],
    ),
    RAGTestCase(
        query="What scholarships are available?",
        ground_truth_sources=["jee_course", "faq_admissions"],
        expected_keywords=["scholarship", "50%", "meritorious"],
    ),
    RAGTestCase(
        query="What is the CBSE board exam fee?",
        ground_truth_sources=["cbse_boards"],
        expected_keywords=["CBSE", "50000", "board"],
    ),
]

_AGENT_TEST_CASES = [
    {
        "id": "agent_001",
        "user_goal": "Find JEE Main coaching fee",
        "expected_tools": ["search_courses", "get_fee_structure"],
        "language": "en",
    },
    {
        "id": "agent_002",
        "user_goal": "Schedule a demo class for NEET",
        "expected_tools": ["search_courses", "schedule_demo"],
        "language": "hinglish",
    },
    {
        "id": "agent_003",
        "user_goal": "Am I eligible for JEE if I'm in Class 10?",
        "expected_tools": ["check_eligibility"],
        "language": "en",
    },
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

async def run_stt_evaluation(stt_provider: Any | None = None) -> dict:
    """Run STT evaluation (placeholder — needs real provider + audio)."""
    from evaluations.stt.evaluator import STTEvaluatorRunner

    if stt_provider is None:
        logger.info("[STT Eval] No STT provider — using simulated metrics")
        # Simulate: inject known WER/CER values
        ev = STTEvaluator()
        import random
        random.seed(42)
        for case in _STT_TEST_CASES:
            wer = random.uniform(0.05, 0.25)  # Simulated WER
            ev.evaluate(
                reference=case.reference,
                hypothesis=case.reference,  # Perfect transcription (simulation)
                language=case.language,
                detected_language=case.language,
                latency_ms=random.uniform(200, 600),
                audio_duration=case.audio_duration,
            )
        return {"summary": ev.get_summary(), "per_language": ev.get_per_language_summary()}

    runner = STTEvaluatorRunner(stt_provider)
    return await runner.run(_STT_TEST_CASES)


async def run_tts_evaluation(tts_provider: Any | None = None) -> dict:
    """Run TTS evaluation (placeholder — needs real provider)."""
    from evaluations.tts.evaluator import TTSEvaluatorRunner

    if tts_provider is None:
        logger.info("[TTS Eval] No TTS provider — using simulated metrics")
        ev = TTSEvaluator()
        import random
        random.seed(42)
        for case in _TTS_TEST_CASES:
            ev.evaluate(
                text=case.text,
                audio_bytes=b"SIMULATED_AUDIO" * 100,
                audio_duration=len(case.text) / 150 * 60,
                latency_ms=random.uniform(300, 800),
            )
        return {"summary": ev.get_summary()}

    runner = TTSEvaluatorRunner(tts_provider)
    return await runner.run(_TTS_TEST_CASES)


async def run_rag_evaluation(retriever: Any | None = None) -> dict:
    """Run RAG evaluation against the knowledge base."""
    from evaluations.rag.evaluator import RAGEvaluatorRunner
    from src.rag import get_knowledge_base

    if retriever is None:
        logger.info("[RAG Eval] Using default knowledge base retriever")
        kb = get_knowledge_base()
        if not kb.is_built:
            await kb.build()
        retriever = kb.retriever

    runner = RAGEvaluatorRunner(retriever)
    return await runner.run(_RAG_TEST_CASES)


async def run_agent_evaluation(orchestrator: Any | None = None) -> dict:
    """Run agent evaluation (placeholder — needs real orchestrator)."""
    if orchestrator is None:
        logger.info("[Agent Eval] No orchestrator — using simulated metrics")
        ev = AgentEvaluator()
        import random
        random.seed(42)
        for case in _AGENT_TEST_CASES:
            ev.evaluate_conversation(
                conversation_id=case["id"],
                user_goal=case["user_goal"],
                expected_tools=case["expected_tools"],
                actual_tools=case["expected_tools"][:1],  # Simulate partial tool use
                final_response=f"Here's information about {case['user_goal']}.",
                num_turns=1,
                latency_ms=random.uniform(800, 2000),
                language=case["language"],
            )
        return {"summary": ev.get_summary(), "per_language": ev.get_per_language_summary()}

    # Real orchestrator evaluation would go here
    raise NotImplementedError("Live orchestrator evaluation not yet implemented")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> dict:
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument(
        "--component",
        choices=["all", "stt", "tts", "rag", "agent"],
        default="all",
        help="Component to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EVALUATION SUITE — Multilingual Indian Voice Agent")
    print("=" * 60)
    print(f"  Component: {args.component.upper()}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 60 + "\n")

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "component": args.component,
    }

    if args.component in ("all", "stt"):
        logger.info("Running STT evaluation...")
        results["stt"] = await run_stt_evaluation()

    if args.component in ("all", "tts"):
        logger.info("Running TTS evaluation...")
        results["tts"] = await run_tts_evaluation()

    if args.component in ("all", "rag"):
        logger.info("Running RAG evaluation...")
        results["rag"] = await run_rag_evaluation()

    if args.component in ("all", "agent"):
        logger.info("Running agent evaluation...")
        results["agent"] = await run_agent_evaluation()

    # Save report
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Report saved to {out_path}")

    # Print summary
    _print_summary(results)

    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    if "stt" in results and results["stt"].get("summary"):
        s = results["stt"]["summary"]
        print("\n  [STT] Performance")
        print(f"     Samples:         {s.get('total_samples', 0)}")
        print(f"     Avg WER:        {s.get('avg_wer', 0):.2%}")
        print(f"     Avg CER:        {s.get('avg_cer', 0):.2%}")
        print(f"     Avg Accuracy:   {s.get('avg_accuracy', 0):.1f}%")
        print(f"     Language Acc:   {s.get('language_accuracy_pct', 0):.1f}%")
        print(f"     Avg Latency:    {s.get('avg_latency_ms', 0):.0f}ms")
        print(f"     p90 Latency:    {s.get('p90_latency_ms', 0):.0f}ms")

    if "tts" in results and results["tts"].get("summary"):
        s = results["tts"]["summary"]
        print("\n  [TTS] Performance")
        print(f"     Samples:            {s.get('total_samples', 0)}")
        print(f"     Avg Latency:       {s.get('avg_latency_ms', 0):.0f}ms")
        print(f"     Avg RTF:           {s.get('avg_real_time_factor', 0):.2f}")
        print(f"     Avg Bytes/sec:     {s.get('avg_bytes_per_second', 0):.0f}")
        print(f"     Pronunciation Fix: {s.get('total_pronunciation_corrections', 0)}")

    if "rag" in results and results["rag"].get("summary"):
        s = results["rag"]["summary"]
        print("\n  [RAG] Performance")
        print(f"     Queries:           {s.get('total_queries', 0)}")
        print(f"     Avg Recall:       {s.get('avg_retrieval_recall', 0):.2%}")
        print(f"     Avg Relevance:    {s.get('avg_context_relevance', 0):.3f}")
        print(f"     Avg Groundedness: {s.get('avg_answer_groundedness', 0):.2%}")
        print(f"     Avg Hallucination:{s.get('avg_hallucination', 0):.2%}")
        print(f"     Avg Latency:      {s.get('avg_latency_ms', 0):.0f}ms")

    if "agent" in results and results["agent"].get("summary"):
        s = results["agent"]["summary"]
        print("\n  [Agent] Performance")
        print(f"     Conversations:     {s.get('total_conversations', 0)}")
        print(f"     Task Completion:  {s.get('task_completion_rate', 0):.1f}%")
        print(f"     Tool Accuracy:    {s.get('avg_tool_accuracy', 0):.1f}%")
        print(f"     Relevance:        {s.get('avg_response_relevance', 0):.1f}%")
        print(f"     Avg Turns:        {s.get('avg_turns', 0):.1f}")
        print(f"     Avg Latency:      {s.get('avg_latency_ms', 0):.0f}ms")
        brkdwn = s.get("completion_breakdown", {})
        print(f"     Success: {brkdwn.get('success', 0)}, Partial: {brkdwn.get('partial', 0)}, Failed: {brkdwn.get('failed', 0)}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
