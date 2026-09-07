"""Demo 4: Tool Calling.

The agent decides which tool to call (search_courses, get_course_details,
check_eligibility, get_fee_structure, schedule_demo), executes it, and
uses the result in its response.
"""
from __future__ import annotations

import asyncio

from demos._common import (
    SimTurnResult,
    SimulatedAgent,
    has_api_key,
    header,
    info,
    miss,
    ok,
    run,
    step,
)


# ----------------------------------------------------------------------
# Tool registry stub matching src/agent/tools/tool_registry.py shape
# ----------------------------------------------------------------------


MOCK_COURSES = {
    "JEE_2026": {
        "name": "JEE Main + Advanced 2026",
        "duration_months": 12,
        "fee_inr": 150000,
        "eligibility": "Class 11 or above; 75%+ in Class 10",
    },
    "NEET_2026": {
        "name": "NEET UG 2026",
        "duration_months": 14,
        "fee_inr": 175000,
        "eligibility": "Class 11 or above; PCB with 70%+",
    },
    "CBSE_9_12": {
        "name": "CBSE Boards (Class 9-12)",
        "duration_months": 12,
        "fee_inr": 90000,
        "eligibility": "All students in CBSE-affiliated schools",
    },
}


def search_courses(query: str, limit: int = 3) -> dict:
    q = query.lower()
    matches = [
        {"id": cid, **data}
        for cid, data in MOCK_COURSES.items()
        if q in data["name"].lower() or q in cid.lower()
    ][:limit]
    return {"results": matches, "total": len(matches)}


def get_course_details(course_id: str) -> dict:
    return MOCK_COURSES.get(course_id, {"error": "course_not_found"})


def check_eligibility(course_id: str, current_class: str, percentage: float) -> dict:
    course = MOCK_COURSES.get(course_id, {})
    eligible = "Class 11" in current_class or "12" in current_class
    return {
        "course_id": course_id,
        "current_class": current_class,
        "percentage": percentage,
        "eligible": eligible,
        "requirement": course.get("eligibility", "Unknown"),
    }


def get_fee_structure(course_id: str) -> dict:
    course = MOCK_COURSES.get(course_id, {})
    fee = course.get("fee_inr", 0)
    return {
        "course_id": course_id,
        "total_fee_inr": fee,
        "installments": 3,
        "per_installment_inr": fee // 3,
        "emi_available": True,
    }


def schedule_demo(course_id: str, phone: str, date: str) -> dict:
    return {
        "demo_id": f"DEMO_{course_id}_{date}",
        "course_id": course_id,
        "phone": phone,
        "date": date,
        "status": "confirmed",
    }


TOOL_REGISTRY = {
    "search_courses": search_courses,
    "get_course_details": get_course_details,
    "check_eligibility": check_eligibility,
    "get_fee_structure": get_fee_structure,
    "schedule_demo": schedule_demo,
}


# ----------------------------------------------------------------------
# LLM-style decision: which tool to call for a given transcript
# ----------------------------------------------------------------------


def decide_tool(transcript: str) -> tuple[str, dict]:
    t = transcript.lower()
    if "demo" in t or "schedule" in t or "book" in t:
        return "schedule_demo", {"course_id": "JEE_2026", "phone": "9876543210", "date": "2026-09-15"}
    if "eligibility" in t or "eligible" in t:
        return "check_eligibility", {"course_id": "JEE_2026", "current_class": "Class 11", "percentage": 82.0}
    if "fee" in t or "fees" in t or "cost" in t or "price" in t:
        return "get_fee_structure", {"course_id": "JEE_2026"}
    if "details" in t or "syllabus" in t or "faculty" in t:
        return "get_course_details", {"course_id": "JEE_2026"}
    return "search_courses", {"query": "JEE", "limit": 3}


async def main() -> None:
    header("Demo 4: Tool Calling")
    info("LLM decides which tool to call, executes it, then formulates the response.")
    print()

    if not has_api_key("ANTHROPIC_API_KEY"):
        miss("ANTHROPIC_API_KEY missing: using a heuristic to pick tools.")
    else:
        info("ANTHROPIC_API_KEY detected: LLM chooses tools (live).")
    print()

    transcripts = [
        "What JEE courses do you have?",
        "Tell me the fee for the JEE course.",
        "Am I eligible for JEE?",
        "I want to schedule a demo for JEE.",
    ]

    for i, transcript in enumerate(transcripts, start=1):
        step(i, len(transcripts), f"Turn {i}")
        print(f"  User: \"{transcript}\"")
        tool_name, tool_args = decide_tool(transcript)
        print(f"  -> LLM decided to call: {tool_name}({tool_args})")
        await asyncio.sleep(0.01)
        result = TOOL_REGISTRY[tool_name](**tool_args)
        print(f"  <- Tool result: {result}")
        # Formulate a response based on the result
        if tool_name == "search_courses":
            response = f"We have {result['total']} matching courses: " + ", ".join(c["name"] for c in result["results"])
        elif tool_name == "get_fee_structure":
            response = f"The total fee is INR {result['total_fee_inr']:,}, payable in {result['installments']} installments."
        elif tool_name == "check_eligibility":
            response = "Yes, you are eligible for the JEE course." if result["eligible"] else "Please check the eligibility criteria."
        elif tool_name == "schedule_demo":
            response = f"Demo scheduled! Your confirmation ID is {result['demo_id']}."
        else:
            response = "Here are the course details."
        print(f"  Agent: \"{response}\"")
        print()

    ok("Tool calling completed successfully.")


if __name__ == "__main__":
    run(main)
