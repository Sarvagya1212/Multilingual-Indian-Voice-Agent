"""Education-specific tools for the counselling agent.

Provides tools for course search, eligibility checks, fee structure,
and demo scheduling — all backed by a realistic mock dataset.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from src.agent.tools.base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Mock course catalogue
# ---------------------------------------------------------------------------

_COURSES = [
    {
        "id": "jee-2024",
        "name": "JEE Main + Advanced 2025",
        "category": "engineering",
        "duration": "2 years",
        "fee": 150000,
        "language": "Hindi + English",
        "description": "Complete JEE preparation with daily tests, mock exams, and faculty from IIT Delhi",
    },
    {
        "id": "neet-2024",
        "name": "NEET UG 2025",
        "category": "medical",
        "duration": "2 years",
        "fee": 180000,
        "language": "Hindi + English",
        "description": "Comprehensive NEET preparation with Biology focus, weekly tests, and expert medical faculty",
    },
    {
        "id": "cbse-12",
        "name": "CBSE Class 12 Board Prep 2025",
        "category": "boards",
        "duration": "1 year",
        "fee": 50000,
        "language": "Hindi + English",
        "description": "Board exam focused preparation with NCERT coverage and previous year papers",
    },
    {
        "id": "iit-foundation-9",
        "name": "IIT Foundation (Class 9-10)",
        "category": "foundation",
        "duration": "2 years",
        "fee": 60000,
        "language": "English",
        "description": "Build strong fundamentals for JEE from Class 9 onwards with Olympiad preparation",
    },
    {
        "id": "ai-ml-course",
        "name": "AI & ML Foundation",
        "category": "technology",
        "duration": "6 months",
        "fee": 45000,
        "language": "English",
        "description": "Introduction to Artificial Intelligence and Machine Learning with Python projects",
    },
]

_FEE_DETAILS = {
    "jee-2024": {
        "total": 150000,
        "installments": [
            {"amount": 75000, "due": "At admission"},
            {"amount": 75000, "due": "After 6 months"},
        ],
        "scholarship": "Up to 50% for meritorious students (based on Class 10/12 percentage)",
        "scholarship_criteria": "90%+ = 50%, 80-89% = 30%, 70-79% = 15%",
    },
    "neet-2024": {
        "total": 180000,
        "installments": [
            {"amount": 90000, "due": "At admission"},
            {"amount": 90000, "due": "After 6 months"},
        ],
        "scholarship": "Up to 40% for meritorious students",
        "scholarship_criteria": "90%+ = 40%, 80-89% = 25%, 70-79% = 10%",
    },
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SearchCoursesTool(Tool):
    """Search courses by keyword, category, or description.

    Use when the user asks about courses, coaching programs, or study options.
    """

    name = "search_courses"
    description = "Search for coaching courses by keyword or category (e.g., JEE, NEET, medical, engineering, boards)"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query — can be a course name, category, "
                    "or topic (e.g., 'JEE', 'NEET', 'physics', 'engineering')"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, limit: int = 5) -> ToolResult:
        query_lower = query.lower()
        results = [
            c for c in _COURSES
            if (
                query_lower in c["name"].lower()
                or query_lower in c["category"].lower()
                or query_lower in c["description"].lower()
            )
        ][:limit]

        if not results:
            # Fuzzy fallback — check any word
            query_words = query_lower.split()
            results = [
                c for c in _COURSES
                if any(
                    w in c["name"].lower() or w in c["description"].lower()
                    for w in query_words
                )
            ][:limit]

        return ToolResult(success=True, data={"courses": results, "count": len(results)})


class GetCourseDetailsTool(Tool):
    """Get detailed information about a specific course by ID.

    Use after the user expresses interest in a specific course.
    """

    name = "get_course_details"
    description = "Get detailed information about a specific coaching course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course ID (e.g., 'jee-2024', 'neet-2024')",
            },
        },
        "required": ["course_id"],
    }

    async def execute(self, course_id: str) -> ToolResult:
        course = next((c for c in _COURSES if c["id"] == course_id), None)
        if not course:
            return ToolResult(
                success=False,
                data=None,
                error=f"Course not found: {course_id}",
            )

        details = {
            **course,
            "syllabus": "Physics, Chemistry, Mathematics / Biology (as applicable)",
            "faculty": "IIT/NIT alumni and experienced educators",
            "study_material": "Provided (printed notes + online access)",
            "tests": "Weekly unit tests + Monthly mock exams",
            "batches": ["June (Regular)", "September (Weekend)", "January (Crash)"],
            "class_timing": "4-6 PM / 6-8 PM on weekdays",
            "demo_available": True,
        }
        return ToolResult(success=True, data=details)


class CheckEligibilityTool(Tool):
    """Check if a student is eligible for a specific course.

    Requires: course_id. Optionally: current_class, percentage.
    """

    name = "check_eligibility"
    description = "Check if a student meets eligibility criteria for a course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course ID to check eligibility for",
            },
            "current_class": {
                "type": "string",
                "description": "User's current class (e.g., '11', '12', '10')",
            },
            "percentage": {
                "type": "number",
                "description": "User's current/past percentage (0-100)",
            },
        },
        "required": ["course_id"],
    }

    async def execute(
        self,
        course_id: str,
        current_class: str | None = None,
        percentage: float | None = None,
    ) -> ToolResult:
        eligible = True
        reasons: List[str] = []

        # Class prerequisites
        if course_id == "jee-2024":
            if current_class and int(current_class) < 11:
                eligible = False
                reasons.append("JEE Main + Advanced course requires Class 11 or above")
            if percentage is not None and percentage < 70:
                reasons.append("Recommended: 70%+ in previous boards for best results")
        elif course_id == "neet-2024":
            if current_class and int(current_class) < 11:
                eligible = False
                reasons.append("NEET UG course requires Class 11 or above")
            if percentage is not None and percentage < 70:
                reasons.append("Recommended: 70%+ in PCB subjects for best results")
        elif course_id == "cbse-12":
            if current_class and int(current_class) < 11:
                eligible = False
                reasons.append("CBSE Board Prep requires Class 11 or above")
        elif course_id == "iit-foundation-9":
            if current_class and int(current_class) not in (9, 10):
                eligible = False
                reasons.append("IIT Foundation is designed for Class 9-10 students")
        elif course_id == "ai-ml-course":
            if percentage is not None and percentage < 60:
                reasons.append("Recommended: 60%+ for the AI & ML course")

        return ToolResult(
            success=True,
            data={"eligible": eligible, "reasons": reasons},
        )


class GetFeeStructureTool(Tool):
    """Get the fee breakdown for a specific course.

    Includes total fee, installment plans, and scholarship information.
    """

    name = "get_fee_structure"
    description = "Get fee structure, installment options, and scholarship details for a course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course ID",
            },
        },
        "required": ["course_id"],
    }

    async def execute(self, course_id: str) -> ToolResult:
        details = _FEE_DETAILS.get(course_id)
        if not details:
            # Fall back to the course fee if no detailed breakdown exists
            course = next((c for c in _COURSES if c["id"] == course_id), None)
            if not course:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Course not found: {course_id}",
                )
            details = {
                "total": course["fee"],
                "installments": [
                    {"amount": course["fee"] // 2, "due": "At admission"},
                    {"amount": course["fee"] - course["fee"] // 2, "due": "After 6 months"},
                ],
                "scholarship": "Contact the centre for scholarship details",
            }

        return ToolResult(success=True, data=details)


class ScheduleDemoTool(Tool):
    """Schedule a free demo class for an interested student.

    Validates that the course exists and returns a demo confirmation.
    """

    name = "schedule_demo"
    description = "Schedule a free demo class for a course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course ID for the demo",
            },
            "date": {
                "type": "string",
                "description": "Preferred demo date (YYYY-MM-DD format)",
            },
            "phone": {
                "type": "string",
                "description": "Student's phone number for confirmation",
            },
        },
        "required": ["course_id", "phone"],
    }

    async def execute(
        self,
        course_id: str,
        phone: str,
        date: str | None = None,
    ) -> ToolResult:
        # Validate course exists
        course = next((c for c in _COURSES if c["id"] == course_id), None)
        if not course:
            return ToolResult(
                success=False,
                data=None,
                error=f"Course not found: {course_id}",
            )

        demo_id = f"DEMO-{random.randint(1000, 9999)}"
        return ToolResult(
            success=True,
            data={
                "demo_id": demo_id,
                "course_name": course["name"],
                "status": "scheduled",
                "confirm_date": date or "Within 24 hours (you will receive a call)",
                "contact_phone": phone,
            },
        )
