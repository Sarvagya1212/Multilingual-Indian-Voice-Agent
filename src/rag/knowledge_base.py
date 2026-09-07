"""Knowledge base — document store and retrieval interface for RAG."""
from __future__ import annotations

from typing import Any

from src.logger import setup_logger
from src.rag.config import RAGConfig
from src.rag.retriever import Retriever, RetrievedChunk

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Document corpus
# ---------------------------------------------------------------------------

COURSE_DOCUMENTS: list[dict[str, Any]] = [
    {
        "source": "jee_course",
        "content": """
        JEE Main + Advanced Course 2024

        Overview: Our comprehensive JEE preparation program covers Physics,
        Chemistry, and Mathematics for students aspiring to crack IIT JEE Main
        and Advanced examinations.

        Eligibility: Students currently in Class 10 or 11 can enroll. Class 12
        students can also join for the one-year intensive dropper batch.

        Duration: 2 years for foundation batch (Class 11), 1 year for
        dropper batch.

        Fee Structure: Total fee is Rs 1,50,000 per year. Installment facility
        available - Rs 75,000 at admission, Rs 75,000 after 6 months.
        Scholarship up to 50% for meritorious students based on Class 10 or 11
        board exam percentage.

        Faculty: All teachers are IIT Delhi Alumni with 10+ years of experience.

        Study Material: Comprehensive study material including theory books,
        problem sets, and previous year JEE questions provided.

        Test Schedule: Weekly tests every Sunday, monthly grand tests, and
        full-length JEE pattern mock tests every quarter.

        Batches: New batches start in June, September, and January.

        Contact: Call 9876543210 for admissions.
        """,
        "metadata": {"category": "engineering", "exam": "JEE"},
    },
    {
        "source": "neet_course",
        "content": """
        NEET UG 2024 Preparation Course

        Overview: Complete NEET preparation covering Physics, Chemistry, and
        Biology for medical college aspirants.

        Eligibility: Class 11 and 12 students. Also open for repeaters.

        Duration: 2 years for Class 11 students, 1 year intensive for
        Class 12 and repeaters.

        Fee Structure: Total fee Rs 1,80,000 per year. Can be paid in two
        installments of Rs 90,000 each.

        Syllabus: Complete NCERT Biology (Class 11 and 12), Physics and
        Chemistry covering both Class 11 and 12 syllabus.

        Faculty: Experienced doctors and medical college professors with
        proven track record.

        Test Series: Regular objective tests, NEET pattern mock tests every
        month, and chapter-wise practice questions.

        Seats: Limited seats per batch. Early registration recommended.
        """,
        "metadata": {"category": "medical", "exam": "NEET"},
    },
    {
        "source": "cbse_boards",
        "content": """
        CBSE Class 11-12 Board Preparation Course

        Overview: Focused preparation for CBSE Class 11 and 12 board
        examinations with integrated competitive exam coaching.

        Subjects: Physics, Chemistry, Mathematics/Biology, English.

        Fee Structure: Rs 50,000 per year. Includes all study material
        and test series.

        Faculty: CBSE experts with deep understanding of board exam patterns.

        Classes: 5 days a week, 3 hours per day.

        Results: 95% of our students score above 90% in boards.
        """,
        "metadata": {"category": "boards", "exam": "CBSE"},
    },
    {
        "source": "faq_admissions",
        "content": """
        Frequently Asked Questions - Admissions

        Q: How do I take admission?
        A: Visit our center with your previous marksheets. We conduct a short
        counseling session and aptitude test. Based on the results, we will
        suggest the best course for you.

        Q: Can I get a demo class?
        A: Yes! We offer one free demo class. Call us at 9876543210 or fill
        the inquiry form on our website to schedule.

        Q: What if I miss a class?
        A: Recorded lectures are available online. You can watch them anytime
        and clear doubts in the next scheduled doubt session.

        Q: Is hostel facility available?
        A: Yes, we have tie-ups with nearby hostels. Separate facilities
        for boys and girls with mess service.

        Q: What payment methods do you accept?
        A: Cash, UPI, Bank Transfer, and Credit/Debit Cards. EMI facility
        also available through our banking partners.

        Q: Is scholarship available?
        A: Yes, meritorious students can get up to 50% scholarship based on
        their previous board exam percentage.

        Q: What are the batch sizes?
        A: Each batch has a maximum of 40 students to ensure personalized
        attention.
        """,
        "metadata": {"category": "faq"},
    },
    {
        "source": "faq_general",
        "content": """
        Frequently Asked Questions - General

        Q: Do you offer online classes?
        A: Yes, we offer hybrid mode. Students can attend physically or
        join live online. Recorded lectures available for revision.

        Q: Can I change batch timing?
        A: Batch change requests are accepted within the first week, subject
        to availability.

        Q: How are doubts cleared?
        A: Dedicated doubt-clearing sessions every Saturday. One-on-one doubt
        sessions available on request.

        Q: What support is provided for competitive exams?
        A: Integrated coaching for JEE, NEET, and other competitive exams
        included in the course fee.

        Q: Are there any scholarships for economically weaker students?
        A: Yes, we have a welfare fund. Contact our counseling team for
        details on eligibility.
        """,
        "metadata": {"category": "faq"},
    },
]


# ---------------------------------------------------------------------------
# KnowledgeBase wrapper
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """High-level interface over the RAG retriever.

    Provides a simple API:
    - index()       → load and embed all documents
    - query()       → retrieve chunks for a question
    - query_rag()   → retrieve + build context string for LLM prompt injection
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        config: RAGConfig | None = None,
    ):
        self.retriever = retriever or Retriever(config=config)
        self._built = False

    async def build(self, documents: list[dict[str, Any]] | None = None) -> None:
        """Index documents into the knowledge base.

        Args:
            documents: List of documents. Defaults to COURSE_DOCUMENTS.
        """
        docs = documents if documents is not None else COURSE_DOCUMENTS
        await self.retriever.index_documents(docs)
        self._built = True
        logger.info(f"[KB] Knowledge base built with {len(docs)} documents")

    async def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a question."""
        if not self._built:
            await self.build()
        return await self.retriever.retrieve(question, filters=filters, top_k=top_k)

    async def query_rag(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> str:
        """Retrieve chunks and build a context string for LLM injection.

        Returns a formatted context string that can be injected into the
        system prompt: "Context from knowledge base:\n{context}\n\nQuestion: {q}"
        """
        chunks = await self.query(question, filters=filters, top_k=top_k)
        context = self.retriever.build_context(chunks)
        return context

    @property
    def is_built(self) -> bool:
        return self._built


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Return the singleton KnowledgeBase instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
