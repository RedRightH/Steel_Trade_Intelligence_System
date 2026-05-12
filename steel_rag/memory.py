"""
memory.py - Conversation memory for the India Steel Trade Intelligence Platform.

ConversationMemory stores the last N question/answer turns and injects them
as context into the classifier and each agent so follow-up questions work
naturally without the user having to repeat themselves.

Usage:
    from memory import ConversationMemory
    mem = ConversationMemory()

    # after each route_query call:
    mem.add(ro.question, ro.result.answer_text, ro.question_type, ro.agent_used)

    # pass to the next call:
    ro = route_query("and what about Vietnam?", memory=mem)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    question:      str
    answer:        str
    question_type: str
    agent_used:    str


class ConversationMemory:
    """
    Sliding window of the last `max_turns` Q&A turns.

    Provides formatted context strings for injection into prompts.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, question: str, answer: str,
            question_type: str = "", agent_used: str = "") -> None:
        self._turns.append(Turn(
            question      = question,
            answer        = answer[:600],   # cap to avoid huge prompts
            question_type = question_type,
            agent_used    = agent_used,
        ))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def clear(self) -> None:
        self._turns.clear()

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_type(self) -> Optional[str]:
        return self._turns[-1].question_type if self._turns else None

    @property
    def last_answer(self) -> Optional[str]:
        return self._turns[-1].answer if self._turns else None

    # ── Context formatters ────────────────────────────────────────────────────

    def classifier_context(self, n: int = 3) -> str:
        """
        Short context for the classifier — enough to resolve pronoun references
        and topic continuity (e.g. 'what about China?' after an AD question).
        """
        recent = self._turns[-n:]
        if not recent:
            return ""
        lines = ["[Conversation so far]"]
        for i, t in enumerate(recent, 1):
            lines.append(f"Turn {i} ({t.question_type}): {t.question}")
        return "\n".join(lines)

    def agent_context(self, n: int = 3) -> str:
        """
        Richer context for agent prompts — includes truncated answers so the
        LLM can resolve references like 'those countries' or 'the rate mentioned'.
        """
        recent = self._turns[-n:]
        if not recent:
            return ""
        lines = ["[Previous conversation — use this to resolve references]"]
        for i, t in enumerate(recent, 1):
            lines.append(f"Q{i}: {t.question}")
            lines.append(f"A{i}: {t.answer[:300]}")
            lines.append("")
        return "\n".join(lines)

    def is_followup(self, question: str) -> bool:
        """
        Heuristic: question likely refers to prior context if it starts with
        a pronoun/article reference and memory is non-empty.
        """
        if self.is_empty:
            return False
        q = question.strip().lower()
        followup_starters = (
            "what about", "and ", "how about", "which of",
            "those ", "that ", "these ", "them", "it ", "its ",
            "same ", "similar ", "compare that", "tell me more",
            "elaborate", "why ", "when ", "how much", "how many",
        )
        return any(q.startswith(s) for s in followup_starters)

    # ── Serialisation (for Streamlit session_state) ───────────────────────────

    def to_list(self) -> list[dict]:
        return [
            {"question": t.question, "answer": t.answer,
             "question_type": t.question_type, "agent_used": t.agent_used}
            for t in self._turns
        ]

    @classmethod
    def from_list(cls, data: list[dict], max_turns: int = 5) -> "ConversationMemory":
        mem = cls(max_turns=max_turns)
        for d in data:
            mem.add(d["question"], d["answer"],
                    d.get("question_type",""), d.get("agent_used",""))
        return mem

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={self.turn_count}/{self.max_turns})"
