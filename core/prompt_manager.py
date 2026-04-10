# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — Prompt Manager                          ║
# ║  Assembles the final system prompt each turn by injecting:      ║
# ║    - Vector memory recall                                       ║
# ║    - Session history summary                                    ║
# ║    - Available tools                                            ║
# ║    - Current datetime + active provider/model                   ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from LLM_Gateway.provider_router import Message
from Memory.session_memory import SessionMemory
from Memory.vector_memory import VectorMemory

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT_PATH = Path("system_prompt.txt")

_PLACEHOLDER_DEFAULTS = {
    "memory_context":  "(no relevant memory found)",
    "session_history": "(new session)",
    "available_tools": "(no tools enabled)",
    "current_datetime": "",
    "active_provider":  "unknown",
    "active_model":     "unknown",
}


# ── Prompt Manager ────────────────────────────────────────────────────────────

class PromptManager:
    """
    Owns the system prompt template and fills it fresh every turn.

    Usage:
        pm = PromptManager(session=session_mem, vector=vector_mem)
        messages = pm.build(
            user_input="explain async iterators",
            provider="anthropic",
            model="claude-opus-4-5",
            tool_names=["web_search", "calculator"],
        )
        # → list[Message] ready to send to ProviderRouter
    """

    def __init__(
        self,
        session: SessionMemory,
        vector: VectorMemory,
        system_prompt_path: Path | str = DEFAULT_SYSTEM_PROMPT_PATH,
    ):
        self._session = session
        self._vector = vector
        self._template = self._load_template(Path(system_prompt_path))

    # ── Template ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_template(path: Path) -> str:
        if not path.exists():
            logger.warning(f"system_prompt.txt not found at {path} — using bare minimum.")
            return (
                "You are Parmana, a technical assistant.\n\n"
                "{memory_context}\n{session_history}\n"
                "{available_tools}\n{current_datetime}\n"
                "{active_provider} / {active_model}"
            )
        return path.read_text(encoding="utf-8")

    def reload_template(self) -> None:
        """Hot-reload system_prompt.txt without restarting."""
        self._template = self._load_template(DEFAULT_SYSTEM_PROMPT_PATH)
        logger.debug("System prompt template reloaded.")

    # ── Builders ──────────────────────────────────────────────────────────────

    def _build_memory_context(self, query: str) -> str:
        if not query:
            return _PLACEHOLDER_DEFAULTS["memory_context"]
        recalled = self._vector.recall(query)
        return recalled if recalled else _PLACEHOLDER_DEFAULTS["memory_context"]

    def _build_session_history(self, last_n: int = 6) -> str:
        """
        Compact rendering of the last N turns for the system prompt.
        Full history is already in the messages list — this is a quick
        summary so the model has narrative context in the system block.
        """
        turns = self._session.get_turns(last_n=last_n)
        if not turns:
            return _PLACEHOLDER_DEFAULTS["session_history"]

        lines = []
        for t in turns:
            tag = t.role.upper()
            # Truncate very long turns to keep system prompt lean
            content = t.content if len(t.content) <= 300 else t.content[:300] + "…"
            lines.append(f"[{tag}] {content}")

        return "\n".join(lines)

    @staticmethod
    def _build_tools_block(tool_names: list[str]) -> str:
        if not tool_names:
            return _PLACEHOLDER_DEFAULTS["available_tools"]
        items = "\n".join(f"- {name}" for name in tool_names)
        return f"## Available Tools\n{items}"

    @staticmethod
    def _build_datetime() -> str:
        return time.strftime("## Date/Time\n%A, %Y-%m-%d %H:%M:%S %Z").strip()

    # ── Main Entry ────────────────────────────────────────────────────────────

    def build(
        self,
        user_input: str,
        provider: str = "unknown",
        model: str = "unknown",
        tool_names: Optional[list[str]] = None,
        extra_context: Optional[str] = None,
        recall_top_k: Optional[int] = None,
        recall_threshold: Optional[float] = None,
    ) -> list[Message]:
        """
        Build the full message list for one agent turn.

        Steps:
        1. Fill system prompt template with live context.
        2. Append optional extra_context (e.g. tool results, file contents).
        3. Pull rolling session messages.
        4. Append current user input.

        Returns list[Message] ready for ProviderRouter.chat().
        """

        # 1 — Fill placeholders
        memory_ctx = self._build_memory_context(user_input)
        if recall_top_k or recall_threshold:
            memory_ctx = self._vector.recall(
                user_input,
                top_k=recall_top_k,
                score_threshold=recall_threshold,
            ) or _PLACEHOLDER_DEFAULTS["memory_context"]

        session_summary = self._build_session_history()
        tools_block = self._build_tools_block(tool_names or [])
        dt_block = self._build_datetime()

        system_text = self._template.format(
            memory_context=memory_ctx,
            session_history=session_summary,
            available_tools=tools_block,
            current_datetime=dt_block,
            active_provider=provider,
            active_model=model,
        )

        # 2 — Append extra_context if provided
        if extra_context:
            system_text += f"\n\n## Additional Context\n{extra_context}"

        # 3 — Build message list
        messages: list[Message] = [Message(role="system", content=system_text)]

        # Pull session turns (excludes system turns, which we just built)
        for turn in self._session.get_turns():
            if turn.role in ("user", "assistant", "tool"):
                messages.append(turn.to_message())

        # 4 — Append current user message
        messages.append(Message(role="user", content=user_input))

        return messages

    # ── Post-turn ─────────────────────────────────────────────────────────────

    def commit(
        self,
        user_input: str,
        assistant_reply: str,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        store_in_vector: bool = True,
    ) -> None:
        """
        After a successful turn: write to session + optionally vector memory.
        Call this AFTER receiving the assistant response.
        """
        self._session.add_user(user_input)
        self._session.add_assistant(
            content=assistant_reply,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if store_in_vector:
            self._vector.store_turn("user", user_input)
            self._vector.store_turn("assistant", assistant_reply, provider=provider, model=model)

        logger.debug(
            f"Turn committed | provider={provider} model={model} "
            f"tokens in={input_tokens} out={output_tokens}"
        )

    def commit_tool_result(self, tool_name: str, result: str) -> None:
        """Store a tool result turn in session memory."""
        self._session.add_tool_result(tool_name=tool_name, content=result)
        self._vector.store_turn("tool", result, provider=tool_name)

    # ── Utils ─────────────────────────────────────────────────────────────────

    def set_system(self, content: str) -> None:
        """Override session system prompt directly (bypasses template)."""
        self._session.set_system(content)

    def clear_session(self, keep_system: bool = True) -> None:
        self._session.clear(keep_system=keep_system)
        logger.debug("Session cleared.")

    def status(self) -> dict:
        return {
            "session": self._session.summary_line(),
            "vector": self._vector.summary_line(),
        }

    def __repr__(self) -> str:
        return (
            f"<PromptManager "
            f"session_turns={self._session.turn_count} "
            f"vector_docs={self._vector.count}>"
        )
