# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — Agent Brain Loop                         ║
# ║  Orchestrates: prompt → LLM → tool calls → response              ║
# ║  Supports streaming, multi-turn tool use, vision routing.        ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Union

import yaml

from Core.prompt_manager import PromptManager
from LLM_Gateway.provider_router import Message, ProviderResponse, ProviderRouter
from Memory.session_memory import SessionMemory
from Memory.vector_memory import VectorMemory
from Skills.registry import SkillRegistry, registry as default_registry
from Vision.vision_handler import VisionHandler

logger = logging.getLogger(__name__)


# ── Turn Result ───────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    reply: str
    provider: str
    model: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    was_streamed: bool = False


# ── Tool Call Parser ──────────────────────────────────────────────────────────
# Models emit tool calls in different formats. We support:
#   1. JSON code block:  ```json\n{"tool": "...", "args": {...}}\n```
#   2. XML-style:        <tool name="web_search"><args>{"query":"..."}</args></tool>
#   3. Plain JSON:       {"tool": "...", "args": {...}}

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\"tool\".*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
_XML_TOOL_RE = re.compile(
    r'<tool\s+name=["\'](\w+)["\']>\s*<args>(.*?)</args>\s*</tool>',
    re.DOTALL,
)
_PLAIN_JSON_RE = re.compile(
    r'\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*\}',
    re.DOTALL,
)


def _parse_tool_calls(text: str) -> list[dict]:
    """
    Extract tool call dicts from model output.
    Returns list of {"tool": str, "args": dict}.
    """
    calls = []

    # 1. JSON code blocks
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            if "tool" in obj:
                calls.append({"tool": obj["tool"], "args": obj.get("args", obj.get("arguments", {}))})
        except json.JSONDecodeError:
            pass

    # 2. XML-style
    for match in _XML_TOOL_RE.finditer(text):
        tool_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {"raw": match.group(2)}
        calls.append({"tool": tool_name, "args": args})

    # 3. Plain JSON (only if nothing found above)
    if not calls:
        for match in _PLAIN_JSON_RE.finditer(text):
            try:
                obj = json.loads(match.group(0))
                if "tool" in obj:
                    calls.append({"tool": obj["tool"], "args": obj.get("args", {})})
            except json.JSONDecodeError:
                pass

    # Deduplicate by (tool, args) identity
    seen = set()
    unique = []
    for c in calls:
        key = (c["tool"], json.dumps(c["args"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def _strip_tool_calls(text: str) -> str:
    """Remove tool call markup from final reply."""
    text = _JSON_BLOCK_RE.sub("", text)
    text = _XML_TOOL_RE.sub("", text)
    text = re.sub(r'\{[^{}]*"tool"\s*:\s*"\w+"[^{}]*\}', "", text)
    return text.strip()


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Parmana's reasoning loop.

    One turn:
        1. build_messages()      — inject memory + session into prompt
        2. llm.chat()            — get response (streaming or not)
        3. parse_tool_calls()    — detect tool invocations in response
        4. execute_tools()       — run each tool via SkillRegistry
        5. inject tool results   — append results, call LLM again
        6. commit()              — write turn to session + vector memory

    Repeat steps 3-5 up to `max_tool_rounds` times per user turn.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        registry: Optional[SkillRegistry] = None,
    ):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        app_cfg = self._cfg.get("app", {})
        self._default_provider: str = app_cfg.get("default_provider", "openai")
        cli_cfg = self._cfg.get("cli", {})
        self._stream: bool = cli_cfg.get("stream", True)
        self._show_provider: bool = cli_cfg.get("show_provider", True)
        self._show_tokens: bool = cli_cfg.get("show_tokens", False)

        self._max_tool_rounds: int = 5       # prevent infinite tool loops
        self._max_response_tokens: int = 4096

        # Core components
        self._router   = ProviderRouter(config_path)
        self._session  = SessionMemory(
            max_messages=self._cfg.get("memory", {}).get("session", {}).get("max_messages", 50),
            include_system=True,
        )
        self._vector   = VectorMemory(config_path)
        self._prompt   = PromptManager(
            session=self._session,
            vector=self._vector,
        )
        self._vision   = VisionHandler()
        self._registry = registry or default_registry

        logger.info(
            f"Agent initialized | provider={self._default_provider} "
            f"skills={self._registry.list_names()}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: Optional[bool] = None,
        image: Optional[Union[str, bytes]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> TurnResult:
        """
        Process one user turn end-to-end.

        Args:
            user_input:  The user's message.
            provider:    Override LLM provider.
            model:       Override model.
            task:        Task hint for routing (code/reasoning/fast/etc).
            stream:      Override stream setting.
            image:       Optional image (path/URL/bytes) — triggers vision routing.
            on_token:    Callback for streaming tokens (CLI uses this for live output).

        Returns:
            TurnResult with reply, provider, model, tool info, token counts.
        """
        t0 = time.monotonic()
        should_stream = stream if stream is not None else self._stream

        # Handle image input
        if image is not None:
            return await self._vision_turn(
                user_input=user_input,
                image=image,
                provider=provider,
                model=model,
                t0=t0,
            )

        # Resolve provider + model
        active_provider = provider or self._default_provider
        active_model = model or self._router.get(active_provider).default_model

        tool_names = self._registry.list_names()

        # Build initial messages
        messages = self._prompt.build(
            user_input=user_input,
            provider=active_provider,
            model=active_model,
            tool_names=tool_names,
        )

        all_tool_calls: list[dict] = []
        all_tool_results: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        final_reply = ""

        # ── Multi-turn tool loop ───────────────────────────────────────────────
        for round_num in range(self._max_tool_rounds + 1):

            if should_stream and on_token and round_num == 0:
                # Stream first response, collect full text
                reply_text, in_tok, out_tok = await self._stream_response(
                    messages=messages,
                    provider=active_provider,
                    model=active_model,
                    task=task,
                    on_token=on_token,
                )
            else:
                resp: ProviderResponse = await self._router.chat(
                    messages=messages,
                    provider=active_provider,
                    model=active_model,
                    task=task,
                    stream=False,
                )
                reply_text = resp.text
                in_tok     = resp.input_tokens
                out_tok    = resp.output_tokens

            input_tokens  += in_tok
            output_tokens += out_tok

            # Check for tool calls
            tool_calls = _parse_tool_calls(reply_text)

            if not tool_calls or round_num == self._max_tool_rounds:
                # No tools or max rounds hit — this is the final reply
                final_reply = _strip_tool_calls(reply_text)
                break

            # Execute tools
            tool_results_text_parts = []
            for call in tool_calls:
                all_tool_calls.append(call)
                result = await self._registry.call(call["tool"], call["args"])
                all_tool_results.append({
                    "tool": call["tool"],
                    "args": call["args"],
                    "result": result.as_text(),
                    "success": result.success,
                    "latency_ms": result.latency_ms,
                })
                tool_results_text_parts.append(
                    f"[Tool: {call['tool']}]\n{result.as_text()}"
                )
                self._prompt.commit_tool_result(call["tool"], result.as_text())
                logger.debug(
                    f"Tool '{call['tool']}' → success={result.success} "
                    f"latency={result.latency_ms:.0f}ms"
                )

            # Inject tool results back into messages for next round
            combined_results = "\n\n".join(tool_results_text_parts)
            messages.append(Message(role="assistant", content=reply_text))
            messages.append(Message(role="user", content=f"Tool results:\n{combined_results}\n\nNow give your final answer."))

        # ── Commit turn ────────────────────────────────────────────────────────
        self._prompt.commit(
            user_input=user_input,
            assistant_reply=final_reply,
            provider=active_provider,
            model=active_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            store_in_vector=True,
        )

        return TurnResult(
            reply=final_reply,
            provider=active_provider,
            model=active_model,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.monotonic() - t0) * 1000,
            was_streamed=should_stream,
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def _stream_response(
        self,
        messages: list[Message],
        provider: str,
        model: str,
        task: Optional[str],
        on_token: Callable[[str], None],
    ) -> tuple[str, int, int]:
        """
        Stream tokens via on_token callback. Returns (full_text, in_tokens, out_tokens).
        Falls back to non-streaming if provider doesn't support it.
        """
        full_text = ""
        try:
            stream = await self._router.chat(
                messages=messages,
                provider=provider,
                model=model,
                task=task,
                stream=True,
            )
            async for chunk in stream:
                if chunk.delta:
                    on_token(chunk.delta)
                    full_text += chunk.delta
            return full_text, 0, 0  # token counts not available mid-stream

        except Exception as e:
            logger.warning(f"Streaming failed: {e} — falling back to non-stream")
            resp = await self._router.chat(
                messages=messages,
                provider=provider,
                model=model,
                task=task,
                stream=False,
            )
            on_token(resp.text)
            return resp.text, resp.input_tokens, resp.output_tokens

    # ── Vision Turn ───────────────────────────────────────────────────────────

    async def _vision_turn(
        self,
        user_input: str,
        image: Union[str, bytes],
        provider: Optional[str],
        model: Optional[str],
        t0: float,
    ) -> TurnResult:
        vision_provider = provider or self._vision._provider
        vision_model    = model    or self._vision._model

        result = await self._vision.analyze(
            image=image,
            prompt=user_input or "Describe this image.",
            provider=vision_provider,
            model=vision_model,
        )

        self._prompt.commit(
            user_input=f"[image] {user_input}",
            assistant_reply=result.text,
            provider=result.provider,
            model=result.model,
            store_in_vector=True,
        )

        return TurnResult(
            reply=result.text,
            provider=result.provider,
            model=result.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ── Control ───────────────────────────────────────────────────────────────

    def set_provider(self, provider: str, model: Optional[str] = None) -> None:
        """Switch default provider mid-session."""
        _ = self._router.get(provider)  # validates it exists
        self._default_provider = provider
        if model:
            self._router.get(provider).default_model = model
        logger.info(f"Provider switched → {provider}" + (f"/{model}" if model else ""))

    def clear_session(self) -> None:
        """Wipe session memory. Vector memory persists."""
        self._prompt.clear_session(keep_system=True)

    def reset(self) -> None:
        """Full reset: session + vector memory."""
        self._prompt.clear_session(keep_system=False)
        self._vector.clear_all()

    @property
    def providers(self) -> list[str]:
        return self._router.list_providers()

    @property
    def skills(self) -> list[str]:
        return self._registry.list_names()

    def status(self) -> dict:
        return {
            "provider": self._default_provider,
            "providers_loaded": self.providers,
            "skills": self.skills,
            **self._prompt.status(),
        }

    def __repr__(self) -> str:
        return (
            f"<Agent provider={self._default_provider} "
            f"skills={len(self._registry)} "
            f"session_turns={self._session.turn_count}>"
        )
