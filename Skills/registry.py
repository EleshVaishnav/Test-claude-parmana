# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — Skills Registry                         ║
# ║  Central tool manager. Registers, validates, and executes       ║
# ║  skills. Produces the tool manifest for the agent brain.        ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ── Skill Schema ──────────────────────────────────────────────────────────────

@dataclass
class SkillParam:
    name: str
    type: str                          # "string" | "number" | "boolean" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list] = None        # constrained value set


@dataclass
class Skill:
    name: str                          # snake_case identifier used by agent
    description: str                   # what it does — shown in system prompt
    params: list[SkillParam]
    handler: Callable                  # sync or async function
    enabled: bool = True
    tags: list[str] = field(default_factory=list)   # e.g. ["search", "io"]
    timeout: float = 30.0              # seconds before skill is aborted

    def to_manifest(self) -> dict:
        """OpenAI-style function schema for agent tool calls."""
        properties = {}
        required = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# ── Execution Result ──────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    skill_name: str
    success: bool
    output: Any                        # raw return value from handler
    error: Optional[str] = None
    latency_ms: float = 0.0

    def as_text(self) -> str:
        if self.success:
            return str(self.output)
        return f"[{self.skill_name} ERROR] {self.error}"


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Central registry for all Parmana skills/tools.

    Usage:
        registry = SkillRegistry()
        registry.register(web_search_skill)
        registry.register(calculator_skill)

        # Agent calls a tool
        result = await registry.call("web_search", {"query": "latest rust release"})

        # Get manifest for system prompt
        manifest_text = registry.manifest_text()
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' already registered — overwriting.")
        self._skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name} (enabled={skill.enabled})")

    def register_fn(
        self,
        fn: Callable,
        name: Optional[str] = None,
        description: str = "",
        params: Optional[list[SkillParam]] = None,
        tags: Optional[list[str]] = None,
        timeout: float = 30.0,
    ) -> Skill:
        """
        Decorator-friendly shorthand. Wraps a plain function as a Skill.

        Example:
            @registry.register_fn
            async def calculator(expression: str) -> str:
                ...
        """
        skill_name = name or fn.__name__
        skill = Skill(
            name=skill_name,
            description=description or (inspect.getdoc(fn) or ""),
            params=params or [],
            handler=fn,
            tags=tags or [],
            timeout=timeout,
        )
        self.register(skill)
        return skill

    def enable(self, name: str) -> None:
        self._get(name).enabled = True

    def disable(self, name: str) -> None:
        self._get(name).enabled = False

    # ── Execution ─────────────────────────────────────────────────────────────

    async def call(self, name: str, args: dict[str, Any]) -> SkillResult:
        """
        Execute a skill by name with given args.
        Handles sync/async, timeout, and error capture.
        """
        t0 = time.monotonic()

        try:
            skill = self._get(name)
        except KeyError:
            return SkillResult(
                skill_name=name,
                success=False,
                output=None,
                error=f"Unknown skill '{name}'",
                latency_ms=0.0,
            )

        if not skill.enabled:
            return SkillResult(
                skill_name=name,
                success=False,
                output=None,
                error=f"Skill '{name}' is disabled.",
            )

        # Validate required params
        missing = [
            p.name for p in skill.params
            if p.required and p.name not in args
        ]
        if missing:
            return SkillResult(
                skill_name=name,
                success=False,
                output=None,
                error=f"Missing required params: {missing}",
            )

        # Fill defaults for optional params
        for p in skill.params:
            if p.name not in args and p.default is not None:
                args[p.name] = p.default

        # Execute
        try:
            if asyncio.iscoroutinefunction(skill.handler):
                output = await asyncio.wait_for(
                    skill.handler(**args),
                    timeout=skill.timeout,
                )
            else:
                output = await asyncio.wait_for(
                    asyncio.to_thread(skill.handler, **args),
                    timeout=skill.timeout,
                )

            latency = (time.monotonic() - t0) * 1000
            logger.debug(f"Skill '{name}' succeeded in {latency:.1f}ms")
            return SkillResult(
                skill_name=name,
                success=True,
                output=output,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return SkillResult(
                skill_name=name,
                success=False,
                output=None,
                error=f"Skill '{name}' timed out after {skill.timeout}s",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            logger.warning(f"Skill '{name}' raised: {e}\n{traceback.format_exc()}")
            return SkillResult(
                skill_name=name,
                success=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def call_sync(self, name: str, args: dict[str, Any]) -> SkillResult:
        """Sync wrapper for non-async callers."""
        return asyncio.get_event_loop().run_until_complete(self.call(name, args))

    # ── Introspection ─────────────────────────────────────────────────────────

    def _get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(name)
        return self._skills[name]

    def list_skills(self, enabled_only: bool = True) -> list[Skill]:
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return skills

    def list_names(self, enabled_only: bool = True) -> list[str]:
        return [s.name for s in self.list_skills(enabled_only)]

    def get_manifest(self, enabled_only: bool = True) -> list[dict]:
        """OpenAI-style tool manifest list."""
        return [s.to_manifest() for s in self.list_skills(enabled_only)]

    def manifest_text(self, enabled_only: bool = True) -> str:
        """
        Human-readable tool list for injection into system prompt.
        Format matches what Parmana's system_prompt.txt expects.
        """
        skills = self.list_skills(enabled_only)
        if not skills:
            return "(no tools enabled)"

        lines = []
        for s in skills:
            param_summary = ", ".join(
                f"{p.name}: {p.type}{'?' if not p.required else ''}"
                for p in s.params
            )
            lines.append(f"- **{s.name}**({param_summary}): {s.description}")

        return "## Available Tools\n" + "\n".join(lines)

    def has(self, name: str) -> bool:
        return name in self._skills and self._skills[name].enabled

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        enabled = sum(1 for s in self._skills.values() if s.enabled)
        return f"<SkillRegistry total={len(self._skills)} enabled={enabled}>"


# ── Global Registry Instance ──────────────────────────────────────────────────
# Import this singleton in skills and agent.py

registry = SkillRegistry()
