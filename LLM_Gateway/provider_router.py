# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — LLM Gateway / Provider Router           ║
# ║  Unified interface over 25+ providers.                          ║
# ║  All adapters return the same ProviderResponse dataclass.       ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


# ── Shared Response Schema ────────────────────────────────────────────────────

@dataclass
class ProviderResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)


@dataclass
class StreamChunk:
    delta: str
    done: bool = False
    provider: str = ""
    model: str = ""


# ── Message Schema ────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str        # "system" | "user" | "assistant"
    content: str


# ── Base Adapter ──────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.default_model = cfg.get("default_model", "")
        self.timeout = cfg.get("timeout", 60)
        self.max_retries = cfg.get("max_retries", 3)
        self.extra = cfg.get("extra", {})

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse | AsyncIterator[StreamChunk]:
        ...

    def _model(self, model: Optional[str]) -> str:
        return model or self.default_model

    def _api_key(self, env_var: str) -> str:
        key = os.getenv(env_var, "")
        if not key:
            raise ValueError(f"Missing env var: {env_var}")
        return key


# ── OpenAI Adapter ────────────────────────────────────────────────────────────

class OpenAIAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=cfg.get("base_url") or None,
            organization=self.extra.get("org_id") or None,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        model = self._model(model)

        if stream:
            return self._stream(msgs, model, t0)

        resp = await self._client.chat.completions.create(
            model=model, messages=msgs, **kwargs
        )
        return ProviderResponse(
            text=resp.choices[0].message.content,
            provider=self.name,
            model=model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )

    async def _stream(self, msgs, model, t0):
        async with await self._client.chat.completions.create(
            model=model, messages=msgs, stream=True
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                done = chunk.choices[0].finish_reason is not None
                yield StreamChunk(delta=delta, done=done, provider=self.name, model=model)


# ── Anthropic Adapter ─────────────────────────────────────────────────────────

class AnthropicAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=cfg.get("base_url") or None,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        self._max_tokens = self.extra.get("max_tokens", 8096)

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)

        # Anthropic separates system prompt from messages
        system = next((m.content for m in messages if m.role == "system"), "")
        msgs = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]

        if stream:
            return self._stream(system, msgs, model, t0)

        resp = await self._client.messages.create(
            model=model,
            system=system,
            messages=msgs,
            max_tokens=self._max_tokens,
            **kwargs,
        )
        return ProviderResponse(
            text=resp.content[0].text,
            provider=self.name,
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )

    async def _stream(self, system, msgs, model, t0):
        async with self._client.messages.stream(
            model=model,
            system=system,
            messages=msgs,
            max_tokens=self._max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text, provider=self.name, model=model)
            yield StreamChunk(delta="", done=True, provider=self.name, model=model)


# ── Gemini Adapter ────────────────────────────────────────────────────────────

class GeminiAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        self._genai = genai

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        gmodel = self._genai.GenerativeModel(model)

        history = []
        prompt = ""
        for m in messages:
            if m.role == "system":
                prompt = m.content + "\n\n"
            elif m.role == "user":
                history.append({"role": "user", "parts": [prompt + m.content]})
                prompt = ""
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        last_user = history.pop() if history and history[-1]["role"] == "user" else None
        chat = gmodel.start_chat(history=history)

        if last_user:
            user_text = last_user["parts"][0]
        else:
            user_text = ""

        resp = await asyncio.to_thread(chat.send_message, user_text)

        return ProviderResponse(
            text=resp.text,
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── Groq Adapter ──────────────────────────────────────────────────────────────

class GroqAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        from groq import AsyncGroq
        self._client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY", ""),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        model = self._model(model)

        resp = await self._client.chat.completions.create(
            model=model, messages=msgs, **kwargs
        )
        return ProviderResponse(
            text=resp.choices[0].message.content,
            provider=self.name,
            model=model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── Mistral Adapter ───────────────────────────────────────────────────────────

class MistralAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        from mistralai import Mistral
        self._client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        model = self._model(model)

        resp = await self._client.chat.complete_async(
            model=model, messages=msgs, **kwargs
        )
        return ProviderResponse(
            text=resp.choices[0].message.content,
            provider=self.name,
            model=model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── Amazon Bedrock Adapter ────────────────────────────────────────────────────

class BedrockAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import boto3, json
        self._json = json
        region = self.extra.get("region", os.getenv("AWS_REGION", "us-east-1"))
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)

        system = next((m.content for m in messages if m.role == "system"), "")
        msgs = [
            {"role": m.role, "content": [{"type": "text", "text": m.content}]}
            for m in messages if m.role != "system"
        ]

        body = self._json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system,
            "messages": msgs,
        })

        resp = await asyncio.to_thread(
            self._client.invoke_model,
            modelId=model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = self._json.loads(resp["body"].read())
        return ProviderResponse(
            text=result["content"][0]["text"],
            provider=self.name,
            model=model,
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=result,
        )


# ── OpenAI-Compatible Adapter (httpx) ─────────────────────────────────────────
# Covers: OpenRouter, DeepSeek, xAI, Fireworks, DashScope, BytePlus,
#         Moonshot, StepFun, Chutes, Venice, Z.AI, OpenCode, Vercel

class OpenAICompatAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        api_key_env = self.extra.get("api_key_env", "")
        self._api_key = os.getenv(api_key_env, "") if api_key_env else ""
        self._base_url = cfg.get("base_url", "").rstrip("/")
        extra_headers = self.extra.get("headers", {})
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **extra_headers,
        }

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        payload = {"model": model, "messages": msgs, **kwargs}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return ProviderResponse(
                        text=data["choices"][0]["message"]["content"],
                        provider=self.name,
                        model=model,
                        input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                        output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                        latency_ms=(time.monotonic() - t0) * 1000,
                        raw=data,
                    )
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)


# ── Cloudflare AI Gateway Adapter ────────────────────────────────────────────

class CloudflareAdapter(OpenAICompatAdapter):
    def __init__(self, name: str, cfg: dict):
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        gateway_id = os.getenv("CLOUDFLARE_GATEWAY_ID", "")
        cfg["base_url"] = (
            f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/openai"
        )
        cfg.setdefault("extra", {})["api_key_env"] = "CLOUDFLARE_API_KEY"
        super().__init__(name, cfg)


# ── MiniMax Adapter ───────────────────────────────────────────────────────────

class MinimaxAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._api_key = os.getenv("MINIMAX_API_KEY", "")
        self._group_id = os.getenv("MINIMAX_GROUP_ID", "")
        self._base_url = cfg.get("base_url", "https://api.minimax.chat/v1")

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        msgs = [{"sender_type": "USER" if m.role == "user" else "BOT",
                 "text": m.content} for m in messages if m.role != "system"]
        system = next((m.content for m in messages if m.role == "system"), "")

        payload = {
            "model": model,
            "messages": msgs,
            "system_prompt": system,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/text/chatcompletion_v2?GroupId={self._group_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return ProviderResponse(
            text=data["choices"][0]["messages"][0]["text"],
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=data,
        )


# ── Zhipu (GLM) Adapter ───────────────────────────────────────────────────────

class ZhipuAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        from zhipuai import ZhipuAI
        self._client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY", ""))

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        resp = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=model, messages=msgs, **kwargs
        )
        return ProviderResponse(
            text=resp.choices[0].message.content,
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── Qianfan (Baidu) Adapter ───────────────────────────────────────────────────

class QianfanAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import qianfan
        os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY", "")
        os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY", "")
        self._qianfan = qianfan

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        msgs = [{"role": m.role, "content": m.content}
                for m in messages if m.role != "system"]

        chat_comp = self._qianfan.ChatCompletion()
        resp = await asyncio.to_thread(
            chat_comp.do, model=model, messages=msgs, **kwargs
        )
        return ProviderResponse(
            text=resp["result"],
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── Ollama Adapter ────────────────────────────────────────────────────────────

class OllamaAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import ollama as _ollama
        self._ollama = _ollama
        self._base_url = cfg.get("base_url", "http://localhost:11434")

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        resp = await asyncio.to_thread(
            self._ollama.chat, model=model, messages=msgs
        )
        return ProviderResponse(
            text=resp["message"]["content"],
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=resp,
        )


# ── fal Adapter ───────────────────────────────────────────────────────────────

class FalAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        import fal_client
        os.environ["FAL_KEY"] = os.getenv("FAL_KEY", "")
        self._fal = fal_client

    async def chat(self, messages, model=None, stream=False, **kwargs):
        """fal is primarily image/video — last user message treated as prompt."""
        t0 = time.monotonic()
        model = self._model(model)
        prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        result = await asyncio.to_thread(
            self._fal.run, model, arguments={"prompt": prompt, **kwargs}
        )
        url = result.get("images", [{}])[0].get("url", str(result))
        return ProviderResponse(
            text=url,
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=result,
        )


# ── Runway Adapter ────────────────────────────────────────────────────────────

class RunwayAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._api_key = os.getenv("RUNWAY_API_KEY", "")
        self._base_url = cfg.get("base_url", "https://api.dev.runwayml.com/v1")

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        model = self._model(model)
        prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06",
        }
        payload = {"model": model, "promptText": prompt, **kwargs}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self._base_url}/image_to_video",
                headers=headers, json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return ProviderResponse(
            text=data.get("id", str(data)),
            provider=self.name,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=data,
        )


# ── ComfyUI Adapter ───────────────────────────────────────────────────────────

class ComfyUIAdapter(BaseAdapter):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._base_url = cfg.get("base_url", "http://localhost:8188")

    async def chat(self, messages, model=None, stream=False, **kwargs):
        t0 = time.monotonic()
        prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        workflow = kwargs.get("workflow", {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42, "steps": 20, "cfg": 7,
                    "sampler_name": "euler", "scheduler": "normal",
                    "positive": prompt,
                },
            }
        })
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self._base_url}/prompt",
                json={"prompt": workflow},
            )
            resp.raise_for_status()
            data = resp.json()

        return ProviderResponse(
            text=data.get("prompt_id", str(data)),
            provider=self.name,
            model="comfyui-workflow",
            latency_ms=(time.monotonic() - t0) * 1000,
            raw=data,
        )


# ── Adapter Registry ──────────────────────────────────────────────────────────

ADAPTER_MAP: dict[str, type[BaseAdapter]] = {
    "OpenAIAdapter":       OpenAIAdapter,
    "AnthropicAdapter":    AnthropicAdapter,
    "GeminiAdapter":       GeminiAdapter,
    "GroqAdapter":         GroqAdapter,
    "MistralAdapter":      MistralAdapter,
    "BedrockAdapter":      BedrockAdapter,
    "OpenAICompatAdapter": OpenAICompatAdapter,
    "CloudflareAdapter":   CloudflareAdapter,
    "MinimaxAdapter":      MinimaxAdapter,
    "ZhipuAdapter":        ZhipuAdapter,
    "QianfanAdapter":      QianfanAdapter,
    "OllamaAdapter":       OllamaAdapter,
    "FalAdapter":          FalAdapter,
    "RunwayAdapter":       RunwayAdapter,
    "ComfyUIAdapter":      ComfyUIAdapter,
}


# ── Provider Router ───────────────────────────────────────────────────────────

class ProviderRouter:
    """
    Central router. Loads all enabled providers from config,
    handles task-based routing, and runs the fallback chain.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self._providers: dict[str, BaseAdapter] = {}
        self._routing: dict[str, str] = self._cfg.get("routing", {})
        self._fallback_chain: list[str] = self._cfg.get("fallback_chain", [])
        self._default: str = self._cfg["app"]["default_provider"]

        self._load_providers()

    def _load_providers(self):
        for name, pcfg in self._cfg.get("providers", {}).items():
            if not pcfg.get("enabled", True):
                continue
            adapter_cls_name = pcfg.get("adapter", "OpenAICompatAdapter")
            adapter_cls = ADAPTER_MAP.get(adapter_cls_name)
            if not adapter_cls:
                logger.warning(f"Unknown adapter '{adapter_cls_name}' for provider '{name}' — skipping")
                continue
            try:
                self._providers[name] = adapter_cls(name, pcfg)
                logger.debug(f"Loaded provider: {name} ({adapter_cls_name})")
            except Exception as e:
                logger.warning(f"Failed to init provider '{name}': {e}")

    def get(self, provider: str) -> BaseAdapter:
        if provider not in self._providers:
            raise ValueError(f"Provider '{provider}' not loaded. Check config + env keys.")
        return self._providers[provider]

    def route(self, task: str) -> BaseAdapter:
        """Return adapter for a named task (code, reasoning, fast, etc.)"""
        provider_name = self._routing.get(task, self._default)
        return self.get(provider_name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    async def chat(
        self,
        messages: list[Message],
        provider: Optional[str] = None,
        task: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        use_fallback: bool = True,
        **kwargs,
    ) -> ProviderResponse | AsyncIterator[StreamChunk]:
        """
        Main entry point.
        Priority: explicit provider > task routing > default > fallback chain
        """
        # Resolve initial adapter
        if provider:
            adapter = self.get(provider)
        elif task:
            adapter = self.route(task)
        else:
            adapter = self.get(self._default)

        chain = [adapter.name]
        if use_fallback:
            for fb in self._fallback_chain:
                if fb not in chain and fb in self._providers:
                    chain.append(fb)

        last_error = None
        for provider_name in chain:
            try:
                adp = self._providers[provider_name]
                logger.debug(f"Trying provider: {provider_name}")
                result = await adp.chat(messages, model=model, stream=stream, **kwargs)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                if provider_name == chain[-1]:
                    break
                logger.info(f"Falling back to next provider in chain...")

        raise RuntimeError(
            f"All providers exhausted. Last error: {last_error}"
        )
