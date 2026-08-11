"""LLM provider abstraction (spec §15).

The app talks to `LLMProvider`; concrete providers translate to Gemini or any
OpenAI-compatible endpoint (OpenAI, vLLM, Ollama). Tool calling is normalized
to ToolCall so the agent loop is provider-independent.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core import config

log = logging.getLogger("llm")


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None  # provider-native assistant content, replayed verbatim
    # (Gemini 3+ requires thought_signature echo in function-call turns)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema


class LLMProvider(ABC):
    """messages: [{role: user|assistant|tool, content: str, ...}]"""

    @abstractmethod
    def chat(self, system: str, messages: list[dict],
             tools: list[ToolSpec] | None = None) -> LLMResponse: ...


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from google import genai
        self.model = model or config.GEMINI_MODEL
        self.client = genai.Client()

    def chat(self, system, messages, tools=None):
        from google.genai import types, errors

        contents = []
        for m in messages:
            if m["role"] == "tool":
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(
                        name=m["name"], response={"result": m["content"]})]))
            elif m["role"] == "assistant" and m.get("tool_calls"):
                if m.get("raw") is not None:  # replay verbatim (thought_signature)
                    contents.append(m["raw"])
                else:
                    contents.append(types.Content(role="model", parts=[
                        types.Part(function_call=types.FunctionCall(
                            name=tc.name, args=tc.args)) for tc in m["tool_calls"]]))
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=0.2,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.parameters)
                for t in tools])] if tools else None,
        )
        import httpx

        for attempt in range(4):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=cfg)
                break
            except (errors.ClientError, errors.ServerError, httpx.HTTPError) as e:
                transient = isinstance(e, httpx.HTTPError) or \
                    getattr(e, "code", None) in (429, 503)
                if transient and attempt < 3:
                    wait = 10 * (attempt + 1)
                    log.warning("gemini transient error (%s), retry in %ss",
                                type(e).__name__, wait)
                    time.sleep(wait)
                    continue
                raise
        calls = [ToolCall(fc.name, dict(fc.args or {}))
                 for fc in (resp.function_calls or [])]
        raw = resp.candidates[0].content if resp.candidates else None
        return LLMResponse(text=resp.text or "", tool_calls=calls, raw=raw)


class OpenAICompatProvider(LLMProvider):
    """OpenAI / vLLM / Ollama — anything speaking /v1/chat/completions."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.model = model or config.OPENAI_MODEL

    def chat(self, system, messages, tools=None):
        import requests

        oai_messages = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                oai_messages.append({"role": "tool", "tool_call_id": m.get("id", m["name"]),
                                     "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                oai_messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.name, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                    for tc in m["tool_calls"]]})
            else:
                oai_messages.append({"role": m["role"], "content": m["content"]})

        body = {"model": self.model, "messages": oai_messages, "temperature": 0.2}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=120,
                          headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = [ToolCall(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                 for tc in msg.get("tool_calls") or []]
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)


def get_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "openai":
        return OpenAICompatProvider()
    return GeminiProvider()
