"""Ollama streaming API client with tool calling support."""

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class LLMResponse:
    """Accumulated response from the LLM."""
    text: str = ""
    tool_calls: list = field(default_factory=list)
    done: bool = False
    thinking: str = ""


class OllamaClient:
    """Streaming chat client for Ollama with tool calling."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx

    def _build_payload(self, messages: list, tools: list | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    def chat_stream(self, messages: list, tools: list | None = None):
        """Stream a chat response, yielding (chunk_type, data) tuples.

        Chunk types:
          - "text": data is a text token string
          - "thinking": data is a thinking token string
          - "tool_call": data is a dict with tool call info
          - "done": data is the final LLMResponse
        """
        payload = self._build_payload(messages, tools)
        response = LLMResponse()

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300.0,
        ) as stream:
            for line in stream.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)

                if chunk.get("done"):
                    response.done = True
                    yield ("done", response)
                    return

                msg = chunk.get("message", {})

                # Handle tool calls
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        call = {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                        response.tool_calls.append(call)
                        yield ("tool_call", call)

                # Handle text content
                content = msg.get("content", "")
                if content:
                    # Detect thinking blocks from Qwen3
                    response.text += content
                    yield ("text", content)

    def chat(self, messages: list, tools: list | None = None) -> LLMResponse:
        """Non-streaming chat, returns complete LLMResponse."""
        payload = self._build_payload(messages, tools)
        payload["stream"] = False

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()

        result = LLMResponse(done=True)
        msg = data.get("message", {})
        result.text = msg.get("content", "")

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                result.tool_calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
        return result

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def has_model(self) -> bool:
        """Check if the configured model is pulled."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(
                m["name"] == self.model or m["name"].startswith(f"{self.model}:")
                for m in models
            )
        except (httpx.ConnectError, httpx.HTTPError):
            return False

    def pull_model(self):
        """Pull the model, yielding status strings."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"model": self.model},
            timeout=None,
        ) as stream:
            for line in stream.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                status = data.get("status", "")
                total = data.get("total", 0)
                completed = data.get("completed", 0)
                if total:
                    pct = int(completed / total * 100)
                    yield f"{status}: {pct}%"
                else:
                    yield status
