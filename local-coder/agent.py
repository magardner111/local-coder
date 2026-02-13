"""Agent loop: LLM <-> tool execution cycle."""

import re

from llm import OllamaClient
from memory import MemoryStore
from prompts import build_system_prompt
from tools import APPROVAL_REQUIRED, execute_tool, get_tool_schemas
from ui import CursesUI


class Agent:
    """Agentic coding assistant that loops between LLM and tools."""

    def __init__(self, client: OllamaClient, memory: MemoryStore, ui: CursesUI):
        self.client = client
        self.memory = memory
        self.ui = ui
        self.history: list[dict] = []
        self.planning = False
        self.max_tool_rounds = 15

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def set_planning(self, enabled: bool):
        self.planning = enabled

    def _system_message(self, user_input: str = "") -> dict:
        mem_ctx = self.memory.get_context(query=user_input)
        prompt = build_system_prompt(memory_context=mem_ctx, planning=self.planning)
        return {"role": "system", "content": prompt}

    def run(self, user_input: str):
        """Run the full agent loop for one user turn."""
        self.history.append({"role": "user", "content": user_input})
        tools = get_tool_schemas()

        for _ in range(self.max_tool_rounds):
            messages = [self._system_message(user_input)] + self.history

            # Stream LLM response
            self.ui.stream_start()
            tool_calls = []
            full_text = ""

            try:
                for chunk_type, data in self.client.chat_stream(messages, tools):
                    if chunk_type == "text":
                        self.ui.stream_token(data)
                        full_text += data
                    elif chunk_type == "tool_call":
                        tool_calls.append(data)
            except KeyboardInterrupt:
                self.ui.stream_end()
                raise
            except Exception as e:
                self.ui.stream_end()
                self.ui.show_error(f"LLM error: {e}")
                return

            self.ui.stream_end()

            # Strip thinking blocks from stored text
            clean_text = re.sub(
                r"<think>.*?</think>", "", full_text, flags=re.DOTALL
            ).strip()

            # Record assistant message
            if clean_text or tool_calls:
                assistant_msg = {"role": "assistant", "content": clean_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ]
                self.history.append(assistant_msg)

            # No tool calls → done
            if not tool_calls:
                return

            # Execute each tool call
            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                self.ui.show_tool_call(name, args)

                result = self._execute_tool_call(name, args)
                if result is None:
                    self.history.append({
                        "role": "tool",
                        "content": f"User denied execution of {name}.",
                    })
                    self.ui.show_info("Action denied by user.")
                    continue

                is_error = result.startswith("Error")
                self.ui.show_tool_result(name, result, is_error=is_error)
                self.history.append({"role": "tool", "content": result})

                if name == "task_complete":
                    return

        self.ui.show_info("Reached maximum tool rounds. Stopping.")

    def _execute_tool_call(self, name: str, arguments: dict) -> str | None:
        """Execute a tool, handling agent-level tools and approvals."""

        if name == "ask_user":
            answer = self.ui.get_question_response(arguments.get("question", ""))
            return f"User answered: {answer}"

        if name == "remember":
            content = arguments.get("content", "")
            tags_str = arguments.get("tags", "")
            tags = (
                [t.strip() for t in tags_str.split(",") if t.strip()]
                if tags_str
                else []
            )
            mem = self.memory.add(content, tags=tags, mem_type="project")
            self.ui.memory_count = self.memory.count()
            self.ui.draw_status()
            self.ui.stdscr.refresh()
            if isinstance(mem, list):
                return f"Saved {len(mem)} memory chunks."
            return f"Saved to memory (id: {mem['id']})."

        if name == "recall":
            query = arguments.get("query", "")
            results = self.memory.search(query)
            if not results:
                return "No relevant memories found."
            lines = []
            for m in results:
                tags = ", ".join(m.get("tags", []))
                entry = f"[{m['type']}] {m['content']}"
                if tags:
                    entry += f" (tags: {tags})"
                lines.append(entry)
            return "\n".join(lines)

        if name == "task_complete":
            summary = arguments.get("summary", "")
            self.memory.add(summary, mem_type="task")
            self.ui.memory_count = self.memory.count()
            self.ui.show_info(f"Task complete: {summary}")
            return "Task summary saved to memory."

        # Approval gate for write/edit/run
        if name in APPROVAL_REQUIRED:
            desc = f"{name}({', '.join(f'{k}={repr(v)[:80]}' for k, v in arguments.items())})"
            if not self.ui.get_approval(desc):
                return None

        return execute_tool(name, arguments)
