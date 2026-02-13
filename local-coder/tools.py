"""Tool registry and implementations for the coding agent."""

import glob
import os
import re
import subprocess


# --- Tool implementations ---

def read_file(path: str, start: int | None = None, end: int | None = None) -> str:
    """Read file contents, optionally a line range."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if start is not None or end is not None:
            s = (start or 1) - 1
            e = end or len(lines)
            lines = lines[s:e]
            # Add line numbers
            return "".join(
                f"{i}: {line}" for i, line in enumerate(lines, start=s + 1)
            )
        if len(lines) > 500:
            return (
                f"File has {len(lines)} lines. Showing first 500:\n"
                + "".join(f"{i}: {line}" for i, line in enumerate(lines[:500], start=1))
            )
        return "".join(f"{i}: {line}" for i, line in enumerate(lines, start=1))
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file."""
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        lines = content.count("\n") + (0 if content.endswith("\n") else 1)
        return f"Wrote {lines} lines to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Find and replace text in a file."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r") as f:
            content = f.read()
        count = content.count(old_text)
        if count == 0:
            return "Error: old_text not found in file."
        if count > 1:
            return f"Warning: old_text found {count} times. Replacing all occurrences."
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Edited {path}: replaced {count} occurrence(s)."
    except Exception as e:
        return f"Error editing file: {e}"


def run_command(command: str) -> str:
    """Execute a shell command. Returns stdout+stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if not output:
            output = "(no output)"
        output += f"\n[exit code: {result.returncode}]"
        # Truncate very long output
        if len(output) > 10000:
            output = output[:5000] + "\n... (truncated) ...\n" + output[-2000:]
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def search_files(pattern: str, path: str | None = None) -> str:
    """Glob search for files matching a pattern."""
    base = path or "."
    base = os.path.expanduser(base)
    try:
        matches = glob.glob(os.path.join(base, pattern), recursive=True)
        # Filter out hidden dirs and common ignores
        filtered = [
            m for m in matches
            if not any(
                part.startswith(".") and part not in (".", "..")
                for part in m.split(os.sep)
            )
            and "__pycache__" not in m
            and "node_modules" not in m
        ]
        if not filtered:
            return "No files found."
        # Limit results
        if len(filtered) > 100:
            return "\n".join(filtered[:100]) + f"\n... and {len(filtered) - 100} more"
        return "\n".join(sorted(filtered))
    except Exception as e:
        return f"Error searching files: {e}"


def search_content(
    regex: str, path: str | None = None, file_glob: str | None = None
) -> str:
    """Search file contents with regex."""
    base = path or "."
    base = os.path.expanduser(base)

    # Find files to search
    if file_glob:
        files = glob.glob(os.path.join(base, file_glob), recursive=True)
    else:
        files = glob.glob(os.path.join(base, "**/*"), recursive=True)

    files = [
        f for f in files
        if os.path.isfile(f)
        and not any(
            part.startswith(".") and part not in (".", "..")
            for part in f.split(os.sep)
        )
        and "__pycache__" not in f
        and "node_modules" not in f
    ]

    try:
        pat = re.compile(regex, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    for fpath in files:
        try:
            with open(fpath, "r", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        results.append(f"{fpath}:{i}: {line.rstrip()}")
                        if len(results) >= 50:
                            results.append("... (results truncated at 50 matches)")
                            return "\n".join(results)
        except (OSError, UnicodeDecodeError):
            continue
    if not results:
        return "No matches found."
    return "\n".join(results)


# --- Tool schemas for Ollama ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns line-numbered content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Starting line number (1-based, optional)",
                    },
                    "end": {
                        "type": "integer",
                        "description": "Ending line number (inclusive, optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file. The old_text must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output. Use for running tests, installing packages, git operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a glob pattern (e.g. '**/*.py', '**/test_*').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in (default: current directory)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "Search file contents using a regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regex": {
                        "type": "string",
                        "description": "Regular expression to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in (default: current directory)",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '**/*.py')",
                    },
                },
                "required": ["regex"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save information to persistent project memory. Use this to remember project conventions, decisions, or task summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags for categorization (e.g. 'project,convention')",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search project memories for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a clarifying question. Pauses execution until the user responds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal that the current task is complete. Provide a summary of what was done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

# Map tool names to implementations
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
    "search_files": search_files,
    "search_content": search_content,
    # remember, recall, ask_user, task_complete are handled by the agent
}

# Tools that need user approval before execution
APPROVAL_REQUIRED = {"run_command", "write_file", "edit_file"}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {e}"


def get_tool_schemas() -> list:
    """Return the tool schemas for the Ollama API."""
    return TOOL_SCHEMAS
