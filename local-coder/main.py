"""Entry point and REPL for localcoder."""

import curses
import json
import os
import subprocess
import sys
import time

import click

from agent import Agent
from llm import OllamaClient
from memory import MemoryStore
from ui import COMMANDS, CursesUI


def load_config(project_dir: str) -> dict:
    config_path = os.path.join(project_dir, ".coder", "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


# ── Slash-command handler ────────────────────────────────────────

def handle_command(command: str, agent: Agent, client: OllamaClient, ui: CursesUI) -> bool:
    """Handle a /command. Returns False to quit."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return False

    elif cmd == "/clear":
        agent.reset()
        ui.show_info("Conversation cleared.")

    elif cmd == "/plan":
        agent.set_planning(not agent.planning)
        ui.mode = "planning" if agent.planning else "direct"
        ui.show_info(f"Switched to {ui.mode} mode.")
        ui.draw_status()
        ui.stdscr.refresh()

    elif cmd == "/model":
        if arg:
            client.model = arg
            ui.model = arg
            ui.show_info(f"Model set to: {arg}")
            ui.draw_status()
            ui.stdscr.refresh()
        else:
            ui.show_info(f"Current model: {client.model}")

    elif cmd == "/memory":
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0] if sub_parts else "list"
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub_cmd == "list":
            ui.show_memories(agent.memory.list_all())
        elif sub_cmd == "search" and sub_arg:
            ui.show_memories(agent.memory.search(sub_arg))
        elif sub_cmd == "forget" and sub_arg:
            if agent.memory.delete(sub_arg):
                ui.memory_count = agent.memory.count()
                ui.show_info(f"Memory {sub_arg} deleted.")
                ui.draw_status()
                ui.stdscr.refresh()
            else:
                ui.show_error(f"Memory '{sub_arg}' not found.")
        else:
            ui.show_info("Usage: /memory [list | search <query> | forget <id>]")

    elif cmd == "/help":
        ui.show_help()

    else:
        ui.show_error(f"Unknown command: {cmd}. Type /help for commands.")

    return True


# ── Setup phases (run inside curses) ─────────────────────────────

def setup_project_dir(ui: CursesUI) -> str | None:
    """Prompt for and validate/create the project directory. Returns path or None."""
    while True:
        raw = ui.setup_prompt("Enter your project directory:", hint="Tab to auto-complete paths")
        if raw is None:
            return None

        raw = raw.strip()
        if not raw:
            ui.setup_status("Please specify a directory.", ok=False)
            continue

        path = os.path.abspath(os.path.expanduser(raw))

        if os.path.isdir(path):
            ui.setup_status(f"Project: {path}")
            return path

        if os.path.exists(path):
            ui.setup_status(f"'{path}' exists but is not a directory.", ok=False)
            continue

        # Offer to create
        if ui.setup_confirm(f"'{path}' does not exist. Create it?"):
            try:
                os.makedirs(path, exist_ok=True)
                ui.setup_status(f"Created {path}")
                return path
            except OSError as e:
                ui.setup_status(f"Could not create directory: {e}", ok=False)
        else:
            ui.setup_status("Directory not created.", ok=False)


def setup_ollama(ui: CursesUI, client: OllamaClient) -> bool:
    """Ensure Ollama is running and the model is available. Returns True on success."""
    # Check connection
    if client.is_available():
        ui.setup_status("Ollama connected")
    else:
        ui.setup_status("Ollama not running -- starting...", ok=True)
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            started = False
            for _ in range(15):
                time.sleep(1)
                if client.is_available():
                    started = True
                    break
            if started:
                ui.setup_status("Ollama started")
            else:
                ui.setup_status("Could not start Ollama. Run 'ollama serve' manually.", ok=False)
                return False
        except FileNotFoundError:
            ui.setup_status("Ollama not found. Install from https://ollama.ai", ok=False)
            return False

    # Check model
    if client.has_model():
        ui.setup_status(f"Model {client.model} ready")
    else:
        ui.setup_status(f"Pulling {client.model} (this may take a while)...", ok=True)
        try:
            for status in client.pull_model():
                ui.setup_progress(f"  Pulling {client.model}: {status}")
            ui.setup_status(f"Model {client.model} ready")
        except Exception as e:
            ui.setup_status(f"Failed to pull model: {e}", ok=False)
            return False

    return True


# ── CLI entry point ──────────────────────────────────────────────

@click.command()
@click.option("--model", default=None, help="Ollama model to use (default: qwen3:8b)")
@click.option(
    "--project-dir",
    default=None,
    type=click.Path(),
    help="Project directory (skips the interactive prompt)",
)
def main(model: str | None, project_dir: str | None):
    """localcoder - Local AI Coding Assistant"""

    def curses_main(stdscr):
        ui = CursesUI(stdscr)

        # ── Setup screen ──────────────────────────────────
        ui.draw_setup_banner()

        # 1. Project directory
        if project_dir is not None:
            chosen = os.path.abspath(os.path.expanduser(project_dir))
            if not os.path.isdir(chosen):
                if ui.setup_confirm(f"'{chosen}' does not exist. Create it?"):
                    os.makedirs(chosen, exist_ok=True)
                    ui.setup_status(f"Created {chosen}")
                else:
                    ui.setup_status("Cannot continue without a project directory.", ok=False)
                    ui.setup_prompt("Press Enter to exit.")
                    return
            else:
                ui.setup_status(f"Project: {chosen}")
        else:
            chosen = setup_project_dir(ui)
            if chosen is None:
                return  # user cancelled

        os.chdir(chosen)

        # 2. Load per-project config
        config = load_config(chosen)
        effective_model = model or config.get("model", "qwen3:8b")

        # 3. Ollama + model
        client = OllamaClient(model=effective_model)
        if not setup_ollama(ui, client):
            ui.setup_prompt("Press Enter to exit.")
            return

        # 4. Memory
        memory = MemoryStore(project_dir=chosen)

        # ── Transition to REPL ────────────────────────────
        ui.setup_status("Ready!", ok=True)
        ui.setup_prompt("Press Enter to start.")

        ui.model = effective_model
        ui.memory_count = memory.count()
        ui.mode = "direct"
        ui.project_dir = chosen
        ui.setup_done()

        agent = Agent(client=client, memory=memory, ui=ui)
        ui.show_welcome()
        ui.refresh()

        # ── Main REPL ────────────────────────────────────
        while True:
            user_input = ui.get_input()

            if user_input is None:
                ui.show_info("Goodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_command(user_input, agent, client, ui):
                    break
                continue

            ui.show_user(user_input)
            try:
                agent.run(user_input)
            except KeyboardInterrupt:
                ui.show_info("Interrupted.")
            except Exception as e:
                ui.show_error(f"Agent error: {e}")

    curses.wrapper(curses_main)


if __name__ == "__main__":
    main()
