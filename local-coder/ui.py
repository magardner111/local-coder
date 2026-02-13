"""Curses-based terminal UI for localcoder."""

import curses
import glob as glob_mod
import os
import textwrap

COMMANDS = {
    "/plan": "Toggle planning/direct mode",
    "/memory": "Manage memories (list|search|forget)",
    "/clear": "Clear conversation history",
    "/model": "Show/change model (/model [name])",
    "/help": "Show help and tips",
    "/quit": "Exit localcoder",
}

COMMAND_NAMES = list(COMMANDS.keys())

# Color pair constants
C_DEFAULT = 0
C_AGENT = 1
C_TOOL = 2
C_QUESTION = 3
C_ERROR = 4
C_INFO = 5
C_MEMORY = 6
C_HEADER = 7
C_STATUS = 8
C_USER = 9
C_THINKING = 10


class CursesUI:
    """Full-screen curses UI with setup wizard and REPL."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.lines = []  # list of (color_pair, text) display lines
        self.scroll_offset = 0  # 0 = pinned to bottom
        self.model = "qwen3:8b"
        self.memory_count = 0
        self.mode = "direct"
        self.project_dir = "."

        # Streaming state
        self._stream_start = 0
        self._stream_buf = ""
        self._thinking = False
        self._think_buf = ""

        # Setup screen state
        self._setup_row = 0

        # Curses setup
        curses.curs_set(1)
        curses.use_default_colors()
        self._init_colors()

    def _init_colors(self):
        curses.init_pair(C_AGENT, curses.COLOR_GREEN, -1)
        curses.init_pair(C_TOOL, curses.COLOR_BLUE, -1)
        curses.init_pair(C_QUESTION, curses.COLOR_YELLOW, -1)
        curses.init_pair(C_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(C_INFO, curses.COLOR_CYAN, -1)
        curses.init_pair(C_MEMORY, curses.COLOR_MAGENTA, -1)
        curses.init_pair(C_HEADER, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(C_STATUS, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(C_USER, curses.COLOR_WHITE, -1)
        curses.init_pair(C_THINKING, curses.COLOR_BLACK, -1)

    # ================================================================
    #  Geometry helpers
    # ================================================================

    @property
    def h(self):
        return self.stdscr.getmaxyx()[0]

    @property
    def w(self):
        return self.stdscr.getmaxyx()[1]

    @property
    def content_h(self):
        """Rows available for content (between header and status bar)."""
        return max(1, self.h - 3)

    def _safe_addstr(self, y, x, text, attr=0):
        h, w = self.h, self.w
        if y < 0 or y >= h or x >= w:
            return
        try:
            self.stdscr.addnstr(y, x, text, w - x - 1, attr)
        except curses.error:
            pass

    # ================================================================
    #  Setup screen  (project dir, ollama checks)
    # ================================================================

    def draw_setup_banner(self):
        """Draw the centered setup banner and reset the setup row cursor."""
        self.stdscr.clear()
        h, w = self.h, self.w
        cy = max(2, h // 5)

        # Decorative separator
        sep = "-" * min(48, w - 4)
        cx = (w - len(sep)) // 2

        self._safe_addstr(cy, cx, sep,
                          curses.color_pair(C_AGENT) | curses.A_DIM)

        # Title
        title = "l o c a l c o d e r"
        self._safe_addstr(cy + 2, (w - len(title)) // 2, title,
                          curses.color_pair(C_AGENT) | curses.A_BOLD)

        # Subtitle
        sub = "Local AI Coding Assistant"
        self._safe_addstr(cy + 4, (w - len(sub)) // 2, sub,
                          curses.color_pair(C_INFO))

        # Bottom separator
        self._safe_addstr(cy + 6, cx, sep,
                          curses.color_pair(C_AGENT) | curses.A_DIM)

        self._setup_row = cy + 8
        self.stdscr.refresh()

    # --- setup input primitives ---

    def _setup_input(self, row, prompt, path_complete=False):
        """Read a line of input at a fixed row, with optional path tab-completion.

        Returns the entered string, or None on Ctrl-C.
        """
        buf = ""
        pos = 0
        hint_row = row + 1  # row used to show tab-completion candidates

        while True:
            # draw prompt + buffer
            try:
                self.stdscr.move(row, 0)
                self.stdscr.clrtoeol()
            except curses.error:
                pass
            display = prompt + buf
            self._safe_addstr(row, 0, display,
                              curses.color_pair(C_AGENT) | curses.A_BOLD)
            cx = len(prompt) + pos
            if cx < self.w:
                try:
                    self.stdscr.move(row, cx)
                except curses.error:
                    pass
            self.stdscr.refresh()

            try:
                ch = self.stdscr.getch()
            except KeyboardInterrupt:
                return None

            if ch in (curses.KEY_ENTER, 10, 13):
                # clear the hint row on submit
                try:
                    self.stdscr.move(hint_row, 0)
                    self.stdscr.clrtoeol()
                except curses.error:
                    pass
                return buf

            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if pos > 0:
                    buf = buf[:pos - 1] + buf[pos:]
                    pos -= 1

            elif ch == curses.KEY_LEFT:
                pos = max(0, pos - 1)
            elif ch == curses.KEY_RIGHT:
                pos = min(len(buf), pos + 1)
            elif ch in (curses.KEY_HOME, 1):
                pos = 0
            elif ch in (curses.KEY_END, 5):
                pos = len(buf)
            elif ch == 21:  # Ctrl-U
                buf = ""
                pos = 0

            elif ch == 9 and path_complete:  # Tab
                buf, pos = self._complete_path(buf, pos, hint_row)

            elif ch == 3:  # Ctrl-C
                return None

            elif ch == curses.KEY_RESIZE:
                pass  # setup is simple enough; just keep going

            elif ch == 27:  # consume ESC sequence
                self.stdscr.nodelay(True)
                self.stdscr.getch()
                self.stdscr.getch()
                self.stdscr.nodelay(False)

            elif 32 <= ch < 127:
                buf = buf[:pos] + chr(ch) + buf[pos:]
                pos += 1

    def _complete_path(self, buf, pos, hint_row):
        """Tab-complete filesystem paths."""
        path = os.path.expanduser(buf)

        # If it's already a directory without trailing /, add one
        if os.path.isdir(path) and not path.endswith("/"):
            result = buf + "/"
            return result, len(result)

        matches = glob_mod.glob(path + "*")
        # Annotate directories with /
        annotated = [m + ("/" if os.path.isdir(m) else "") for m in matches]

        if not annotated:
            return buf, pos

        if len(annotated) == 1:
            result = annotated[0]
            # Preserve ~ prefix
            if buf.startswith("~"):
                home = os.path.expanduser("~")
                if result.startswith(home):
                    result = "~" + result[len(home):]
            return result, len(result)

        # Multiple matches: advance to common prefix
        common = os.path.commonprefix(annotated)
        if len(common) > len(path):
            result = common
            if buf.startswith("~"):
                home = os.path.expanduser("~")
                if result.startswith(home):
                    result = "~" + result[len(home):]
            return result, len(result)

        # Show candidates on the hint row
        names = [
            os.path.basename(m.rstrip("/")) + ("/" if m.endswith("/") else "")
            for m in annotated
        ]
        display = "  ".join(names[:10])
        if len(names) > 10:
            display += f"  ... ({len(names)} total)"
        try:
            self.stdscr.move(hint_row, 0)
            self.stdscr.clrtoeol()
        except curses.error:
            pass
        self._safe_addstr(hint_row, 4, display,
                          curses.color_pair(C_INFO) | curses.A_DIM)
        self.stdscr.refresh()

        return buf, pos

    # --- high-level setup helpers ---

    def setup_prompt(self, label, hint=""):
        """Show a labelled prompt on the setup screen and return the input."""
        row = self._setup_row
        text = label
        if hint:
            text += f"  ({hint})"
        self._safe_addstr(row, 4, text, curses.color_pair(C_INFO))
        self.stdscr.refresh()
        self._setup_row = row + 1

        result = self._setup_input(self._setup_row, "  > ", path_complete=True)
        self._setup_row += 3  # leave space for hint row + gap
        return result

    def setup_confirm(self, question):
        """Ask a yes/no question on the setup screen."""
        row = self._setup_row
        self._safe_addstr(row, 4, question, curses.color_pair(C_QUESTION))
        self.stdscr.refresh()
        self._setup_row = row + 1

        resp = self._setup_input(self._setup_row, "  (y/n) > ")
        self._setup_row += 2
        if resp is None:
            return False
        return resp.strip().lower() in ("y", "yes")

    def setup_status(self, msg, ok=True):
        """Print a status line on the setup screen."""
        row = self._setup_row
        if ok:
            prefix = "  [ok] "
            attr = curses.color_pair(C_AGENT)
        else:
            prefix = "  [!!] "
            attr = curses.color_pair(C_ERROR)
        self._safe_addstr(row, 2, prefix + msg, attr)
        self._setup_row = row + 1
        self.stdscr.refresh()

    def setup_progress(self, msg):
        """Update the current status line in place (for download progress etc)."""
        row = max(0, self._setup_row - 1)
        try:
            self.stdscr.move(row, 0)
            self.stdscr.clrtoeol()
        except curses.error:
            pass
        self._safe_addstr(row, 4, msg, curses.color_pair(C_INFO))
        self.stdscr.refresh()

    def setup_done(self):
        """Transition from setup screen to normal REPL layout."""
        self.stdscr.clear()
        self.lines = []
        self.scroll_offset = 0
        self.refresh()

    # ================================================================
    #  Normal REPL drawing
    # ================================================================

    def draw_header(self):
        w = self.w
        cmds = "  ".join(COMMAND_NAMES)
        label = "[localcoder]"
        gap = max(1, w - len(cmds) - len(label) - 3)
        header = f" {cmds}{' ' * gap}{label} "
        attr = curses.color_pair(C_HEADER) | curses.A_BOLD
        self._safe_addstr(0, 0, header.ljust(w)[:w], attr)

    def draw_status(self):
        w = self.w
        row = self.h - 2
        parts = [self.model, f"{self.memory_count} memories", self.mode, self.project_dir]
        status = f" {' | '.join(parts)} "
        attr = curses.color_pair(C_STATUS) | curses.A_BOLD
        self._safe_addstr(row, 0, status.ljust(w)[:w], attr)

    def draw_content(self):
        ch = self.content_h
        total = len(self.lines)

        if total <= ch:
            start = 0
        else:
            start = total - ch - self.scroll_offset
            start = max(0, start)

        for i in range(ch):
            row = 1 + i
            idx = start + i
            try:
                self.stdscr.move(row, 0)
                self.stdscr.clrtoeol()
            except curses.error:
                continue
            if idx < total:
                color, text = self.lines[idx]
                text = text.replace("\t", "    ")
                text = "".join(c if c.isprintable() or c == " " else "?" for c in text)
                self._safe_addstr(row, 0, text, curses.color_pair(color))

    def draw_input_line(self, prompt, buf, cursor_pos):
        row = self.h - 1
        try:
            self.stdscr.move(row, 0)
            self.stdscr.clrtoeol()
        except curses.error:
            pass
        display = prompt + buf
        attr = curses.color_pair(C_AGENT) | curses.A_BOLD
        self._safe_addstr(row, 0, display, attr)
        cx = len(prompt) + cursor_pos
        if cx < self.w:
            try:
                self.stdscr.move(row, cx)
            except curses.error:
                pass

    def refresh(self):
        self.draw_header()
        self.draw_content()
        self.draw_status()
        self.stdscr.refresh()

    # ================================================================
    #  Content management
    # ================================================================

    def _wrap(self, text, color):
        w = max(10, self.w - 1)
        result = []
        for raw_line in text.split("\n"):
            if not raw_line:
                result.append((color, ""))
            else:
                for wl in textwrap.wrap(raw_line, w,
                                        break_long_words=True,
                                        break_on_hyphens=False) or [""]:
                    result.append((color, wl))
        return result

    def add_line(self, text, color=C_DEFAULT):
        self.lines.extend(self._wrap(text, color))
        self.scroll_offset = 0
        self.refresh()

    def add_blank(self):
        self.lines.append((C_DEFAULT, ""))

    # ================================================================
    #  Streaming
    # ================================================================

    def stream_start(self):
        self._stream_start = len(self.lines)
        self._stream_buf = ""
        self._thinking = False
        self._think_buf = ""

    def stream_token(self, token):
        combined = self._stream_buf + token

        if "<think>" in combined and not self._thinking:
            before, _, after = combined.partition("<think>")
            self._stream_buf = before
            self._thinking = True
            self._think_buf = after
            self._render_stream()
            return

        if self._thinking:
            self._think_buf += token
            if "</think>" in self._think_buf:
                _, _, after = self._think_buf.partition("</think>")
                self._thinking = False
                self._think_buf = ""
                self._stream_buf += after
            self._render_stream()
            return

        self._stream_buf += token
        self._render_stream()

    def _render_stream(self):
        del self.lines[self._stream_start:]

        if self._thinking:
            short = self._think_buf[-80:].replace("\n", " ").strip()
            if short:
                self.lines.append((C_THINKING, f"  thinking: {short}..."))
            else:
                self.lines.append((C_THINKING, "  thinking..."))

        if self._stream_buf:
            self.lines.extend(self._wrap(self._stream_buf, C_AGENT))

        self.scroll_offset = 0
        self.refresh()

    def stream_end(self):
        self._thinking = False
        self._think_buf = ""
        self._stream_buf = ""

    # ================================================================
    #  REPL input
    # ================================================================

    def get_input(self, prompt="> "):
        """Get user input with /command tab completion and Page Up/Down."""
        buf = ""
        pos = 0

        while True:
            self.draw_input_line(prompt, buf, pos)
            self.stdscr.refresh()

            try:
                ch = self.stdscr.getch()
            except KeyboardInterrupt:
                return None

            if ch in (curses.KEY_ENTER, 10, 13):
                return buf

            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if pos > 0:
                    buf = buf[:pos - 1] + buf[pos:]
                    pos -= 1

            elif ch == curses.KEY_DC:
                if pos < len(buf):
                    buf = buf[:pos] + buf[pos + 1:]

            elif ch == curses.KEY_LEFT:
                pos = max(0, pos - 1)

            elif ch == curses.KEY_RIGHT:
                pos = min(len(buf), pos + 1)

            elif ch in (curses.KEY_HOME, 1):
                pos = 0

            elif ch in (curses.KEY_END, 5):
                pos = len(buf)

            elif ch == 21:  # Ctrl+U
                buf = ""
                pos = 0

            elif ch == 9:  # Tab
                if buf.startswith("/"):
                    completions = [c for c in COMMAND_NAMES if c.startswith(buf)]
                    if len(completions) == 1:
                        buf = completions[0]
                        pos = len(buf)
                    elif completions:
                        self.add_line("  ".join(completions), C_INFO)

            elif ch == curses.KEY_PPAGE:
                max_scroll = max(0, len(self.lines) - self.content_h)
                self.scroll_offset = min(self.scroll_offset + self.content_h, max_scroll)
                self.refresh()

            elif ch == curses.KEY_NPAGE:
                self.scroll_offset = max(0, self.scroll_offset - self.content_h)
                self.refresh()

            elif ch == curses.KEY_RESIZE:
                self.refresh()

            elif ch == 3:  # Ctrl+C
                return None

            elif ch == 4:  # Ctrl+D
                if not buf:
                    return None

            elif ch == 27:  # ESC sequence
                self.stdscr.nodelay(True)
                self.stdscr.getch()
                self.stdscr.getch()
                self.stdscr.nodelay(False)

            elif 32 <= ch < 127:
                buf = buf[:pos] + chr(ch) + buf[pos:]
                pos += 1

    def get_approval(self, message):
        self.add_line(f"[Approval Required] {message}", C_QUESTION)
        while True:
            resp = self.get_input("Allow? (y/n) > ")
            if resp is None:
                return False
            resp = resp.strip().lower()
            if resp in ("y", "yes", ""):
                return True
            if resp in ("n", "no"):
                return False

    def get_question_response(self, question):
        self.add_line(f"[Question] {question}", C_QUESTION)
        resp = self.get_input("answer > ")
        return resp or ""

    # ================================================================
    #  Display helpers
    # ================================================================

    def show_tool_call(self, name, args):
        self.add_blank()
        self.add_line(f"--- tool: {name} ---", C_TOOL)
        for k, v in args.items():
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            self.add_line(f"  {k}: {s}", C_TOOL)

    def show_tool_result(self, name, result, is_error=False):
        c = C_ERROR if is_error else C_TOOL
        if len(result) > 2000:
            result = result[:1500] + "\n...(truncated)...\n" + result[-300:]
        self.add_line(f"--- result: {name} ---", c)
        self.add_line(result, c)
        self.add_blank()
        self.refresh()

    def show_info(self, msg):
        self.add_line(msg, C_INFO)

    def show_error(self, msg):
        self.add_line(f"Error: {msg}", C_ERROR)

    def show_user(self, msg):
        self.add_blank()
        self.add_line(f"> {msg}", C_USER)

    def show_memories(self, memories):
        if not memories:
            self.add_line("No memories found.", C_INFO)
            return
        for m in memories:
            tags = ", ".join(m.get("tags", []))
            hdr = f"[{m.get('type', 'general')}]"
            if tags:
                hdr += f" ({tags})"
            self.add_line(f"  #{m['id'][:8]} {hdr}: {m['content']}", C_MEMORY)

    def show_welcome(self):
        self.add_line("Welcome to localcoder!", C_AGENT)
        self.add_blank()
        self.add_line("Getting Started:", C_INFO)
        self.add_line('  Explore your project:  "What files are in this directory?"', C_INFO)
        self.add_line('  Read & explain code:   "Read main.py and explain it"', C_INFO)
        self.add_line('  Make changes:          "Add input validation to signup.py"', C_INFO)
        self.add_line('  Save knowledge:        "Remember that this project uses pytest"', C_INFO)
        self.add_blank()
        self.add_line("Tip: Type / and press Tab for command auto-completion.", C_INFO)
        self.add_line("     Page Up/Down to scroll output. Ctrl+C to interrupt.", C_INFO)
        self.add_blank()

    def show_help(self):
        self.add_blank()
        self.add_line("Commands:", C_INFO)
        for name, desc in COMMANDS.items():
            self.add_line(f"  {name:12s}  {desc}", C_INFO)
        self.add_blank()
        self.add_line("Tips:", C_INFO)
        self.add_line("  Type / then Tab to auto-complete commands", C_INFO)
        self.add_line("  Page Up / Page Down to scroll output", C_INFO)
        self.add_line("  Ctrl+C to interrupt the agent", C_INFO)
        self.add_blank()
