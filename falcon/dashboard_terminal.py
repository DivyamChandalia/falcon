"""Bounded local PTYs for interactive kubectl exec, rendered as VT screens."""

from __future__ import annotations

import codecs
import errno
import os
import shlex
import signal
import struct
import subprocess
from typing import Sequence

import pyte
from rich.style import Style
from rich.text import Text
from textual import events
from textual.containers import ScrollableContainer
from textual.widgets import Static

from .kubernetes import debug_shell_rc

TERMINAL_SCROLLBACK_LINES = 200


def terminal_shell_rc(shell: str) -> str:
    """Debug startup with a compact prompt; the pane title identifies the Pod."""
    if shell == "zsh":
        prompt = """
function _falcon_prompt_prefix {
  PROMPT='%1/%% '
  RPROMPT=""
}
_falcon_prompt_prefix
"""
    else:
        prompt = r"""
__falcon_prompt_prefix() {
  PS1='\W% '
}
__falcon_prompt_prefix
"""
    return debug_shell_rc(shell) + prompt


class TerminalScreen(pyte.HistoryScreen):
    """Count discarded rows so a scrolled viewport keeps its output anchor."""

    def reset(self) -> None:
        self.history_rows_seen = 0
        super().reset()

    def index(self) -> None:
        bottom = self.margins.bottom if self.margins else self.lines - 1
        if self.cursor.y == bottom:
            self.history_rows_seen += 1
        super().index()


def exec_command(
    namespace: str,
    pod: str,
    container: str,
    shell: str,
    executable: str = "kubectl",
    *,
    rc_path: str = "",
    prompt_label: str = "falcon",
) -> list[str]:
    preferred = "zsh" if shell.rsplit("/", 1)[-1] == "zsh" else "bash"
    # Reuse debug-job startup; user paths/labels are argv, never shell source.
    script = """\
export TERM=linux CONDA_AUTO_ACTIVATE_BASE=false CONDA_CHANGEPS1=false
export FALCON_USER_RC="$2" FALCON_PROMPT_LABEL="$3"
for shell in "$1" bash zsh sh; do
  shell_path=$(command -v "$shell") || continue
  if [ "$shell" != "$1" ]; then
    printf 'Falcon terminal: %s unavailable; using %s\\n' "$1" "$shell"
  fi
  case "$shell" in
    zsh|bash)
      wrapper_dir=$(mktemp -d /tmp/falcon-terminal.XXXXXX) || exit 1
      if [ ! -r "$FALCON_USER_RC" ]; then
        export FALCON_USER_RC="$HOME/.$shell"rc
      fi
      ;;
  esac
  case "$shell" in
    zsh)
      printf '%s' ZSH_WRAPPER > "$wrapper_dir/.zshrc" || exit 1
      export ZDOTDIR="$wrapper_dir"
      exec "$shell_path" -i ;;
    bash)
      printf '%s' BASH_WRAPPER > "$wrapper_dir/.bashrc" || exit 1
      exec "$shell_path" --noprofile --rcfile "$wrapper_dir/.bashrc" -i ;;
    *) exec "$shell_path" -i ;;
  esac
done
echo "No shell available" >&2; exit 127
""".replace("ZSH_WRAPPER", shlex.quote(terminal_shell_rc("zsh"))).replace(
        "BASH_WRAPPER", shlex.quote(terminal_shell_rc("bash"))
    )
    return [
        executable,
        "exec",
        "--namespace",
        namespace,
        pod,
        "--container",
        container or "main",
        "--stdin",
        "--tty",
        "--",
        "/bin/sh",
        "-c",
        script,
        "falcon-terminal",
        preferred,
        rc_path,
        prompt_label,
    ]


class TerminalSession:
    def __init__(
        self, command: Sequence[str], *, width: int = 80, height: int = 24, label: str = "Terminal"
    ):
        import fcntl
        import pty
        import termios
        import tty

        self.label = label
        self.screen = TerminalScreen(
            max(1, width), max(1, height), history=TERMINAL_SCROLLBACK_LINES
        )
        self.stream = pyte.Stream(self.screen)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.revision = 0
        self.returncode: int | None = None
        self.closed = False
        self._input = bytearray()
        self.master, slave = pty.openpty()
        self.screen.write_process_input = self.send_text
        try:
            tty.setraw(slave)
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
            self.process = subprocess.Popen(
                list(command),
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                env={**os.environ, "TERM": "linux"},
            )
            os.set_blocking(self.master, False)
        except BaseException:
            os.close(self.master)
            raise
        finally:
            os.close(slave)

    def send_text(self, text: str) -> None:
        self.send(text.encode("utf-8"))

    def send(self, data: bytes) -> None:
        if self.closed or self.returncode is not None:
            return
        if len(self._input) + len(data) > 1024 * 1024:
            raise ValueError("terminal input exceeds 1 MiB buffer")
        self._input.extend(data)

    def poll(self) -> bool:
        if self.closed:
            return False
        changed = False
        if self._input:
            try:
                count = os.write(self.master, self._input)
                del self._input[:count]
            except BlockingIOError:
                pass
            except OSError:
                self._input.clear()
        # Bound each tick so a noisy shell cannot block Dashboard interaction.
        for _ in range(2):
            try:
                data = os.read(self.master, 4096)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            self.stream.feed(self.decoder.decode(data))
            changed = True
        code = self.process.poll()
        if code is not None and self.returncode is None:
            self.returncode = code
            changed = True
        if changed:
            self.revision += 1
        return changed

    def resize(self, width: int, height: int) -> None:
        import fcntl
        import termios

        width, height = max(1, width), max(1, height)
        if self.closed or (width, height) == (self.screen.columns, self.screen.lines):
            return
        if height < self.screen.lines:
            self.screen.history.top.extend(
                self.screen.buffer[y] for y in range(self.screen.lines - height)
            )
            self.screen.history_rows_seen += self.screen.lines - height
        self.screen.resize(lines=height, columns=width)
        self.screen.ensure_vbounds()
        self.screen.ensure_hbounds()
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGWINCH)
            except ProcessLookupError:
                pass
        self.revision += 1

    def render(self, *, cursor: bool = True, scrollback: bool = False) -> Text:
        result = Text(no_wrap=True, overflow="crop")
        colors = {"brown": "yellow", "brightblack": "bright_black"}
        styles = {}

        def color(value):
            if value == "default":
                return None
            if len(value) == 6 and all(c in "0123456789abcdef" for c in value.lower()):
                return "#" + value
            if value.startswith("bright"):
                return "bright_" + colors.get(value[6:], value[6:])
            return colors.get(value, value)

        history = list(self.screen.history.top) if scrollback else []
        rows = history + [self.screen.buffer[y] for y in range(self.screen.lines)]
        if scrollback:
            rows = rows[-max(TERMINAL_SCROLLBACK_LINES, self.screen.lines) :]
        first_screen_row = len(rows) - self.screen.lines
        for y, row in enumerate(rows):
            if y:
                result.append("\n")
            run = []
            previous_style = None
            for x in range(self.screen.columns):
                char = row.get(x, self.screen.default_char)
                is_cursor = (
                    cursor
                    and self.returncode is None
                    and (x, y - first_screen_row)
                    == (min(self.screen.columns - 1, self.screen.cursor.x), self.screen.cursor.y)
                )
                if is_cursor:
                    char = char._replace(fg="000000", bg="00d7ff", reverse=False)
                attributes = (
                    char.fg,
                    char.bg,
                    char.bold,
                    char.italics,
                    char.underscore,
                    char.reverse,
                    char.strikethrough,
                )
                style = styles.get(attributes)
                if style is None:
                    style = Style(
                        color=color(char.fg),
                        bgcolor=color(char.bg),
                        bold=char.bold,
                        italic=char.italics,
                        underline=char.underscore,
                        reverse=char.reverse,
                        strike=char.strikethrough,
                    )
                    styles[attributes] = style
                if run and style != previous_style:
                    result.append("".join(run), previous_style)
                    run = []
                run.append(char.data)
                previous_style = style
            result.append("".join(run), previous_style)
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=1)
            except ProcessLookupError:
                pass
        os.close(self.master)


def key_bytes(key: str, character: str | None = None) -> bytes:
    special = {
        "enter": b"\r",
        "tab": b"\t",
        "backspace": b"\x7f",
        "delete": b"\x1b[3~",
        "escape": b"\x1b",
        "up": b"\x1b[A",
        "down": b"\x1b[B",
        "home": b"\x1b[H",
        "end": b"\x1b[F",
        "pageup": b"\x1b[5~",
        "pagedown": b"\x1b[6~",
        "alt+left": b"\x1b[D",
        "alt+right": b"\x1b[C",
    }
    if key in special:
        return special[key]
    if key.startswith("ctrl+") and len(key) == 6:
        char = key[-1]
        if "a" <= char <= "z":
            return bytes([ord(char) - ord("a") + 1])
    if key.startswith("alt+") and len(key) == 5:
        return b"\x1b" + key[-1].encode()
    return character.encode("utf-8") if character else b""


class DashboardTerminal(ScrollableContainer, inherit_bindings=False):
    can_focus = True
    BINDINGS = []

    def compose(self):
        yield Static(id="selected-terminal-screen", markup=False)

    def update_screen(self, content: Text, *, source: tuple[int, int]) -> None:
        previous = getattr(self, "_source", None)
        follow = (
            self.scroll_y >= self.max_scroll_y - 0.01
            or previous is None
            or previous[0] != source[0]
        )
        position = self.scroll_y
        removed = max(0, source[1] - previous[1]) if previous and previous[0] == source[0] else 0
        self._source = source
        self.query_one("#selected-terminal-screen", Static).update(content)
        if follow:
            self.call_after_refresh(self._follow_output, position)
        elif removed:
            self.call_after_refresh(
                self.scroll_to, y=max(0, position - removed), animate=False, force=True
            )

    def _follow_output(self, previous_position: float) -> None:
        # A wheel/drag may have moved the viewport since output was queued.
        if abs(self.scroll_y - previous_position) < 0.01:
            self.scroll_end(animate=False)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.app.set_focus(self, scroll_visible=False)
        self.app.selected_section_focused(self.id or "")
        event.stop()

    def on_click(self, event: events.Click) -> None:
        self.app.set_focus(self, scroll_visible=False)
        event.stop()

    def on_focus(self) -> None:
        if self.screen.focused is self:
            self.app.selected_section_focused(self.id or "")

    def selection_for_copy(self) -> str | None:
        content = self.query_one("#selected-terminal-screen", Static)
        selection = content.text_selection
        if selection is None:
            return None
        selected = content.get_selection(selection)
        return selected[0] if selected and selected[0] else None

    def on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+c" and self.app._copy_terminal_selection():
            pass
        elif event.key in {"left", "right"}:
            self.app._move_selected_window(-1 if event.key == "left" else 1)
        else:
            self.screen.clear_selection()
            self.scroll_end(animate=False)
            self.app._send_terminal_input(key_bytes(event.key, event.character))
        event.prevent_default()
        event.stop()

    def on_paste(self, event: events.Paste) -> None:
        session = self.app._active_terminal()
        if session is not None:
            self.screen.clear_selection()
            text = event.text.replace("\r\n", "\n").replace("\n", "\r")
            if 2004 << 5 in session.screen.mode:
                text = "\x1b[200~" + text + "\x1b[201~"
            self.app._send_terminal_input(text.encode("utf-8"))
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_relative(y=3, animate=False)
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_relative(y=-3, animate=False)
        event.stop()
