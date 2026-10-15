# -*- coding: utf-8 -*-
"""
Terminal Formatter Utility for MkConsole
Provides real-time terminal syntax highlighting, progress bar replacement,
Consolas monospace typography, and custom scrollbar theming.
Optimized for high-contrast dark and light themes (Catppuccin / VSCode / iTerm2).
Ultra-high performance: O(1) last line cache, batched updates, debounced badge refreshes.
"""
import re
from typing import Optional, List, Tuple
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTextEdit, QScrollBar
from PySide6.QtGui import QFont, QColor, QTextCursor, QTextCharFormat
from monkeyqt import ThemeEngine, MkConsole


def _detect_is_dark(console=None) -> bool:
    """Accurately determine if console is rendering in dark mode."""
    try:
        if ThemeEngine.is_dark():
            return True
    except Exception:
        pass

    if console:
        win = console.window()
        if win:
            if hasattr(win, "is_dark_mode") and win.is_dark_mode is not None:
                return bool(win.is_dark_mode)
            if hasattr(win, "is_dark") and win.is_dark is not None:
                return bool(win.is_dark)
        try:
            pal_col = console.palette().window().color()
            if pal_col.lightness() < 128:
                return True
        except Exception:
            pass

    return False


def apply_terminal_console_style(console: MkConsole, is_dark: Optional[bool] = None):
    """Ensure console is synced with MonkeyQt native styling and theme adaptation."""
    if not console:
        return
    if hasattr(console, "apply_theme_colors"):
        console.apply_theme_colors()


def _get_line_style(text: str, clean_text: str, is_dark: bool) -> Tuple[str, bool, Optional[re.Match], bool, Optional[str]]:
    """Determine font color, boldness, progress flags, and badge counter category."""
    m_new_epoch = re.match(r'^\s*(\d+/\d+)\s+', text)
    is_val_progress = ('Class' in text and 'Images' in text and 'Instances' in text and 'Box(P' in text)
    badge_key = None

    if any(err in text for err in ("[ERROR]", "[Runner] 训练异常", "Traceback (most recent call last):", "Error:", "Exception:", "[FAILED]")):
        fg_color = "#f87171" if is_dark else "#dc2626"
        is_bold = True
        badge_key = "error"
    elif any(succ in text for succ in ("[SUCCESS]", "训练完成并成功保存", "YOLO 模型训练成功", "weights/best.pt")):
        fg_color = "#4ade80" if is_dark else "#16a34a"
        is_bold = True
        badge_key = "success"
    elif any(warn in text for warn in ("[WARN]", "[WARNING]", "Warning:", "UserWarning")):
        fg_color = "#fde047" if is_dark else "#b45309"
        is_bold = False
        badge_key = "warning"
    elif "Epoch" in text and "GPU_mem" in text:
        fg_color = "#38bdf8" if is_dark else "#1d4ed8"
        is_bold = True
        badge_key = "info"
    elif "Class" in text and "Images" in text and "Instances" in text:
        fg_color = "#67e8f9" if is_dark else "#0284c7"
        is_bold = True
        badge_key = "info"
    elif clean_text.startswith("all ") or re.match(r'^\s*all\s+\d+', text):
        fg_color = "#34d399" if is_dark else "#047857"
        is_bold = True
        badge_key = "info"
    elif m_new_epoch:
        fg_color = "#f8fafc" if is_dark else "#0f172a"
        is_bold = False
        badge_key = "info"
    elif "from  n    params" in text or "YOLO26n-seg summary:" in text or "layers," in text:
        fg_color = "#60a5fa" if is_dark else "#2563eb"
        is_bold = True
        badge_key = "info"
    elif clean_text.startswith("Ultralytics") or clean_text.startswith("engine\\trainer:") or clean_text.startswith("Overriding"):
        fg_color = "#c084fc" if is_dark else "#7c3aed"
        is_bold = False
        badge_key = "info"
    else:
        fg_color = "#f1f5f9" if is_dark else "#0f172a"
        is_bold = False
        badge_key = "info"

    return fg_color, is_bold, m_new_epoch, is_val_progress, badge_key


def _schedule_badge_update(console: MkConsole):
    """Debounce badge updates so high-speed log streams never freeze the GUI thread."""
    if not hasattr(console, "_update_badges"):
        return
    if getattr(console, "_badge_update_scheduled", False):
        return
    console._badge_update_scheduled = True

    def _do_update():
        console._badge_update_scheduled = False
        if hasattr(console, "_update_badges"):
            console._update_badges()

    QTimer.singleShot(150, _do_update)


def append_terminal_line_to_console(console: MkConsole, text: str, is_dark: Optional[bool] = None):
    """Append raw terminal line preserving exact column spacing, alignment, and semantic high-contrast colors."""
    if not console or not hasattr(console, "text_edit") or not console.text_edit:
        return

    clean_text = text.strip()
    if clean_text == "__TRAIN_SUCCESS__":
        return

    if is_dark is None:
        is_dark = _detect_is_dark(console)

    if not clean_text:
        # Avoid full toPlainText() call
        cursor = console.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n")
        console._last_line_cache = ""
        return

    fg_color, is_bold, m_new_epoch, is_val_progress, badge_key = _get_line_style(text, clean_text, is_dark)

    # O(1) last line check using cached string
    last_line = getattr(console, "_last_line_cache", "")
    is_replace = False
    if m_new_epoch:
        new_ep = m_new_epoch.group(1)
        m_last = re.match(r'^\s*(\d+/\d+)\s+', last_line)
        if m_last and m_last.group(1) == new_ep:
            is_replace = True
    elif is_val_progress:
        if 'Class' in last_line and 'Images' in last_line and 'Instances' in last_line and 'Box(P' in last_line:
            is_replace = True

    if badge_key and hasattr(console, "_counts") and not is_replace:
        console._counts[badge_key] = console._counts.get(badge_key, 0) + 1
        _schedule_badge_update(console)

    cursor = console.text_edit.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)

    fmt = QTextCharFormat()
    fmt.setFont(QFont("Consolas", 9, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
    fmt.setForeground(QColor(fg_color))

    if is_replace:
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
    else:
        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")

    console._last_line_cache = text
    console.text_edit.setTextCursor(cursor)

    vbar = console.text_edit.verticalScrollBar()
    if vbar:
        vbar.setValue(vbar.maximum())


def append_terminal_lines_batch(console: MkConsole, lines: List[str], is_dark: Optional[bool] = None):
    """
    Ultra-fast batch insertion of multiple terminal lines (e.g. initial 1500 log load).
    Uses beginEditBlock and pauses widget updates to prevent GUI thread freeze.
    """
    if not console or not hasattr(console, "text_edit") or not console.text_edit or not lines:
        return

    if is_dark is None:
        is_dark = _detect_is_dark(console)

    te = console.text_edit
    te.setUpdatesEnabled(False)
    cursor = te.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.beginEditBlock()

    counts = getattr(console, "_counts", {"info": 0, "success": 0, "warning": 0, "error": 0})
    last_line = getattr(console, "_last_line_cache", "")

    fmt = QTextCharFormat()
    fmt.setFont(QFont("Consolas", 9, QFont.Weight.Normal))

    try:
        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            clean_text = line.strip()
            if not clean_text or clean_text == "__TRAIN_SUCCESS__":
                continue

            fg_color, is_bold, m_new_epoch, is_val_progress, badge_key = _get_line_style(line, clean_text, is_dark)

            is_replace = False
            if m_new_epoch:
                new_ep = m_new_epoch.group(1)
                m_last = re.match(r'^\s*(\d+/\d+)\s+', last_line)
                if m_last and m_last.group(1) == new_ep:
                    is_replace = True
            elif is_val_progress:
                if 'Class' in last_line and 'Images' in last_line and 'Instances' in last_line and 'Box(P' in last_line:
                    is_replace = True

            if badge_key and not is_replace:
                counts[badge_key] = counts.get(badge_key, 0) + 1

            fmt.setFontWeight(QFont.Weight.Bold if is_bold else QFont.Weight.Normal)
            fmt.setForeground(QColor(fg_color))

            if is_replace:
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.setCharFormat(fmt)
                cursor.insertText(line)
            else:
                cursor.setCharFormat(fmt)
                cursor.insertText(line + "\n")

            last_line = line

        console._last_line_cache = last_line
    finally:
        cursor.endEditBlock()
        te.setTextCursor(cursor)
        te.setUpdatesEnabled(True)

    if hasattr(console, "_counts"):
        console._counts = counts
    if hasattr(console, "_update_badges"):
        console._update_badges()

    vbar = te.verticalScrollBar()
    if vbar:
        vbar.setValue(vbar.maximum())
    hbar = te.horizontalScrollBar()
    if hbar:
        hbar.setValue(0)
