# -*- coding: utf-8 -*-
"""
Terminal Formatter Utility for MkConsole
Provides real-time terminal syntax highlighting, progress bar replacement,
Consolas monospace typography, and custom scrollbar theming.
Optimized for high-contrast dark and light themes (Catppuccin / VSCode / iTerm2).
"""
import re
from typing import Optional
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
        content = console.text_edit.toPlainText()
        if content and not content.endswith("\n\n"):
            cursor = console.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n")
        return

    # Check if incoming line is an in-progress update of the SAME epoch (e.g. 28/30 0% -> 50% -> 100%)
    m_new_epoch = re.match(r'^\s*(\d+/\d+)\s+', text)
    is_val_progress = ('Class' in text and 'Images' in text and 'Instances' in text and 'Box(P' in text)

    full_content = console.text_edit.toPlainText().rstrip('\r\n')
    last_line = full_content.splitlines()[-1] if full_content.splitlines() else ''

    is_replace = False
    if m_new_epoch:
        new_ep = m_new_epoch.group(1)
        m_last = re.match(r'^\s*(\d+/\d+)\s+', last_line)
        if m_last and m_last.group(1) == new_ep:
            is_replace = True
    elif is_val_progress:
        if 'Class' in last_line and 'Images' in last_line and 'Instances' in last_line and 'Box(P' in last_line:
            is_replace = True

    # ── High-contrast syntax colorization ──
    is_bold = False
    if any(err in text for err in ("[ERROR]", "[Runner] 训练异常", "Traceback (most recent call last):", "Error:", "Exception:", "[FAILED]")):
        fg_color = "#f87171" if is_dark else "#dc2626"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["error"] = console._counts.get("error", 0) + 1
    elif any(succ in text for succ in ("[SUCCESS]", "训练完成并成功保存", "YOLO 模型训练成功", "weights/best.pt")):
        fg_color = "#4ade80" if is_dark else "#16a34a"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["success"] = console._counts.get("success", 0) + 1
    elif any(warn in text for warn in ("[WARN]", "[WARNING]", "Warning:", "UserWarning")):
        fg_color = "#fde047" if is_dark else "#b45309"
        is_bold = False
        if hasattr(console, "_counts") and not is_replace:
            console._counts["warning"] = console._counts.get("warning", 0) + 1
    elif "Epoch" in text and "GPU_mem" in text:
        # Epoch table header
        fg_color = "#38bdf8" if is_dark else "#1d4ed8"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    elif "Class" in text and "Images" in text and "Instances" in text:
        # Validation table header
        fg_color = "#67e8f9" if is_dark else "#0284c7"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    elif clean_text.startswith("all ") or re.match(r'^\s*all\s+\d+', text):
        # Validation metrics row (e.g. all  4  4  0.777  1  0.829  0.595) - highlight in bright emerald green
        fg_color = "#34d399" if is_dark else "#047857"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    elif m_new_epoch:
        # Training epoch progress row (e.g. 84/100 2.85G 0.7491 ...) - bright crisp white in dark mode
        fg_color = "#f8fafc" if is_dark else "#0f172a"
        is_bold = False
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    elif "from  n    params" in text or "YOLO26n-seg summary:" in text or "layers," in text:
        fg_color = "#60a5fa" if is_dark else "#2563eb"
        is_bold = True
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    elif clean_text.startswith("Ultralytics") or clean_text.startswith("engine\\trainer:") or clean_text.startswith("Overriding"):
        fg_color = "#c084fc" if is_dark else "#7c3aed"
        is_bold = False
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1
    else:
        # Default terminal output - bright crisp off-white in dark mode, dark slate in light mode
        fg_color = "#f1f5f9" if is_dark else "#0f172a"
        is_bold = False
        if hasattr(console, "_counts") and not is_replace:
            console._counts["info"] = console._counts.get("info", 0) + 1

    if hasattr(console, "_update_badges"):
        console._update_badges()

    cursor = console.text_edit.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)

    fmt = QTextCharFormat()
    fmt.setFont(QFont("Consolas", 9, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
    fmt.setForeground(QColor(fg_color))

    if is_replace:
        if console.text_edit.toPlainText().endswith('\n'):
            cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.MoveAnchor, 1)
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
    else:
        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")

    console.text_edit.setTextCursor(cursor)

    vbar = console.text_edit.verticalScrollBar()
    vbar.setValue(vbar.maximum())
    hbar = console.text_edit.horizontalScrollBar()
    hbar.setValue(0)
