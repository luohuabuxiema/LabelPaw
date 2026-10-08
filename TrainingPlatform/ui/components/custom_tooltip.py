# -*- coding: utf-8 -*-
"""
Unified Custom Tooltip & Popover Component
Seamlessly delegates to MonkeyQt's official MkTooltip, MkTooltipPopover, and MkInfoIcon.
Provides native cross-theme support without redundant custom drawing.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget
from monkeyqt.components.feedback.tooltip import (
    MkTooltip, MkTooltipPopover, MkInfoIcon,
    create_field_header, create_input_field, create_switch_field
)

CustomTooltipPopover = MkTooltipPopover


class InfoIconButton(MkInfoIcon):
    """
    Backward-compatible subclass of MonkeyQt's MkInfoIcon.
    """
    def set_theme(self, is_dark: bool = False):
        self._update_appearance()


def attach_tooltip(widget: QWidget, text: str):
    """
    Attach MonkeyQt official speech-bubble tooltip to any widget.
    Automatically adapts to all light, dark, and specialized themes.
    """
    if not widget or not text:
        return
    MkTooltip.attach(widget, text)
    return widget
