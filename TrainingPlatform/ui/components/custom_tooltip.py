# -*- coding: utf-8 -*-
"""
Unified Custom Tooltip & Popover Component
Seamlessly delegates to MonkeyQt's official MkTooltip, MkTooltipPopover, and MkInfoIcon.
Provides native cross-theme support without redundant custom drawing.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid
from monkeyqt.components.feedback.tooltip import (
    MkTooltip, MkTooltipPopover, MkInfoIcon,
    create_field_header, create_input_field, create_switch_field
)

class CustomTooltipPopover(MkTooltipPopover):
    """Use MonkeyQt's drawing with a fresh Windows layered-window surface.

    Qt 6.4 can retain the first composited frame of a reused translucent tooltip:
    QLabel.text() changes while the desktop still displays the previous text.
    Recreating only the native surface preserves the component and its theme.
    """
    @classmethod
    def get_instance(cls):
        instance = MkTooltipPopover._instance
        if instance is None or not isValid(instance) or not isinstance(instance, cls):
            if instance is not None and isValid(instance):
                MkTooltipPopover.hide_popover()
                instance.deleteLater()
            MkTooltipPopover._instance = cls()
        return MkTooltipPopover._instance

    def set_content(self, text, theme_override=None):
        self._raw_hint = text
        super().set_content(text, theme_override=theme_override)
        self.update()

    def on_theme_changed(self, theme_name=""):
        if self.isVisible() and getattr(self, "_raw_hint", ""):
            target = self._target_widget
            if target is not None and isValid(target) and target.isVisible():
                self.show_popover(target, self._raw_hint)


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


class _ComponentTooltipFilter(QObject):
    """Keep the shared MonkeyQt bubble bound to the current hover target."""
    _EVENTS = frozenset((
        QEvent.Type.ApplicationDeactivate, QEvent.Type.ToolTip, QEvent.Type.Enter,
        QEvent.Type.Leave, QEvent.Type.ToolTipChange, QEvent.Type.Hide,
        QEvent.Type.Close, QEvent.Type.Destroy, QEvent.Type.WindowDeactivate,
        QEvent.Type.MouseButtonPress, QEvent.Type.Wheel,
    ))

    def __init__(self, parent):
        super().__init__(parent)
        self.active_target = None

    @staticmethod
    def _in_scope(widget):
        while widget is not None and isValid(widget):
            if widget.property("platform_tooltips"):
                return True
            widget = widget.parentWidget()
        return False

    @staticmethod
    def _hint(widget):
        # Qt inherits tooltips from parents. Attached MonkeyQt hints deliberately
        # clear QWidget.toolTip(), so resolve their text as well.
        while widget is not None and isValid(widget):
            attached = MkTooltip._FILTERS.get(widget)
            text = widget.toolTip() or (attached.hint_text if attached else "")
            if not text and isinstance(widget, MkInfoIcon):
                text = widget.hint_text
            if text:
                return widget, text
            widget = widget.parentWidget()
        return None, ""

    @staticmethod
    def _popover_target(popover, name="_target_widget"):
        target = getattr(popover, name, None)
        return target if target is not None and isValid(target) else None

    def _dismiss(self):
        MkTooltipPopover.hide_popover()
        if MkTooltipPopover._instance is not None:
            MkTooltipPopover._instance._target_widget = None
        self.active_target = None

    def _show(self, target, text):
        popover = MkTooltipPopover._instance
        if (popover is not None and isValid(popover) and popover.isVisible() and
            self._popover_target(popover) is target and
            getattr(popover, "_raw_hint", None) == text):
            return
        # A delayed request for a previous anchor must not overwrite this hint.
        MkTooltipPopover.cancel_open()
        self.active_target = target
        MkTooltipPopover.show_popover(target, text)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind not in self._EVENTS:
            return False
        popover = MkTooltipPopover._instance
        if kind == QEvent.Type.ApplicationDeactivate:
            self._dismiss()
            return False
        if not isinstance(watched, QWidget):
            return False
        if watched is popover:
            if kind == QEvent.Type.Leave:
                self._dismiss()
            return False
        if not self._in_scope(watched):
            return False

        target = self._popover_target(popover)
        pending = self._popover_target(popover, "_pending_target")
        if kind == QEvent.Type.ToolTip:
            hint_target, text = self._hint(watched)
            if text:
                if (popover is not None and isValid(popover) and popover.isVisible() and
                    self._popover_target(popover) is hint_target and
                    getattr(popover, "_raw_hint", None) == text):
                    event.accept()
                    return True
                self._show(hint_target, text)
            else:
                self._dismiss()
            event.accept()
            return True
        if kind == QEvent.Type.Enter:
            hint_target, text = self._hint(watched)
            if pending is not None and pending is not hint_target:
                MkTooltipPopover.cancel_open()
            if popover is not None and popover.isVisible():
                if text:
                    self._show(hint_target, text)
                else:
                    self._dismiss()
        elif kind == QEvent.Type.Leave and watched in (target, pending):
            # Passive hints are mouse-transparent: prevent false dismissal if cursor is still inside watched widget
            try:
                from PySide6.QtGui import QCursor
                if watched.rect().contains(watched.mapFromGlobal(QCursor.pos())):
                    return False
            except Exception:
                pass
            if popover is not None and not popover.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents):
                MkTooltipPopover.schedule_close(180)
            else:
                self._dismiss()
        elif kind == QEvent.Type.ToolTipChange and watched is target:
            hint_target, text = self._hint(watched)
            if text:
                self._show(hint_target, text)
            else:
                self._dismiss()
        elif kind in (QEvent.Type.Hide, QEvent.Type.Close, QEvent.Type.Destroy, QEvent.Type.WindowDeactivate):
            if any(anchor is not None and (watched is anchor or watched.isAncestorOf(anchor)) for anchor in (target, pending)):
                self._dismiss()
        elif kind in (QEvent.Type.MouseButtonPress, QEvent.Type.Wheel) and not isinstance(watched, MkInfoIcon):
            self._dismiss()
        return False


def enable_component_tooltips(root):
    root.setProperty("platform_tooltips", True)
    app = QApplication.instance()
    if app is not None:
        # Library-native MkTooltip, MkInfoIcon and table cells use this same
        # singleton, so the rendering fix also covers their direct calls.
        CustomTooltipPopover.get_instance()
        if not hasattr(app, "_platform_tooltip_filter"):
            app._platform_tooltip_filter = _ComponentTooltipFilter(app)
            app.installEventFilter(app._platform_tooltip_filter)
