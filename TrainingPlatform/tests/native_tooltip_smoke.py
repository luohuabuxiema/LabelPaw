"""Windows desktop check for stale translucent tooltip frames.

Run explicitly in an interactive Windows session:
  python -m TrainingPlatform.tests.native_tooltip_smoke --output-dir <folder>
Captures the desktop before any QWidget.grab() can force a fresh repaint.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget, QPushButton
from monkeyqt import ThemeEngine
from TrainingPlatform.ui.components.custom_tooltip import enable_component_tooltips, MkTooltipPopover, MkTooltip


def ink_pixels(image, dark_theme):
    count = 0
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            rgb = (color.red(), color.green(), color.blue())
            if (max(rgb) < 80) if dark_theme else (min(rgb) > 200):
                count += 1
    return count


def run(output_dir):
    app = QApplication.instance() or QApplication([])
    if app.platformName() != "windows":
        raise RuntimeError("This check requires Qt's Windows platform and a visible desktop")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window = QWidget()
    window.setWindowTitle("LabelPaw tooltip rendering check")
    window.resize(600, 320)
    window.move(300, 300)
    first = QPushButton("Model action", window)
    first.setGeometry(20, 130, 130, 30)
    first.setToolTip("移除模型记录")
    second = QPushButton("Title bar action", window)
    second.setGeometry(380, 30, 130, 30)
    second.setToolTip("还原")
    enable_component_tooltips(window)
    window.show()
    QTest.qWait(200)
    results = []
    try:
        ThemeEngine.enable_palette_theme(True)
        for dark, theme in ((False, "Elegant Light"), (True, "Dark Mode (OLED)")):
            ThemeEngine.set_theme(theme)
            baseline_ink = None
            for index, target in enumerate((first, second, first, second)):
                if index < 2:
                    app.sendEvent(target, QHelpEvent(QEvent.Type.ToolTip, QPoint(5, 5), target.mapToGlobal(QPoint(5, 5))))
                else:
                    # Component-library calls must share the same renderer.
                    MkTooltip.show(target, target.toolTip())
                QTest.qWait(180)
                popover = MkTooltipPopover.get_instance()
                assert popover.isVisible()
                assert popover.label.height() <= popover.label.fontMetrics().height() + 8, "Short hint unexpectedly wraps"
                point = popover.label.mapToGlobal(QPoint(0, 0))
                screen = popover.screen()
                native = screen.grabWindow(0, point.x(), point.y(), popover.label.width(), popover.label.height())
                ink = ink_pixels(native.toImage(), dark)
                assert ink > 0, "Desktop capture contains no tooltip text"
                rect = popover.frameGeometry()
                frame = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
                filename = output_dir / f"tooltip_{'dark' if dark else 'light'}_{index}.png"
                frame.save(str(filename))
                if index == 0:
                    baseline_ink = ink
                elif index % 2:
                    assert ink < baseline_ink * .65, "Desktop still shows the previous six-character hint instead of 还原"
                else:
                    assert .75 < ink / baseline_ink < 1.25, "Desktop did not restore the model action hint"
                results.append({"theme": theme, "text": target.toolTip(), "ink_pixels": ink,
                                "native_window": int(popover.winId()), "image": str(filename)})
                app.sendEvent(target, QEvent(QEvent.Type.Leave))
                QTest.qWait(40)
        return results
    finally:
        MkTooltipPopover.hide_popover()
        window.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(tempfile.gettempdir()) / "labelpaw-tooltip-qa"))
    print(json.dumps(run(parser.parse_args().output_dir), ensure_ascii=False, indent=2))
