# -*- coding: utf-8 -*-
"""
Main Entry Point for AI Vision Training Platform
"""
import sys
import os

# Add workspace directory to sys.path so modules and ultralytics can be imported smoothly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(CURRENT_DIR)

if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from PySide6.QtWidgets import QApplication
from monkeyqt import ThemeEngine, use_theme
from TrainingPlatform.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    ThemeEngine.enable_palette_theme()
    use_theme("默认")
    
    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
