# -*- coding: utf-8 -*-
"""
Reusable Python Virtual Environment Selector & Live Console Probe Component
Features:
- SegmentedPillToggle: Pill-shaped segmented control inspired by Ultralytics HUB (Fig 2)
- EnvironmentSelectorWidget: Dropdown + Refresh + Browse + Status Badges + Embedded Terminal Console
- Reusable across Training, Inference, and Model Export dialogs
"""
import os
import sys
import re
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QPlainTextEdit, QFrame, QToolButton, QApplication
)
from PySide6.QtCore import Qt, Signal, QSize, QRectF, QEvent
from PySide6.QtGui import QFont, QColor, QPainter, QCursor, QPainterPath, QRegion

from monkeyqt import MkComboBox
from monkeyqt_icons import (
    PhArrowsClockwise, PhFolderOpen, PhPlay, PhCopy, PhTrash,
    PhDesktop, PhSliders, PhCheckCircle, PhWarningCircle, PhCpu
)

from TrainingPlatform.core.env_manager import EnvironmentManager, format_gpu_model_name


class RoundedComboBox(MkComboBox):
    """
    Enhanced MkComboBox with smooth rounded corners, solid opaque background,
    and soft drop shadow for the popup card menu on Windows OS.
    Directly inherits from the upgraded MonkeyQt MkComboBox.
    """
    def __init__(self, parent=None, is_dark: bool = False, popup_radius: int = 10):
        self._is_dark = is_dark
        self._popup_radius = popup_radius
        super().__init__(parent)
        if is_dark:
            self.set_dark_mode(True)

    def set_dark_mode(self, is_dark: bool):
        self._is_dark = is_dark
        if hasattr(self, "_popup_widget") and self._popup_widget is not None:
            self._popup_widget.update_theme_style()



class SegmentedPillToggle(QFrame):
    """
    Pill-shaped segmented tab control (referencing Fig 2):
    - Rounded pill capsule container (radius 9999px)
    - Active tab: solid elevated card with soft border/shadow
    - Inactive tab: clean muted text with hover transition
    """
    tab_changed = Signal(int)

    def __init__(self, tabs=None, parent=None, is_dark=False):
        super().__init__(parent)
        self._tabs_data = tabs or []  # List of tuples: (icon_class, title_text)
        self._current_index = 0
        self._is_dark = is_dark
        self._buttons = []
        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        for idx, (icon_cls, text) in enumerate(self._tabs_data):
            btn = QPushButton(text, self)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            if icon_cls:
                btn.setIconSize(QSize(14, 14))
            btn.clicked.connect(lambda checked=False, i=idx: self.set_current_index(i))
            self._buttons.append(btn)
            layout.addWidget(btn)

        self._apply_style()

    def set_current_index(self, index: int):
        if 0 <= index < len(self._buttons) and index != self._current_index:
            self._current_index = index
            self._apply_style()
            self.tab_changed.emit(index)

    def current_index(self) -> int:
        return self._current_index

    def set_dark_mode(self, is_dark: bool):
        if self._is_dark != is_dark:
            self._is_dark = is_dark
            self._apply_style()

    def _apply_style(self):
        if self._is_dark:
            self.setStyleSheet("""
                QFrame {
                    background-color: #18181b;
                    border: 1px solid #27272a;
                    border-radius: 17px;
                }
            """)
            for idx, btn in enumerate(self._buttons):
                is_active = (idx == self._current_index)
                icon_cls, _ = self._tabs_data[idx]
                if is_active:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #27272a;
                            color: #f8fafc;
                            border: 1px solid #3f3f46;
                            border-radius: 14px;
                            font-size: 12px;
                            font-weight: 600;
                            padding: 0px 14px;
                        }
                    """)
                    if icon_cls:
                        btn.setIcon(icon_cls.regular(size=14, color="#38bdf8"))
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            color: #94a3b8;
                            border: none;
                            border-radius: 14px;
                            font-size: 12px;
                            font-weight: 500;
                            padding: 0px 14px;
                        }
                        QPushButton:hover {
                            color: #e2e8f0;
                            background-color: rgba(255, 255, 255, 0.05);
                        }
                    """)
                    if icon_cls:
                        btn.setIcon(icon_cls.regular(size=14, color="#94a3b8"))
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #f1f5f9;
                    border: 1px solid #e2e8f0;
                    border-radius: 17px;
                }
            """)
            for idx, btn in enumerate(self._buttons):
                is_active = (idx == self._current_index)
                icon_cls, _ = self._tabs_data[idx]
                if is_active:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #ffffff;
                            color: #0f172a;
                            border: 1px solid #e2e8f0;
                            border-radius: 14px;
                            font-size: 12px;
                            font-weight: 600;
                            padding: 0px 14px;
                        }
                    """)
                    if icon_cls:
                        btn.setIcon(icon_cls.regular(size=14, color="#0284c7"))
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            color: #64748b;
                            border: none;
                            border-radius: 14px;
                            font-size: 12px;
                            font-weight: 500;
                            padding: 0px 14px;
                        }
                        QPushButton:hover {
                            color: #334155;
                            background-color: rgba(0, 0, 0, 0.04);
                        }
                    """)
                    if icon_cls:
                        btn.setIcon(icon_cls.regular(size=14, color="#64748b"))


class EnvironmentConsoleViewer(QFrame):
    """
    Dedicated embedded terminal console for live PyTorch / GPU probe output:
    - Displays real-time streaming test output
    - Monospaced Consolas typography
    - Floating copy and clear buttons
    """
    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self._is_dark = is_dark
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("EnvConsoleCard")
        self.setFixedHeight(115)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 42, 6)
        layout.setSpacing(0)

        self.editor = QPlainTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.editor.setFrameShape(QFrame.Shape.NoFrame)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.editor)

        # Floating Copy Button
        self.btn_copy = QToolButton(self)
        self.btn_copy.setFixedSize(22, 22)
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setToolTip("复制控制台日志")
        self.btn_copy.clicked.connect(self._copy_content)

        # Floating Clear Button
        self.btn_clear = QToolButton(self)
        self.btn_clear.setFixedSize(22, 22)
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setToolTip("清空控制台日志")
        self.btn_clear.clicked.connect(self.clear_log)

        self._apply_style()

    def append_log(self, text: str):
        # Filter ANSI escape sequences (e.g., \x1b[0m, \x1b[32m)
        cleaned = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        if not cleaned.strip() and not text.strip():
            return
        self.editor.appendPlainText(cleaned)
        self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())

    def clear_log(self):
        self.editor.clear()

    def _copy_content(self):
        text = self.editor.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        self.btn_copy.move(w - 28, 6)
        self.btn_clear.move(w - 28, 32)

    def set_dark_mode(self, is_dark: bool):
        if self._is_dark != is_dark:
            self._is_dark = is_dark
            self._apply_style()

    def _apply_style(self):
        if self._is_dark:
            self.setStyleSheet("""
                QFrame#EnvConsoleCard {
                    background-color: #09090b;
                    border: 1px solid #27272a;
                    border-radius: 6px;
                }
            """)
            self.editor.setStyleSheet("""
                QPlainTextEdit {
                    background: transparent;
                    color: #e2e8f0;
                    border: none;
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    width: 5px;
                    height: 5px;
                }
                QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                    background: #3f3f46;
                    border-radius: 2px;
                }
            """)
            btn_qss = """
                QToolButton {
                    border: 1px solid #3f3f46;
                    background-color: #18181b;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #27272a;
                }
            """
            self.btn_copy.setStyleSheet(btn_qss)
            self.btn_clear.setStyleSheet(btn_qss)
            self.btn_copy.setIcon(PhCopy.regular(size=12, color="#a1a1aa"))
            self.btn_clear.setIcon(PhTrash.regular(size=12, color="#a1a1aa"))
        else:
            self.setStyleSheet("""
                QFrame#EnvConsoleCard {
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 6px;
                }
            """)
            self.editor.setStyleSheet("""
                QPlainTextEdit {
                    background: transparent;
                    color: #0f172a;
                    border: none;
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    width: 5px;
                    height: 5px;
                }
                QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                    background: #cbd5e1;
                    border-radius: 2px;
                }
            """)
            btn_qss = """
                QToolButton {
                    border: 1px solid #cbd5e1;
                    background-color: #ffffff;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #f1f5f9;
                }
            """
            self.btn_copy.setStyleSheet(btn_qss)
            self.btn_clear.setStyleSheet(btn_qss)
            self.btn_copy.setIcon(PhCopy.regular(size=12, color="#64748b"))
            self.btn_clear.setIcon(PhTrash.regular(size=12, color="#64748b"))


class StatusBadge(QLabel):
    """Sleek colored status pill badge."""
    def __init__(self, text: str = "", badge_type: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.badge_type = badge_type
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(22)
        self.update_style(badge_type, text)

    def update_style(self, badge_type: str, text: str = ""):
        self.badge_type = badge_type
        if text:
            self.setText(text)

        styles = {
            "success": "background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0;",
            "warning": "background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a;",
            "danger": "background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca;",
            "info": "background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;",
            "neutral": "background-color: #f1f5f9; color: #475569; border: 1px solid #e2e8f0;"
        }
        style_str = styles.get(badge_type, styles["neutral"])
        self.setStyleSheet(f"""
            QLabel {{
                {style_str}
                border-radius: 11px;
                font-size: 11px;
                font-weight: 600;
                padding: 1px 9px;
            }}
        """)


class EnvironmentSelectorWidget(QWidget):
    """
    Reusable environment selection & validation widget:
    - Environment selection combobox with auto-detected Conda / venv / system / custom envs
    - Rescan, Browse, and Run Test buttons
    - Status badges (PyTorch version, GPU/CUDA available, Device model, Ultralytics source)
    - Live streaming console for probe testing
    """
    env_changed = Signal(dict)

    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self._is_dark = is_dark
        self.env_mgr = EnvironmentManager.get_instance()
        self._current_probe_info = {}
        self._init_ui()
        self.refresh_environments()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # ── Top Row: Combo + Action Buttons ──
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        lbl_env = QLabel("执行环境:")
        lbl_env.setStyleSheet("font-size: 12px; font-weight: 600; color: #64748b; background: transparent; border: none;")
        top_row.addWidget(lbl_env)

        self.combo_env = RoundedComboBox(self, is_dark=self._is_dark, popup_radius=10)
        self.combo_env.setFixedHeight(30)
        self.combo_env.setMinimumWidth(260)
        self.combo_env.currentIndexChanged.connect(self._on_combo_index_changed)
        top_row.addWidget(self.combo_env, stretch=1)

        # Refresh button
        self.btn_refresh = QPushButton("刷新", self)
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.setFixedHeight(30)
        self.btn_refresh.setIcon(PhArrowsClockwise.regular(size=13, color="#64748b"))
        self.btn_refresh.setToolTip("重新扫描本机 Conda 与虚拟环境")
        self.btn_refresh.clicked.connect(self.refresh_environments)
        top_row.addWidget(self.btn_refresh)

        # Browse button
        self.btn_browse = QPushButton("浏览...", self)
        self.btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse.setFixedHeight(30)
        self.btn_browse.setIcon(PhFolderOpen.regular(size=13, color="#64748b"))
        self.btn_browse.setToolTip("选择外部虚拟环境中的 python.exe")
        self.btn_browse.clicked.connect(self._on_browse_python)
        top_row.addWidget(self.btn_browse)

        # Test button
        self.btn_test = QPushButton("测试环境", self)
        self.btn_test.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_test.setFixedHeight(30)
        self.btn_test.setIcon(PhPlay.regular(size=13, color="#0284c7"))
        self.btn_test.setToolTip("运行 GPU/CUDA 探测代码测试当前选中的环境")
        self.btn_test.clicked.connect(self.run_probe_test)
        top_row.addWidget(self.btn_test)

        main_layout.addLayout(top_row)

        # ── Middle Row: Status Badges ──
        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)

        self.badge_torch = StatusBadge("PyTorch: 待检测", "neutral")
        self.badge_gpu = StatusBadge("GPU: 待检测", "neutral")
        self.badge_cuda = StatusBadge("CUDA: --", "neutral")
        self.badge_ultra = StatusBadge("Ultralytics: --", "neutral")

        badge_row.addWidget(self.badge_torch)
        badge_row.addWidget(self.badge_gpu)
        badge_row.addWidget(self.badge_cuda)
        badge_row.addWidget(self.badge_ultra)
        badge_row.addStretch()

        main_layout.addLayout(badge_row)

        # ── Bottom Row: Console Viewer ──
        self.console_viewer = EnvironmentConsoleViewer(self, is_dark=self._is_dark)
        main_layout.addWidget(self.console_viewer)

        self._apply_button_styles()

    def refresh_environments(self):
        """Rescan all python environments and populate combobox."""
        envs = self.env_mgr.scan_environments()
        selected_path = self.env_mgr.get_selected_python()

        self.combo_env.blockSignals(True)
        self.combo_env.clear()

        select_idx = 0
        for idx, env in enumerate(envs):
            name = env["name"]
            path = env["path"]
            # Category prefix tag
            tag = " [当前]" if env.get("is_current") else ""
            display_text = f"{name}{tag} ({path})"
            self.combo_env.addItem(display_text, path)

            if os.path.normpath(path).lower() == os.path.normpath(selected_path).lower():
                select_idx = idx

        self.combo_env.setCurrentIndex(select_idx)
        self.combo_env.blockSignals(False)

        # Run probe on initially selected item
        self.run_probe_test()

    def _on_combo_index_changed(self, index: int):
        py_path = self.combo_env.currentData()
        if py_path and os.path.isfile(py_path):
            self.env_mgr.set_selected_python(py_path)
            self.run_probe_test()

    def _on_browse_python(self):
        filt = "Python 解释器 (python.exe);;所有文件 (*.*)" if os.name == 'nt' else "Python 解释器 (python*);;所有文件 (*)"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Python 虚拟环境解释器",
            os.path.expanduser("~"),
            filt
        )
        if path and os.path.isfile(path):
            if self.env_mgr.add_custom_environment(path):
                self.refresh_environments()

    def run_probe_test(self):
        """Run asynchronous probe testing in the selected python environment."""
        py_path = self.combo_env.currentData()
        if not py_path or not os.path.isfile(py_path):
            self.console_viewer.append_log("[WARN] 请先选择一个有效的 Python 解释器环境")
            return

        self.btn_test.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.console_viewer.clear_log()
        self.console_viewer.append_log(f">>> 启动环境检测: {py_path}")
        self.console_viewer.append_log(">>> 测试代码:")
        self.console_viewer.append_log("    import torch")
        self.console_viewer.append_log("    print(torch.__version__)")
        self.console_viewer.append_log("    print(torch.cuda.is_available())")
        self.console_viewer.append_log("    print(torch.cuda.device_count())")
        self.console_viewer.append_log("    print(f'CUDA：{torch.version.cuda}')")
        self.console_viewer.append_log("--------------------------------------------------")

        # Set badges to testing state
        self.badge_torch.update_style("neutral", "PyTorch: 检测中...")
        self.badge_gpu.update_style("neutral", "GPU: 检测中...")
        self.badge_cuda.update_style("neutral", "CUDA: 检测中...")
        self.badge_ultra.update_style("neutral", "Ultralytics: 检测中...")

        def _on_log(line: str):
            self.console_viewer.append_log(line)

        def _on_finish(res: dict):
            self.btn_test.setEnabled(True)
            self.btn_refresh.setEnabled(True)
            self._current_probe_info = res
            self._update_badges(res)
            self.env_changed.emit(res)

        self.env_mgr.probe_environment_async(py_path, log_cb=_on_log, finish_cb=_on_finish)

    def _update_badges(self, res: dict):
        # 1. PyTorch
        if res.get("torch_installed"):
            ver = res.get("torch_version", "")
            self.badge_torch.update_style("success", f"PyTorch {ver}")
        else:
            self.badge_torch.update_style("danger", "PyTorch 未就绪")

        # 2. GPU
        if res.get("cuda_available"):
            dev_names = res.get("device_names", [])
            raw_dev = dev_names[0] if dev_names else "可用"
            dev_str = format_gpu_model_name(raw_dev)
            self.badge_gpu.update_style("success", f"GPU 可用 ({dev_str})")
        elif res.get("torch_installed"):
            self.badge_gpu.update_style("warning", "仅 CPU 模式")
        else:
            self.badge_gpu.update_style("danger", "GPU 不可用")

        # 3. CUDA
        c_ver = res.get("cuda_version")
        if c_ver and c_ver != "None" and c_ver != "Unknown":
            self.badge_cuda.update_style("info", f"CUDA {c_ver}")
        else:
            self.badge_cuda.update_style("neutral", "CUDA: 无")

        # 4. Ultralytics
        if res.get("ultralytics_installed"):
            src = res.get("ultralytics_source", "")
            u_ver = res.get("ultralytics_version", "")
            if "labelpaw" in src.lower() or "trainingplatform" in src.lower():
                self.badge_ultra.update_style("info", f"Ultralytics {u_ver} (项目源码)")
            else:
                self.badge_ultra.update_style("success", f"Ultralytics {u_ver} (环境包)")
        else:
            self.badge_ultra.update_style("danger", "Ultralytics 未导入")

    def get_selected_python(self) -> str:
        data = self.combo_env.currentData()
        return data if data else sys.executable

    def get_selected_env_info(self) -> dict:
        return self._current_probe_info

    def is_gpu_available(self) -> bool:
        return bool(self._current_probe_info.get("cuda_available", False))

    def set_dark_mode(self, is_dark: bool):
        if self._is_dark != is_dark:
            self._is_dark = is_dark
            if hasattr(self, "combo_env") and hasattr(self.combo_env, "set_dark_mode"):
                self.combo_env.set_dark_mode(is_dark)
            self.console_viewer.set_dark_mode(is_dark)
            self._apply_button_styles()

    def _apply_button_styles(self):
        btn_style = """
            QPushButton {
                background-color: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 0px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f8fafc;
                border-color: #94a3b8;
            }
        """ if not self._is_dark else """
            QPushButton {
                background-color: #27272a;
                color: #e2e8f0;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 0px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3f3f46;
            }
        """
        self.btn_refresh.setStyleSheet(btn_style)
        self.btn_browse.setStyleSheet(btn_style)
        self.btn_test.setStyleSheet(btn_style)
