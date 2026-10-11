# -*- coding: utf-8 -*-
"""
New Project Creation Dialog for AI Vision Platform
Using MonkeyQt MkForm (top-aligned labels), MkInput, MkComboBox, MkButton, MkMessage
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QWidget, QFrame, QPushButton,
                               QGraphicsDropShadowEffect)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Signal, QPoint
from monkeyqt import MkForm, MkInput, MkComboBox, MkButton, MkMessage, apply_monkeyqt_theme, ThemeEngine
from monkeyqt_icons import PhFolderPlus, PhX, PhCheck

class ProjectCreateDialog(QDialog):
    project_created = Signal(int)  # emits new project_id

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.created_project_id = None

        self.setWindowTitle("新建视觉项目 (New Project)")
        # Frameless modern modal window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        _MkThemeEngine.apply_style_sheet(self, "QDialog { background: transparent; background-color: transparent; border: none; }")
        self.setMinimumWidth(500)
        self.resize(540, 420)

        self._init_ui()
        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)

    def _init_ui(self):
        # Outer layout leaving padding for shadow
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)

        # Dialog container card with rounded corners & shadow
        self.container_card = QFrame(self)
        self.container_card.setObjectName("DialogContainer")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 45))
        shadow.setOffset(0, 6)
        self.container_card.setGraphicsEffect(shadow)
        root_layout.addWidget(self.container_card)

        main_layout = QVBoxLayout(self.container_card)
        main_layout.setContentsMargins(28, 22, 28, 22)
        main_layout.setSpacing(16)

        # 1. Dialog Header with Close Button (X)
        self.header_widget = QWidget()
        _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(PhFolderPlus.regular(size=22, color="#3b82f6").pixmap(22, 22))
        _MkThemeEngine.apply_style_sheet(icon_lbl, "background: transparent; border: none;")
        header_layout.addWidget(icon_lbl)

        self.title_lbl = QLabel("新建视觉训练项目 (New Project)")
        _MkThemeEngine.apply_style_sheet(self.title_lbl, "background: transparent; border: none;")
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch()

        self.btn_close = QPushButton()
        self.btn_close.setToolTip("关闭")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        header_layout.addWidget(self.btn_close)

        main_layout.addWidget(self.header_widget)

        self.subtitle_lbl = QLabel("项目是管理模型实验、训练轮次及评估看板的统一容器。")
        _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "background: transparent; background-color: transparent; border: none;")
        main_layout.addWidget(self.subtitle_lbl)

        # 2. Form with top-aligned labels (Ultralytics modern style)
        self.form = MkForm(label_position="top")

        self.input_name = MkInput(placeholder="例如: PCB缺陷检测_v1 / Industrial_Screw")
        self.form.add_row("项目名称 (Project Name) *", self.input_name)

        self.input_desc = MkInput(placeholder="简述项目目标、应用场景或实验设想 (可选)...")
        self.form.add_row("项目描述 (Description)", self.input_desc)

        main_layout.addWidget(self.form)
        main_layout.addStretch()

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        self.btn_cancel = MkButton("取消", btn_type="default")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_create = MkButton("立即创建", btn_type="primary")
        self.btn_create.setIcon(PhCheck.regular(size=16))
        self.btn_create.clicked.connect(self.handle_create)
        btn_layout.addWidget(self.btn_create)

        main_layout.addLayout(btn_layout)

    def handle_create(self):
        name = self.input_name.text().strip()
        if not name:
            MkMessage.warning(self, "请输入有效的项目名称！")
            self.input_name.setFocus()
            return

        task_type = "mixed"
        desc = self.input_desc.text().strip()

        try:
            new_id = self.db.add_project(name=name, description=desc, task_type=task_type)
            self.created_project_id = new_id
            self.project_created.emit(new_id)
            self.accept()
        except Exception as e:
            if "UNIQUE" in str(e).upper():
                MkMessage.error(self, f"已存在同名项目 '{name}'，请更换名称！")
            else:
                MkMessage.error(self, f"创建失败: {str(e)}")

    # ── Window Dragging Handling for Frameless Dialog ──
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 70:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, '_drag_pos'):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, '_drag_pos'):
            delattr(self, '_drag_pos')
        super().mouseReleaseEvent(event)

    def _disconnect_theme(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ThemeEngine.instance().themeChanged.disconnect(self.apply_theme)
        except Exception:
            pass

    def closeEvent(self, event):
        self._disconnect_theme()
        super().closeEvent(event)

    def reject(self):
        self._disconnect_theme()
        super().reject()

    def accept(self):
        self._disconnect_theme()
        super().accept()

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self.container_card, """
                QFrame#DialogContainer {
                    background-color: #1e1e20;
                    border: 1px solid #303036;
                    border-radius: 12px;
                }
                QFrame#DialogContainer > QWidget {
                    background-color: transparent;
                    background: transparent;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 18px; font-weight: bold; color: #f4f4f5; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "color: #a1a1aa; font-size: 13px; margin-top: -6px; background: transparent; border: none;")
            self.btn_close.setIcon(PhX.bold(size=14, color="#a1a1aa"))
            _MkThemeEngine.apply_style_sheet(self.btn_close, """
                QPushButton {
                    border: none;
                    background-color: transparent;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #27272a;
                }
            """)
        else:
            _MkThemeEngine.apply_style_sheet(self.container_card, """
                QFrame#DialogContainer {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                }
                QFrame#DialogContainer > QWidget {
                    background-color: transparent;
                    background: transparent;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 18px; font-weight: bold; color: #0f172a; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "color: #64748b; font-size: 13px; margin-top: -6px; background: transparent; border: none;")
            self.btn_close.setIcon(PhX.bold(size=14, color="#64748b"))
            _MkThemeEngine.apply_style_sheet(self.btn_close, """
                QPushButton {
                    border: none;
                    background-color: transparent;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                }
            """)
        apply_monkeyqt_theme(self)
