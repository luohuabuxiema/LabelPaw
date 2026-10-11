# -*- coding: utf-8 -*-
"""
Model Management & Export View for AI Vision Training Platform
Using MonkeyQt MkCard, MkButton, MkComboBox, MkDataTable
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
import subprocess
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QDialog, QFrame, QPushButton,
                               QGraphicsDropShadowEffect)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Signal, QPoint

from monkeyqt import MkCard, MkButton, MkComboBox, MkForm, MkSwitch, MkAlert, MkQWidget, MkMessageBox
from monkeyqt_icons import PhX

class ExportModelDialog(QDialog):
    def __init__(self, model_info, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.setWindowTitle(f"一键导出模型 - {model_info['name']}")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(480, 340)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)

        self.container_card = QFrame(self)
        self.container_card.setObjectName("DialogContainer")
        _MkThemeEngine.apply_style_sheet(self.container_card, """
            QFrame#DialogContainer {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 45))
        shadow.setOffset(0, 6)
        self.container_card.setGraphicsEffect(shadow)
        root_layout.addWidget(self.container_card)

        layout = QVBoxLayout(self.container_card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel(f"导出模型: {model_info['name']}")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 16px; font-weight: bold; color: #0f172a;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        btn_close = QPushButton()
        btn_close.setIcon(PhX.bold(size=14, color="#64748b"))
        btn_close.setToolTip("关闭")
        btn_close.setFixedSize(28, 28)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        _MkThemeEngine.apply_style_sheet(btn_close, """
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        btn_close.clicked.connect(self.reject)
        header_layout.addWidget(btn_close)
        layout.addWidget(header_widget)

        self.combo_format = MkComboBox()
        self.combo_format.addItems([
            "onnx (ONNX 跨平台格式)",
            "engine (TensorRT GPU 高速引擎)",
            "openvino (Intel CPU/NPU 加速)",
            "coreml (Apple iOS/Mac 加速)",
            "tflite (TensorFlow Lite 移动端)"
        ])

        self.switch_dynamic = MkSwitch()
        self.switch_half = MkSwitch()

        form = MkForm()
        form.add_row("目标导出格式:", self.combo_format)
        form.add_row("动态 Shape (Dynamic):", self.switch_dynamic)
        form.add_row("半精度 (FP16):", self.switch_half)

        layout.addWidget(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = MkButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_export = MkButton("开始导出", btn_type="primary")
        self.btn_export.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_export)

        layout.addLayout(btn_layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 60:
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

    def get_export_args(self):
        fmt_raw = self.combo_format.currentText().split()[0]
        return {
            "format": fmt_raw,
            "dynamic": self.switch_dynamic.isChecked(),
            "half": self.switch_half.isChecked()
        }


class ModelView(MkQWidget):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header Bar
        hdr_layout = QHBoxLayout()
        title_lbl = QLabel("模型产出与导出管理 Model Management")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 20px; font-weight: bold; color: #303133;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()
        layout.addLayout(hdr_layout)

        # Table Card
        card = MkCard(title="已产出 YOLO 模型列表")
        card_layout = card.content_layout

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID", "模型名称", "关联项目", "模型架构", "最佳权重路径", "操作"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        _MkThemeEngine.apply_style_sheet(self.table, """
            QTableWidget { background: white; border: 1px solid #e2e8f0; border-radius: 6px; }
            QHeaderView::section { background: #f8fafc; font-weight: bold; color: #475569; border: none; border-bottom: 1px solid #e2e8f0; height: 34px; }
            QTableWidget::item { border-bottom: 1px solid #f1f5f9; padding: 8px; }
        """)
        card_layout.addWidget(self.table)

        layout.addWidget(card)
        self.refresh_models()

    def refresh_models(self):
        models = self.db.get_models()
        self.table.setRowCount(len(models))

        for row, m in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(str(m["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(m["name"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(m.get("project_name", "-"))))
            self.table.setItem(row, 3, QTableWidgetItem(str(m["model_arch"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(m["best_pt_path"])))

            # Action Buttons Layout
            btn_box = QWidget()
            b_layout = QHBoxLayout(btn_box)
            b_layout.setContentsMargins(0, 0, 0, 0)
            b_layout.setSpacing(5)

            btn_open = MkButton("打开目录")
            btn_open.clicked.connect(lambda _, p=m["best_pt_path"]: self.open_file_dir(p))

            btn_export = MkButton("导出模型")
            btn_export.clicked.connect(lambda _, info=m: self.export_model(info))

            btn_del = MkButton("删除")
            _MkThemeEngine.apply_style_sheet(btn_del, "background-color: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2;")
            btn_del.clicked.connect(lambda _, mid=m["id"]: self.delete_model(mid))

            b_layout.addWidget(btn_open)
            b_layout.addWidget(btn_export)
            b_layout.addWidget(btn_del)

            self.table.setCellWidget(row, 5, btn_box)

    def open_file_dir(self, file_path):
        if not file_path or not os.path.exists(file_path):
            MkMessageBox.warning(self, "路径错误", f"找不到指定的文件: {file_path}", cancel_text=None)
            return
        folder = os.path.dirname(file_path)
        if os.name == 'nt':
            os.startfile(folder)
        else:
            subprocess.Popen(['xdg-open', folder])

    def export_model(self, model_info):
        best_pt = model_info.get("best_pt_path", "")
        if not os.path.exists(best_pt):
            MkMessageBox.error(self, "错误", f"未找到最佳模型文件 best.pt:<br>{best_pt}")
            return

        dlg = ExportModelDialog(model_info, self)
        if dlg.exec():
            args = dlg.get_export_args()
            fmt = args["format"]
            try:
                from ultralytics import YOLO
                m = YOLO(best_pt)
                MkMessageBox.info(self, "导出中", f"正在后台导出为 {fmt} 格式，请稍等系统执行处理...")
                exported_path = m.export(format=fmt, dynamic=args["dynamic"], half=args["half"])
                MkMessageBox.success(self, "导出成功", f"模型成功导出至:<br>{exported_path}")
            except Exception as e:
                MkMessageBox.error(self, "导出失败", f"模型转码导出过程发生错误:<br>{str(e)}")

    def delete_model(self, model_id):
        if MkMessageBox.confirm(self, "确认删除", "确定要删除该模型记录吗？", danger=True, ok_text="删除", cancel_text="取消"):
            self.db.delete_model(model_id)
            self.refresh_models()
