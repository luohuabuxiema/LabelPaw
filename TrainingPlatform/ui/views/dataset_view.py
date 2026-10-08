# -*- coding: utf-8 -*-
"""
Dataset Management View for AI Vision Training Platform
Includes upload, structure inspection, zip extraction, and dataset tools.
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QFileDialog, QMessageBox, QDialog, QFrame,
                               QPushButton, QGraphicsDropShadowEffect)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QPoint

from monkeyqt import MkCard, MkButton, MkInput, MkComboBox, MkForm, MkAlert, MkQWidget
from monkeyqt_icons import PhWrench, PhUploadSimple, PhX
from TrainingPlatform.core.dataset_inspector import DatasetInspector

class ImportDatasetDialog(QDialog):
    def __init__(self, projects, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入/上传 YOLO 数据集")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(540, 420)

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
        layout.setSpacing(14)

        # Header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("导入/上传 YOLO 数据集")
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

        self.combo_project = MkComboBox()
        self.projects = projects
        for p in projects:
            self.combo_project.addItem(f"[{p['id']}] {p['name']}", p['id'])

        self.input_name = MkInput()
        self.input_name.setPlaceholderText("数据集名称 (例: PCB检测数据集v1)")

        self.input_path = MkInput()
        self.input_path.setPlaceholderText("选择/解压后的 YOLO 数据集文件夹路径")

        btn_browse = MkButton("选择目录")
        btn_browse.clicked.connect(self.browse_dir)

        btn_browse_zip = MkButton("从 .zip 导入解压")
        btn_browse_zip.clicked.connect(self.browse_zip)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.input_path)
        path_layout.addWidget(btn_browse)
        path_layout.addWidget(btn_browse_zip)

        form = MkForm()
        form.add_row("所属项目:", self.combo_project)
        form.add_row("数据集名称:", self.input_name)
        form.add_row("数据集路径:", path_layout)

        layout.addWidget(form)

        # Inspection result preview
        self.lbl_inspect = QLabel("请选择有效的 YOLO 数据集路径进行体检分析...")
        self.lbl_inspect.setWordWrap(True)
        _MkThemeEngine.apply_style_sheet(self.lbl_inspect, "color: #909399; font-size: 12px; border: 1px dashed #dcdfe6; padding: 10px; border-radius: 4px;")
        layout.addWidget(self.lbl_inspect)

        self.inspection_res = None

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = MkButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save = MkButton("确认导入数据", btn_type="primary")
        self.btn_save.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)

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

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if d:
            self.input_path.setText(d)
            self.run_inspection(d)

    def browse_zip(self):
        zip_f, _ = QFileDialog.getOpenFileName(self, "选择数据集 Zip 压缩包", "", "Zip Archives (*.zip)")
        if zip_f:
            target_dir = os.path.join(os.path.expanduser("~"), ".ai_vision_platform", "datasets",
                                      os.path.splitext(os.path.basename(zip_f))[0])
            try:
                DatasetInspector.extract_zip(zip_f, target_dir)
                self.input_path.setText(target_dir)
                self.run_inspection(target_dir)
                QMessageBox.information(self, "成功", f"Zip 解压完成，存放至:\n{target_dir}")
            except Exception as e:
                QMessageBox.critical(self, "解压失败", str(e))

    def run_inspection(self, path):
        res = DatasetInspector.inspect(path)
        self.inspection_res = res

        if not res["valid"]:
            self.lbl_inspect.setText(f"数据集体检未通过: {res['error_msg']}")
            _MkThemeEngine.apply_style_sheet(self.lbl_inspect, "color: #f56c6c; font-size: 12px; border: 1px dashed #fde2e2; padding: 10px; border-radius: 4px;")
        else:
            txt = (f"数据集体检合格！\n"
                   f"总图片数: {res['image_count']} (Train: {res['train_images']} | Val: {res['val_images']} | Test: {res['test_images']})\n"
                   f"类别数: {res['class_count']} | 类别名称: {', '.join(res['classes'][:8])}")
            self.lbl_inspect.setText(txt)
            _MkThemeEngine.apply_style_sheet(self.lbl_inspect, "color: #67c23a; font-size: 12px; border: 1px dashed #e1f3d8; padding: 10px; border-radius: 4px;")

            if not self.input_name.text():
                self.input_name.setText(os.path.basename(path))

    def get_data(self):
        idx = self.combo_project.currentIndex()
        pid = self.projects[idx]['id'] if self.projects and idx >= 0 else None
        return {
            "project_id": pid,
            "name": self.input_name.text().strip(),
            "path": self.input_path.text().strip(),
            "inspection": self.inspection_res
        }


class DatasetView(MkQWidget):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header Bar
        hdr_layout = QHBoxLayout()
        title_lbl = QLabel("数据集管理 Dataset Management")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 20px; font-weight: bold; color: #303133;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()

        self.btn_tools = MkButton("打开高级转换/划分工具箱")
        self.btn_tools.setIcon(PhWrench.regular())
        self.btn_tools.clicked.connect(self.open_dataset_tool)
        hdr_layout.addWidget(self.btn_tools)

        self.btn_add = MkButton("导入/上传数据集")
        self.btn_add.setIcon(PhUploadSimple.regular())
        self.btn_add.clicked.connect(self.open_import_dialog)
        hdr_layout.addWidget(self.btn_add)

        layout.addLayout(hdr_layout)

        # Table Card
        card = MkCard(title="数据集列表")
        card_layout = card.content_layout

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "数据集名称", "关联项目", "图片数量", "类别数量", "路径", "操作"
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
        self.refresh_datasets()

    def refresh_datasets(self):
        datasets = self.db.get_datasets()
        projects = {p['id']: p['name'] for p in self.db.get_projects()}
        self.table.setRowCount(len(datasets))

        for row, d in enumerate(datasets):
            self.table.setItem(row, 0, QTableWidgetItem(str(d["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(d["name"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(projects.get(d["project_id"], "-"))))
            self.table.setItem(row, 3, QTableWidgetItem(str(d["image_count"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(d["class_count"])))
            self.table.setItem(row, 5, QTableWidgetItem(str(d["path"])))

            btn_del = MkButton("删除记录")
            _MkThemeEngine.apply_style_sheet(btn_del, "background-color: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2;")
            btn_del.clicked.connect(lambda _, did=d["id"]: self.delete_dataset(did))
            self.table.setCellWidget(row, 6, btn_del)

    def open_import_dialog(self):
        projects = self.db.get_projects()
        if not projects:
            QMessageBox.warning(self, "提示", "请先在【项目管理】中创建一个训练项目！")
            return

        dlg = ImportDatasetDialog(projects, self)
        if dlg.exec():
            data = dlg.get_data()
            if not data["name"] or not data["path"]:
                QMessageBox.warning(self, "警告", "数据集名称和路径不能为空！")
                return

            insp = data["inspection"] or DatasetInspector.inspect(data["path"])
            img_count = insp.get("image_count", 0)
            class_count = insp.get("class_count", 0)
            classes = insp.get("classes", [])

            self.db.add_dataset(
                project_id=data["project_id"],
                name=data["name"],
                path=data["path"],
                fmt='yolo',
                image_count=img_count,
                class_count=class_count,
                classes=classes
            )
            self.refresh_datasets()

    def open_dataset_tool(self):
        try:
            win = self.window()
            if hasattr(win, 'sidebar') and hasattr(win, 'stacked'):
                win.sidebar.set_active("tool_to_yolo")
                win.stacked.setCurrentIndex(6)
            else:
                from TrainingPlatform.ui.views.dataset_tool_view import DatasetToolView
                self.tool_win = DatasetToolView()
                self.tool_win.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法切换至数据集转换工具: {str(e)}")

    def delete_dataset(self, dataset_id):
        reply = QMessageBox.question(self, "确认删除", "确定要删除该数据集记录吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_dataset(dataset_id)
            self.refresh_datasets()
