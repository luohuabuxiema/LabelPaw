# -*- coding: utf-8 -*-
"""
Project Management View for AI Vision Training Platform
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QDialog)
from PySide6.QtCore import Qt

from monkeyqt import MkCard, MkButton, MkInput, MkComboBox, MkForm, MkAlert, MkQWidget, MkMessageBox
from monkeyqt_icons import PhPlus

class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建训练项目")
        self.resize(420, 300)
        _MkThemeEngine.apply_style_sheet(self, "background-color: white;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.form = MkForm()
        self.input_name = MkInput(placeholder="例如: 猫狗目标检测")
        self.form.add_row("项目名称:", self.input_name)

        self.input_desc = MkInput(placeholder="简述项目训练目标或场景")
        self.form.add_row("项目描述:", self.input_desc)

        layout.addWidget(self.form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = MkButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_submit = MkButton("创建项目")
        _MkThemeEngine.apply_style_sheet(self.btn_submit, "background-color: #409eff; color: white; border: none;")
        self.btn_submit.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_submit)

        layout.addLayout(btn_layout)

    def get_data(self):
        return {
            "name": self.input_name.text().strip(),
            "description": self.input_desc.text().strip(),
            "task_type": "mixed"
        }


class ProjectView(MkQWidget):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header Bar
        hdr_layout = QHBoxLayout()
        title_lbl = QLabel("训练项目管理 Project Management")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 20px; font-weight: bold; color: #303133;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()

        self.btn_add = MkButton("新建项目")
        self.btn_add.setIcon(PhPlus.regular())
        self.btn_add.clicked.connect(self.open_add_dialog)
        hdr_layout.addWidget(self.btn_add)
        layout.addLayout(hdr_layout)

        # Project Table Card
        card = MkCard(title="训练项目列表")
        card_layout = card.content_layout

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "ID", "项目名称", "创建时间", "操作"
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
        self.refresh_projects()

    def refresh_projects(self):
        projects = self.db.get_projects()
        self.table.setRowCount(len(projects))

        for row, p in enumerate(projects):
            self.table.setItem(row, 0, QTableWidgetItem(str(p["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(p["name"])))

            self.table.setItem(row, 2, QTableWidgetItem(str(p["create_time"])))

            # Action button container
            btn_del = MkButton("删除")
            _MkThemeEngine.apply_style_sheet(btn_del, "background-color: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2;")
            btn_del.clicked.connect(lambda _, pid=p["id"]: self.delete_project(pid))
            self.table.setCellWidget(row, 3, btn_del)

    def open_add_dialog(self):
        from TrainingPlatform.ui.views.dialogs import ProjectCreateDialog
        dlg = ProjectCreateDialog(self.db, self)
        if dlg.exec():
            self.refresh_projects()

    def delete_project(self, project_id):
        if MkMessageBox.confirm(
            self,
            "确认删除",
            "确定要删除该项目吗？这不会自动删除文件磁盘数据。",
            danger=True,
            ok_text="删除",
            cancel_text="取消"
        ):
            self.db.delete_project(project_id)
            self.refresh_projects()
