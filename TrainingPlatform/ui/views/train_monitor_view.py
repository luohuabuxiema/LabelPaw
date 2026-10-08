# -*- coding: utf-8 -*-
"""
Real-time Training Monitor View for AI Vision Training Platform
Using MonkeyQt MkConsole, MkProgressBar, MetricChart, MkCard, MkButton
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QTextEdit, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QTextCursor

from monkeyqt import (MkCard, MkButton, MkProgressBar,
                      MkComboBox, MkConsole, MkAlert, MkQWidget)
from monkeyqt_icons import PhStop, PhPlay
from TrainingPlatform.ui.components.metric_chart import MetricChart

class TrainMonitorView(MkQWidget):
    def __init__(self, db_manager, task_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager
        self.task_mgr = task_manager
        self.current_task_id = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header Bar
        hdr_layout = QHBoxLayout()
        title_lbl = QLabel("实时训练监控 Training Monitor")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 20px; font-weight: bold; color: #303133;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()

        hdr_layout.addWidget(QLabel("当前训练任务:"))
        self.combo_tasks = MkComboBox()
        self.combo_tasks.setMinimumWidth(250)
        self.combo_tasks.currentIndexChanged.connect(self.on_task_selected)
        hdr_layout.addWidget(self.combo_tasks)

        self.btn_stop = MkButton("终止训练")
        self.btn_stop.setIcon(PhStop.regular())
        _MkThemeEngine.apply_style_sheet(self.btn_stop, "background-color: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2;")
        self.btn_stop.clicked.connect(self.stop_current_task)
        hdr_layout.addWidget(self.btn_stop)

        self.btn_start = MkButton("启动/恢复训练")
        self.btn_start.setIcon(PhPlay.regular())
        self.btn_start.clicked.connect(self.start_current_task)
        hdr_layout.addWidget(self.btn_start)

        main_layout.addLayout(hdr_layout)

        # Status Summary Bar
        status_card = MkCard(title="")
        s_layout = status_card.content_layout

        self.lbl_status = QLabel("状态: 待定")
        _MkThemeEngine.apply_style_sheet(self.lbl_status, "font-size: 14px; font-weight: bold; color: #409eff;")

        self.lbl_model = QLabel("模型: -")
        self.lbl_epoch = QLabel("Epoch: 0 / 0")

        self.progress_bar = MkProgressBar()
        self.progress_bar.percentage = 0

        self.lbl_map50 = QLabel("mAP50: 0.0000")
        _MkThemeEngine.apply_style_sheet(self.lbl_map50, "font-size: 14px; font-weight: bold; color: #67c23a;")

        box_info = QHBoxLayout()
        box_info.addWidget(self.lbl_status)
        box_info.addWidget(self.lbl_model)
        box_info.addWidget(self.lbl_epoch)
        box_info.addWidget(self.progress_bar, 1)
        box_info.addWidget(self.lbl_map50)
        s_layout.addLayout(box_info)

        main_layout.addWidget(status_card)

        # Main Splitter: Top Real-time Metric Charts, Bottom Log Console
        splitter = QSplitter(Qt.Horizontal)

        # Top Charts Widget
        self.chart = MetricChart()
        splitter.addWidget(self.chart)

        # Bottom Log Console Widget
        log_card = MkCard(title="训练标准输出日志 (Real-time Log Console)")
        log_layout = log_card.content_layout

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        _MkThemeEngine.apply_style_sheet(self.console, """
            QTextEdit {
                background-color: #1e1e1e;
                color: #67c23a;
                font-family: Consolas, monospace;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }
        """)
        log_layout.addWidget(self.console)
        splitter.addWidget(log_card)

        splitter.setSizes([350, 250])
        main_layout.addWidget(splitter, 1)

        # Connect TaskManager Signals
        self.task_mgr.task_log.connect(self.on_task_log)
        self.task_mgr.task_progress.connect(self.on_task_progress)
        self.task_mgr.task_finished.connect(self.on_task_finished)

        self.refresh_task_list()

    def refresh_task_list(self, select_task_id=None):
        tasks = self.db.get_tasks()
        self.combo_tasks.blockSignals(True)
        self.combo_tasks.clear()
        selected_idx = 0

        for idx, t in enumerate(tasks):
            txt = f"[{t['id']}] {t['task_name']} ({t['status']})"
            self.combo_tasks.addItem(txt, t['id'])
            if select_task_id and t['id'] == select_task_id:
                selected_idx = idx

        if tasks:
            self.combo_tasks.setCurrentIndex(selected_idx)

        self.combo_tasks.blockSignals(False)
        self.on_task_selected()

    def on_task_selected(self):
        idx = self.combo_tasks.currentIndex()
        if idx < 0:
            self.current_task_id = None
            return

        task_id = self.combo_tasks.itemData(idx)
        self.current_task_id = task_id
        task = self.db.get_task_by_id(task_id)

        if task:
            self.lbl_status.setText(f"状态: {task['status']}")
            self.lbl_model.setText(f"模型: {task['model_name']}")
            self.lbl_epoch.setText(f"Epoch: {task['current_epoch']} / {task['total_epochs']}")
            self.progress_bar.percentage = int(task['progress'])
            self.lbl_map50.setText(f"mAP50: {task['best_map50']:.4f}")

            # Load history metrics into chart
            self.chart.clear()
            metrics_list = self.db.get_task_metrics(task_id)
            for m in metrics_list:
                self.chart.add_epoch_metric(
                    m["epoch"],
                    m["train_box_loss"],
                    m["train_cls_loss"],
                    m["map50"]
                )

    def select_task_by_id(self, task_id):
        self.refresh_task_list(select_task_id=task_id)

    @Slot(int, str)
    def on_task_log(self, task_id, msg):
        if task_id == self.current_task_id:
            self.console.append(msg)
            self.console.moveCursor(QTextCursor.End)

    @Slot(int, dict)
    def on_task_progress(self, task_id, metrics):
        if task_id == self.current_task_id:
            epoch = metrics.get("epoch", 0)
            total = metrics.get("total_epochs", 100)
            progress = int(metrics.get("progress", 0))
            map50 = metrics.get("map50", 0.0)

            self.lbl_status.setText("状态: RUNNING 运行中")
            self.lbl_epoch.setText(f"Epoch: {epoch} / {total}")
            self.progress_bar.percentage = progress
            self.lbl_map50.setText(f"mAP50: {map50:.4f}")

            self.chart.add_epoch_metric(
                epoch,
                metrics.get("train_box_loss", 0.0),
                metrics.get("train_cls_loss", 0.0),
                map50
            )

    @Slot(int, bool, str)
    def on_task_finished(self, task_id, success, msg):
        if task_id == self.current_task_id:
            st = "SUCCESS 已完成" if success else "FAILED 异常/停止"
            self.lbl_status.setText(f"状态: {st}")
            self.console.append(f"\n[{st}] {msg}")
            self.refresh_task_list(select_task_id=task_id)

    def start_current_task(self):
        if not self.current_task_id:
            return
        success, msg = self.task_mgr.start_task(self.current_task_id)
        if not success:
            QMessageBox.warning(self, "启动失败", msg)
        else:
            self.refresh_task_list(select_task_id=self.current_task_id)

    def stop_current_task(self):
        if not self.current_task_id:
            return
        success, msg = self.task_mgr.stop_task(self.current_task_id)
        QMessageBox.information(self, "终止提示", msg)
        self.refresh_task_list(select_task_id=self.current_task_id)
