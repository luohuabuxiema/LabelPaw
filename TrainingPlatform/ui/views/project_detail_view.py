# -*- coding: utf-8 -*-
"""
Project Detail View (Unified Workspace) for AI Vision Platform
Benchmarked against Ultralytics Platform Project Dashboard:
- Project Top Summary & Navigation Header
- Empty State: Centered Card + Drag-and-Drop weights (.pt) upload + Primary CTA Button
- Active State:
  - Left Pane: Master Task List + Mini Weight Upload Zone
  - Right Pane: Real-time MetricChart + Status Controls + MkConsole Terminal Output
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
from typing import Any
try:
    import pandas as pd
except ImportError:
    pd = None
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSplitter, QListWidget, QListWidgetItem,
                               QMessageBox, QStackedWidget, QFrame, QToolButton,
                               QCheckBox, QSizePolicy, QScrollArea)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

from monkeyqt import (MkCard, MkButton, MkProgressBar, MkConsole, 
                      MkUpload, MkMessage, MkQWidget, ThemeEngine,
                      MkSelectCard, MkSelectItem, MkItemActionButton)
from monkeyqt_icons import (PhPlus, PhPlay, PhStop, PhExport, 
                            PhArrowClockwise, PhTrash, PhFolder, PhCaretRight)

from TrainingPlatform.ui.components.thesis_chart import ThesisChartCard, ACADEMIC_COLORS
from TrainingPlatform.ui.components.terminal_formatter import (apply_terminal_console_style,
                                                               append_terminal_line_to_console)
from TrainingPlatform.ui.views.experiment_detail_view import ExperimentDetailView
from TrainingPlatform.ui.views.dialogs.train_create_dialog import TrainCreateDialog
from TrainingPlatform.ui.views.model_view import ExportModelDialog


class ProjectDetailView(MkQWidget):
    project_deleted = Signal(int)  # emits project_id
    project_updated = Signal(int)  # emits project_id

    def __init__(self, db_manager, task_manager, project_id=None, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager
        self.task_mgr = task_manager
        self.project_id = project_id
        self.project_info = None
        self.current_task_id = None
        self.experiment_detail_view = None

        self._init_ui()
        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)
        self._wire_task_manager_signals()

        if self.project_id:
            self.load_project(self.project_id)

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(16)
        main_layout = self.main_layout

        # ── 1. Top Project Header ──
        self.header_card = QWidget()
        h_layout = QHBoxLayout(self.header_card)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(12)

        # Left Info
        v_info = QVBoxLayout()
        v_info.setSpacing(4)

        row_title = QHBoxLayout()
        row_title.setSpacing(10)

        self.lbl_title = QLabel("项目详情")
        _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 22px; font-weight: bold; color: #1e293b;")
        row_title.addWidget(self.lbl_title)

        self.lbl_task_type = QLabel("detect")
        _MkThemeEngine.apply_style_sheet(self.lbl_task_type, "background: #eff6ff; color: #2563eb; font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 12px;")
        row_title.addWidget(self.lbl_task_type)
        row_title.addStretch()

        v_info.addLayout(row_title)

        self.lbl_desc = QLabel("加载中...")
        _MkThemeEngine.apply_style_sheet(self.lbl_desc, "color: #64748b; font-size: 13px;")
        v_info.addWidget(self.lbl_desc)

        h_layout.addLayout(v_info, stretch=1)

        # Right Action Buttons
        self.btn_new_train = MkButton("新建训练任务", btn_type="primary")
        self.btn_new_train.setIcon(PhPlus.bold(size=15))
        _MkThemeEngine.apply_style_sheet(self.btn_new_train, "font-weight: bold; padding: 8px 16px;")
        self.btn_new_train.clicked.connect(self.open_train_dialog)
        h_layout.addWidget(self.btn_new_train)

        self.btn_refresh = MkButton("", btn_type="default")
        self.btn_refresh.setIcon(PhArrowClockwise.regular(size=16))
        self.btn_refresh.setToolTip("刷新项目状态")
        self.btn_refresh.setFixedSize(36, 36)
        self.btn_refresh.clicked.connect(lambda: self.load_project(self.project_id))
        h_layout.addWidget(self.btn_refresh)

        self.btn_delete_proj = MkButton("", btn_type="default")
        self.btn_delete_proj.setIcon(PhTrash.regular(size=16))
        self.btn_delete_proj.setToolTip("删除项目")
        self.btn_delete_proj.setFixedSize(36, 36)
        self.btn_delete_proj.clicked.connect(self.handle_delete_project)
        h_layout.addWidget(self.btn_delete_proj)

        main_layout.addWidget(self.header_card)

        # ── 2. Content Stack (Active Dashboard vs Empty State vs Experiment Detail) ──
        self.content_stack = QStackedWidget()

        # State B: Dashboard Master-Visualizer Widget (Like Ultralytics Figure 1)
        self.dashboard_widget = self._create_dashboard_widget()
        self.content_stack.addWidget(self.dashboard_widget)

        # State A: Empty State Widget (Like Ultralytics Figure 4)
        self.empty_widget = self._create_empty_state_widget()
        self.content_stack.addWidget(self.empty_widget)

        # State C: Experiment Detail View (Lazy initialized on first demand)
        self.experiment_detail_view = None

        main_layout.addWidget(self.content_stack, stretch=1)

    def _create_empty_state_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        card = MkCard(title="")
        card_layout = card.content_layout
        card_layout.setContentsMargins(36, 36, 36, 36)
        card_layout.setSpacing(20)
        card_layout.setAlignment(Qt.AlignCenter)

        # Drag and Drop Upload Area for .pt files
        self.empty_upload = MkUpload(accept_filters=[".pt", "*.pt"], multiple=True, max_size_mb=2048)
        self.empty_upload.filesSelected.connect(self.handle_weights_dropped)
        card_layout.addWidget(self.empty_upload)

        self.empty_title = QLabel("暂无训练记录与模型权重")
        self.empty_title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.empty_title)

        subtitle = QLabel("您可以直接在此拖放已有 PyTorch .pt 权重文件进行导入管理，或者发起全新的本地 YOLO 训练任务。")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        _MkThemeEngine.apply_style_sheet(subtitle, "color: #64748b; font-size: 13px; max-width: 500px;")
        card_layout.addWidget(subtitle)

        btn_cta = MkButton("立即发起首个本地训练任务", btn_type="primary")
        btn_cta.setIcon(PhPlay.regular(size=16))
        _MkThemeEngine.apply_style_sheet(btn_cta, "font-size: 15px; font-weight: bold; padding: 12px 28px;")
        btn_cta.clicked.connect(self.open_train_dialog)
        card_layout.addWidget(btn_cta)

        layout.addWidget(card)
        return widget

    def _create_dashboard_widget(self):
        is_dark = ThemeEngine.is_dark()
        # Horizontal Splitter: Left Task List, Right Visualizer Dashboard
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(12)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        spl_qss = """
            QSplitter {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: transparent;
                border: none;
            }
            QSplitter::handle:hover {
                background: transparent;
                border: none;
            }
            QSplitter::handle:pressed {
                background: transparent;
                border: none;
            }
        """
        _MkThemeEngine.apply_style_sheet(self.splitter, spl_qss)

        # ── Left Pane: Models / Tasks Picker (Width ~ 300px) ──
        left_pane = MkQWidget(role="transparent")
        left_pane.setMinimumWidth(260)
        left_pane.setMaximumWidth(360)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # MonkeyQt MkSelectCard component (integrated title, badge, select all, and list)
        self.task_card = MkSelectCard(
            title="训练任务与模型",
            icon=None,
            show_select_all=True,
            select_all_text="全选 / 取消全选",
            badge_pattern="{count} 已选",
            parent=left_pane
        )
        self.task_card.selection_changed.connect(self._on_tasks_selection_changed)
        self.task_card.item_navigated.connect(self.open_task_detail)
        self.task_card.item_action_clicked.connect(self._on_task_action_clicked)
        left_layout.addWidget(self.task_card, stretch=1)

        # Mini drop area for existing .pt models
        lbl_drop = QLabel("快速添加已有权重 (.pt):")
        _MkThemeEngine.apply_style_sheet(lbl_drop, "color: #94a3b8; font-size: 11px;")
        left_layout.addWidget(lbl_drop)

        self.mini_upload = MkUpload(accept_filters=[".pt", "*.pt"], multiple=True, max_size_mb=2048)
        self.mini_upload.filesSelected.connect(self.handle_weights_dropped)
        left_layout.addWidget(self.mini_upload)

        self.splitter.addWidget(left_pane)

        # ── Right Pane: Visualizer Dashboard ──
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Vertical Splitter: Top 3 Core Metrics Comparison Charts, Bottom Status & Console
        self.rv_splitter = QSplitter(Qt.Vertical)
        self.rv_splitter.setHandleWidth(12)
        self.rv_splitter.setChildrenCollapsible(False)
        self.rv_splitter.setCollapsible(0, False)
        self.rv_splitter.setCollapsible(1, False)
        _MkThemeEngine.apply_style_sheet(self.rv_splitter, spl_qss)

        # Top Container: 3 Academic Thesis Charts (Requirement 2 & Image 1, 2, 3)
        charts_container = QWidget()
        charts_container.setMinimumHeight(280)
        charts_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        charts_layout = QVBoxLayout(charts_container)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(8)

        metrics_hdr = QHBoxLayout()
        metrics_hdr.setSpacing(8)
        self.lbl_metrics_title = QLabel("核心评估指标对比 (3/3)")
        _MkThemeEngine.apply_style_sheet(self.lbl_metrics_title, "font-size: 14px; font-weight: 700; color: #0f172a;")
        metrics_hdr.addWidget(self.lbl_metrics_title)

        metrics_hdr.addStretch()

        self.lbl_metrics_hint = QLabel("数据实时读取自 results.csv · 悬停可查看数值与准线 · 支持全屏")
        _MkThemeEngine.apply_style_sheet(self.lbl_metrics_hint, "color: #94a3b8; font-size: 11px;")
        metrics_hdr.addWidget(self.lbl_metrics_hint)

        charts_layout.addLayout(metrics_hdr)

        # 3 Charts side-by-side: precision(B), recall(B), mAP50-95(B)
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        self.chart_precision = ThesisChartCard(title="precision(B)", is_dark=is_dark)
        self.chart_recall = ThesisChartCard(title="recall(B)", is_dark=is_dark)
        self.chart_map50_95 = ThesisChartCard(title="mAP50-95(B)", is_dark=is_dark)

        charts_row.addWidget(self.chart_precision, stretch=1)
        charts_row.addWidget(self.chart_recall, stretch=1)
        charts_row.addWidget(self.chart_map50_95, stretch=1)

        charts_layout.addLayout(charts_row, stretch=1)
        self.rv_splitter.addWidget(charts_container)

        # Bottom Container: Task Status & Real-time Console Output
        bottom_container = QWidget()
        bottom_container.setMinimumHeight(180)
        bottom_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        b_layout = QVBoxLayout(bottom_container)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(8)

        # Selected Task Status Bar
        self.task_status_card = MkCard(title="")
        sc_layout = self.task_status_card.content_layout
        sc_layout.setContentsMargins(14, 8, 14, 8)

        s_bar = QHBoxLayout()
        s_bar.setSpacing(10)

        self.lbl_cur_task = QLabel("任务: -")
        _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 14px; color: #1e293b;")
        s_bar.addWidget(self.lbl_cur_task)

        self.lbl_cur_status = QLabel("待定")
        _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #e2e8f0; color: #475569;")
        s_bar.addWidget(self.lbl_cur_status)

        self.lbl_cur_model = QLabel("模型: -")
        _MkThemeEngine.apply_style_sheet(self.lbl_cur_model, "color: #64748b; font-size: 12px;")
        s_bar.addWidget(self.lbl_cur_model)

        self.progress_bar = MkProgressBar()
        self.progress_bar.percentage = 0
        s_bar.addWidget(self.progress_bar, stretch=1)

        # Action Buttons on status bar
        self.btn_view_detail = MkButton("进入详情页面 →", btn_type="primary")
        _MkThemeEngine.apply_style_sheet(self.btn_view_detail, "font-weight: bold; padding: 4px 12px;")
        self.btn_view_detail.clicked.connect(lambda: self.open_task_detail(self.current_task_id) if self.current_task_id else None)
        s_bar.addWidget(self.btn_view_detail)

        self.btn_stop_task = MkButton("终止训练", btn_type="danger")
        self.btn_stop_task.setIcon(PhStop.regular(size=14))
        self.btn_stop_task.clicked.connect(self.handle_stop_task)
        s_bar.addWidget(self.btn_stop_task)

        self.btn_resume_task = MkButton("启动/恢复", btn_type="default")
        self.btn_resume_task.setIcon(PhPlay.regular(size=14))
        self.btn_resume_task.clicked.connect(self.handle_resume_task)
        s_bar.addWidget(self.btn_resume_task)

        self.btn_export_model = MkButton("导出", btn_type="default")
        self.btn_export_model.setIcon(PhExport.regular(size=14))
        self.btn_export_model.clicked.connect(self.handle_export_model)
        s_bar.addWidget(self.btn_export_model)

        sc_layout.addLayout(s_bar)
        b_layout.addWidget(self.task_status_card)

        # MkConsole Terminal Output (uses native MonkeyQt styling & theme adaptation)
        self.console = MkConsole()
        self.console.setMinimumHeight(140)
        self.console.set_max_lines(3000)
        b_layout.addWidget(self.console, stretch=1)
        self.rv_splitter.addWidget(bottom_container)
        self.rv_splitter.setCollapsible(0, False)
        self.rv_splitter.setCollapsible(1, False)
        self.rv_splitter.setStretchFactor(0, 2)
        self.rv_splitter.setStretchFactor(1, 3)
        self.rv_splitter.setSizes([280, 440])

        right_layout.addWidget(self.rv_splitter, stretch=1)
        self.splitter.addWidget(right_pane)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 880])
        return self.splitter

    def _ensure_splitter_sizes(self):
        """Ensure comfortable height distribution (charts ~36%, console ~64%, reduced by ~40% from orig ~58%) on initial launch or view switch."""
        if hasattr(self, "rv_splitter") and self.rv_splitter:
            total_h = self.rv_splitter.height()
            if total_h > 350:
                top_h = max(200, int(total_h * 0.36))  # 原 58% 高度比例减少约 40% (0.58 * 0.6 ≈ 0.35~0.36)
                bottom_h = max(180, total_h - top_h)
                self.rv_splitter.setSizes([top_h, bottom_h])
            else:
                self.rv_splitter.setSizes([240, 360])
        if hasattr(self, "splitter") and self.splitter:
            total_w = self.splitter.width()
            if total_w > 600:
                left_w = min(320, max(260, int(total_w * 0.22)))
                self.splitter.setSizes([left_w, total_w - left_w])

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_splitter_sizes_initialized", False):
            self._splitter_sizes_initialized = True
            QTimer.singleShot(40, self._ensure_splitter_sizes)

    def _apply_console_style(self):
        pass

    def append_terminal_line(self, text: str):
        """Append raw terminal line preserving exact column spacing, alignment, and semantic colors."""
        if hasattr(self, "console") and self.console:
            is_dark = ThemeEngine.is_dark()
            append_terminal_line_to_console(self.console, text, is_dark=is_dark)


    def _wire_task_manager_signals(self):
        self.task_mgr.task_started.connect(self.on_task_started)
        self.task_mgr.task_log.connect(self.on_task_log)
        self.task_mgr.task_progress.connect(self.on_task_progress)
        self.task_mgr.task_finished.connect(self.on_task_finished)

    def load_project(self, project_id, preserve_console=False, defer_heavy=False):
        if not project_id:
            return
        self.project_id = project_id
        self.project_info = self.db.get_project_by_id(project_id)
        if not self.project_info:
            return

        # 1. Update Header (即时同步，耗时 < 0.5ms)
        self.lbl_title.setText(self.project_info["name"])
        self.lbl_task_type.setText(self.project_info["task_type"])
        summary = self.db.get_project_summary(project_id)
        desc = self.project_info.get("description") or "无描述"
        self.lbl_desc.setText(f"{desc} · 包含 {summary['task_count']} 个训练任务 · {summary['model_count']} 个产出模型")

        # 2. Check Tasks to Toggle Empty State vs Dashboard (即时同步，耗时 < 0.5ms)
        tasks = self.db.get_tasks(project_id=project_id)
        if not tasks:
            self.content_stack.setCurrentWidget(self.empty_widget)
            self.empty_upload.clear()
            self.header_card.setVisible(True)
        else:
            if self.content_stack.currentWidget() != getattr(self, "experiment_detail_view", None):
                self.content_stack.setCurrentWidget(self.dashboard_widget)
                self.header_card.setVisible(True)
                QTimer.singleShot(40, self._ensure_splitter_sizes)
            self._render_task_list(tasks, preserve_console=preserve_console, defer_heavy=defer_heavy)
            # Build the experiment inspector when it is opened, rather than
            # adding hundreds of hidden widgets to every theme update.

    def _warmup_experiment_detail_view(self):
        """Idle pre-warming: quietly instantiate ExperimentDetailView during idle event loop."""
        try:
            self._ensure_experiment_detail_view()
        except Exception:
            pass

    def _ensure_experiment_detail_view(self):
        if not hasattr(self, "experiment_detail_view") or self.experiment_detail_view is None:
            self.experiment_detail_view = ExperimentDetailView(self.db, self.task_mgr, parent=self)
            self.experiment_detail_view.back_requested.connect(self.show_project_dashboard)
            curr = self.content_stack.currentWidget()
            self.content_stack.addWidget(self.experiment_detail_view)
            # 确保预热加入 stack 时绝不影响当前正常展示的看板页面
            if curr:
                self.content_stack.setCurrentWidget(curr)
        return self.experiment_detail_view

    def open_task_detail(self, task_id):
        """Navigate smoothly into the comprehensive Experiment Detail View (Images 3, 4, 5)."""
        if not task_id:
            return
        task_id = int(task_id)
        self.current_task_id = task_id
        self.header_card.setVisible(False)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        exp_view = self._ensure_experiment_detail_view()
        # 1. 0 毫秒立即切换页面，用户屏幕即刻响应切入
        self.content_stack.setCurrentWidget(exp_view)
        exp_view.scroll_to_top()
        # 2. 详情页进入骨架屏流光加载态并异步读取数据
        exp_view.load_task(task_id)

    def show_project_dashboard(self):
        """Navigate back from Experiment Detail View to Project Dashboard."""
        self.header_card.setVisible(True)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(16)
        self.content_stack.setCurrentWidget(self.dashboard_widget)
        QTimer.singleShot(40, self._ensure_splitter_sizes)
        if self.project_id:
            self.load_project(self.project_id, preserve_console=True)

    def _render_task_list(self, tasks, preserve_console=False, defer_heavy=False):
        prev_task_id = self.current_task_id
        self.task_card.clear_items()

        # Preserve or initialize checked task IDs
        all_tids = set(t["id"] for t in tasks)
        if not hasattr(self, "checked_task_ids") or not self.checked_task_ids:
            self.checked_task_ids = set(all_tids)
        else:
            self.checked_task_ids = self.checked_task_ids & all_tids
            if not self.checked_task_ids:
                self.checked_task_ids = set(all_tids)

        selected_task_id = None
        for row, t in enumerate(tasks):
            tid = t["id"]
            color = ACADEMIC_COLORS[row % len(ACADEMIC_COLORS)]
            is_checked = (tid in self.checked_task_ids)
            task_name = t.get("task_name", f"task_{tid}")
            model_name = t.get("model_name", "-")
            prog = t.get("progress", 0.0)
            subtitle = f"{prog:.1f}% · {model_name}" if prog > 0 else model_name

            item = self.task_card.add_item({
                "id": tid,
                "title": task_name,
                "subtitle": subtitle,
                "dot_color": color,
                "checked": is_checked,
                "actions": [
                    {
                        "key": "delete",
                        "icon": "trash",
                        "tooltip": "删除该训练任务",
                        "danger": True
                    }
                ]
            })
            item.setMinimumHeight(56)

            if prev_task_id and tid == prev_task_id:
                selected_task_id = tid

        if tasks:
            if not selected_task_id:
                selected_task_id = tasks[0]["id"]
            self.select_task(selected_task_id, clear_console=(not preserve_console and selected_task_id != prev_task_id), defer_logs=defer_heavy)

        if defer_heavy:
            QTimer.singleShot(220, self._update_comparison_charts)
        else:
            self._update_comparison_charts()

    def _on_tasks_selection_changed(self, selected_ids: list):
        self.checked_task_ids = set(selected_ids)
        self._update_comparison_charts()

    def _on_task_item_clicked(self, item_id: Any):
        if item_id:
            self.select_task(int(item_id), clear_console=(int(item_id) != self.current_task_id))

    def _on_task_action_clicked(self, action_key: str, item_id: Any):
        if action_key == "delete" and item_id:
            self.handle_delete_task(int(item_id))

    def _update_comparison_charts(self):
        """Read results.csv from each checked task's run directory and render 3 core charts together."""
        prec_series = {}
        rec_series = {}
        map_series = {}

        if not self.project_id or not self.db:
            return

        tasks = self.db.get_tasks(project_id=self.project_id)
        task_dict = {t["id"]: t for t in tasks}

        checked_ids = sorted(list(getattr(self, "checked_task_ids", set())))
        for idx, tid in enumerate(checked_ids):
            t = task_dict.get(tid)
            if not t:
                continue

            color = ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)]
            task_name = t.get("task_name", f"task_{tid}")
            save_dir = t.get("save_dir", "")

            epochs = []
            prec_vals = []
            rec_vals = []
            map_vals = []

            if save_dir and os.path.isdir(save_dir):
                csv_file = os.path.join(save_dir, "results.csv")
                if os.path.isfile(csv_file):
                    try:
                        if pd is None:
                            import pandas as pd
                        df = pd.read_csv(csv_file)
                        df.columns = [c.strip() for c in df.columns]

                        if "epoch" in df.columns:
                            epochs = [int(e) for e in df["epoch"].tolist()]
                        else:
                            epochs = list(range(1, len(df) + 1))

                        for col in df.columns:
                            if "metrics/precision(B)" in col or "precision(B)" in col:
                                prec_vals = [float(v) for v in df[col].tolist()]
                                break

                        for col in df.columns:
                            if "metrics/recall(B)" in col or "recall(B)" in col:
                                rec_vals = [float(v) for v in df[col].tolist()]
                                break

                        for col in df.columns:
                            if "metrics/mAP50-95(B)" in col or "mAP50-95(B)" in col:
                                map_vals = [float(v) for v in df[col].tolist()]
                                break
                    except Exception:
                        pass

            if not epochs:
                metrics = self.db.get_task_metrics(tid)
                if metrics:
                    epochs = [m["epoch"] for m in metrics]
                    map_vals = [m.get("map50", 0.0) for m in metrics]

            if epochs:
                if prec_vals:
                    prec_series[task_name] = {"epochs": epochs, "values": prec_vals, "color": color}
                if rec_vals:
                    rec_series[task_name] = {"epochs": epochs, "values": rec_vals, "color": color}
                if map_vals:
                    map_series[task_name] = {"epochs": epochs, "values": map_vals, "color": color}

        if hasattr(self, "chart_precision"):
            self.chart_precision.update_series(prec_series)
        if hasattr(self, "chart_recall"):
            self.chart_recall.update_series(rec_series)
        if hasattr(self, "chart_map50_95"):
            self.chart_map50_95.update_series(map_series)

    def select_task(self, task_id, clear_console=True, defer_logs=False):
        switching_task = (self.current_task_id != task_id)
        self.current_task_id = task_id
        task = self.db.get_task_by_id(task_id)
        if not task:
            return

        # Visual state in task_card
        if hasattr(self, "task_card") and self.task_card:
            for it in self.task_card.get_items():
                it._apply_style()

        # Update Status Bar
        self.lbl_cur_task.setText(f"任务: {task['task_name']}")
        self.lbl_cur_model.setText(f"模型: {task['model_name']}")
        status = task.get("status", "WAITING")
        self.lbl_cur_status.setText(status)

        if status == "RUNNING":
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dbeafe; color: #1d4ed8;")
            self.btn_stop_task.setVisible(True)
            self.btn_resume_task.setVisible(False)
        elif status == "SUCCESS":
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dcfce7; color: #15803d;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)
        else:
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #fee2e2; color: #b91c1c;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)

        prog = task.get("progress", 0.0)
        self.progress_bar.percentage = int(prog)

        # Only clear/reload console when switching tasks or explicitly requested
        if switching_task or clear_console:
            self.console.clear()
            self.append_terminal_line(f"=== 切换至训练任务 [{task['task_name']}] (ID: {task_id}) ===")
            self.append_terminal_line(f"模型架构: {task['model_name']} | 状态: {status} | 进度: {prog:.1f}%")
            if task.get("save_dir"):
                self.append_terminal_line(f"保存路径: {task['save_dir']}")
            self.append_terminal_line("-" * 60)

            def _load_logs():
                save_dir = task.get("save_dir", "")
                if save_dir and os.path.isdir(save_dir):
                    log_file = os.path.join(save_dir, "train.log")
                    if os.path.isfile(log_file):
                        try:
                            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                                for line in lines[-500:]:
                                    self.append_terminal_line(line.rstrip('\r\n'))
                        except Exception:
                            pass

            if defer_logs:
                QTimer.singleShot(220, _load_logs)
            else:
                _load_logs()

    def open_train_dialog(self):
        if not self.project_info:
            return
        dlg = TrainCreateDialog(self.project_info, self.db, self.task_mgr, self)
        dlg.task_started.connect(self.on_new_task_started)
        main_win = self.window()
        if hasattr(main_win, "show_dialog_with_blur"):
            main_win.show_dialog_with_blur(dlg)
        else:
            dlg.exec()

    def on_new_task_started(self, task_id):
        MkMessage.success(self, "新训练任务已成功启动！")
        self.current_task_id = task_id
        self.console.clear()
        self.append_terminal_line(f"=== 新建训练任务已启动 (ID: {task_id}) ===")
        self.load_project(self.project_id, preserve_console=True)
        self.project_updated.emit(self.project_id)

    def handle_stop_task(self):
        if not self.current_task_id:
            return
        success, msg = self.task_mgr.stop_task(self.current_task_id)
        if success:
            MkMessage.info(self, "已向训练任务发送终止信号...")
            self.load_project(self.project_id, preserve_console=True)
        else:
            MkMessage.error(self, f"操作失败: {msg}")

    def handle_resume_task(self):
        if not self.current_task_id:
            return
        self.console.clear()
        self.console.info(f"正在准备启动/恢复任务 [{self.current_task_id}]...")
        success, msg = self.task_mgr.start_task(self.current_task_id)
        if success:
            MkMessage.success(self, "训练任务已恢复运行！")
            self.load_project(self.project_id, preserve_console=True)
        else:
            MkMessage.error(self, f"启动失败: {msg}")
            self.console.error(f"[ERROR] 启动失败: {msg}")

    def handle_export_model(self):
        if not self.current_task_id:
            return
        task = self.db.get_task_by_id(self.current_task_id)
        if not task:
            return

        # Check for best.pt in task save_dir
        save_dir = task.get("save_dir", "")
        weights_dir = os.path.join(save_dir, "weights") if save_dir else ""
        best_pt = os.path.join(weights_dir, "best.pt") if weights_dir else ""

        if not os.path.exists(best_pt):
            # Check save_dir directly
            if os.path.exists(os.path.join(save_dir, "best.pt")):
                best_pt = os.path.join(save_dir, "best.pt")
            else:
                MkMessage.warning(self, f"未在任务目录中检测到已产出的 best.pt 权重文件！")
                return

        model_info = {
            "name": f"{task['task_name']}_best",
            "best_pt_path": best_pt
        }
        dlg = ExportModelDialog(model_info, self)
        if dlg.exec():
            args = dlg.get_export_args()
            fmt = args["format"]
            try:
                from ultralytics import YOLO
                m = YOLO(best_pt)
                self.console.info(f"开始导出模型为 {fmt} 格式: {best_pt}")
                out_path = m.export(**args)
                self.console.success(f"导出成功！保存至: {out_path}")
                MkMessage.success(self, f"模型已成功导出为 {fmt} 格式！")
            except Exception as e:
                self.console.error(f"导出异常失败: {str(e)}")
                MkMessage.error(self, f"导出失败: {str(e)}")

    def handle_weights_dropped(self, file_paths):
        if not file_paths or not self.project_id:
            return
        for f in file_paths:
            if f.lower().endswith(".pt") and os.path.exists(f):
                m_name = os.path.splitext(os.path.basename(f))[0]
                arch = self.project_info.get("task_type", "yolo") if self.project_info else "yolo"
                self.db.add_model(
                    name=m_name,
                    task_id=None,
                    project_id=self.project_id,
                    model_arch=arch,
                    best_pt_path=f
                )
        MkMessage.success(self, f"已成功导入 {len(file_paths)} 个权重模型！")
        self.load_project(self.project_id)
        self.project_updated.emit(self.project_id)

    def handle_delete_task(self, task_id):
        task = self.db.get_task_by_id(task_id)
        task_name = task.get("task_name", f"任务 #{task_id}") if task else f"任务 #{task_id}"

        reply = QMessageBox.question(
            self,
            "确认删除任务",
            f"确定要删除训练任务 [{task_name}] 吗？\n该操作将删除该任务记录及其指标数据，且不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        if self.task_mgr and self.task_mgr.is_running(task_id):
            self.task_mgr.stop_task(task_id)

        self.db.delete_task(task_id)

        if self.current_task_id == task_id:
            self.current_task_id = None

        MkMessage.success(self, f"训练任务 [{task_name}] 已成功删除！")
        self.load_project(self.project_id)
        self.project_updated.emit(self.project_id)

    def handle_delete_project(self):
        if not self.project_id or not self.project_info:
            return
        reply = QMessageBox.question(
            self,
            "确认删除项目",
            f"确定要删除项目 [{self.project_info['name']}] 吗？\n该操作将同时清理项目下的训练历史记录。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            pid = self.project_id
            self.db.delete_project(pid)
            MkMessage.success(self, "项目已成功删除！")
            self.project_deleted.emit(pid)

    # ── TaskManager Slots ──
    @Slot(int)
    def on_task_started(self, task_id):
        if self.current_task_id == task_id:
            self.lbl_cur_status.setText("RUNNING")
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dbeafe; color: #1d4ed8;")
            self.btn_stop_task.setVisible(True)
            self.btn_resume_task.setVisible(False)
            self.append_terminal_line(f"[INFO] 训练任务 [{task_id}] 已在后台启动！")

    @Slot(int, str)
    def on_task_log(self, task_id, msg):
        if self.current_task_id == task_id:
            self.append_terminal_line(msg)

    @Slot(int, dict)
    def on_task_progress(self, task_id, metrics):
        if self.current_task_id == task_id:
            prog = metrics.get("progress", 0.0)
            self.progress_bar.percentage = int(prog)
            if hasattr(self, "checked_task_ids") and task_id in self.checked_task_ids:
                self._update_comparison_charts()

    @Slot(int, bool, str)
    def on_task_finished(self, task_id, success, msg):
        if self.current_task_id == task_id:
            if success:
                self.lbl_cur_status.setText("SUCCESS")
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dcfce7; color: #15803d;")
            else:
                self.lbl_cur_status.setText("STOPPED")
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #fee2e2; color: #b91c1c;")
                self.append_terminal_line(f"[FAILED] 任务执行失败: {msg}")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)
            self.load_project(self.project_id, preserve_console=True)

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            if hasattr(self, 'lbl_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 22px; font-weight: bold; color: #f4f4f5;")
            if hasattr(self, 'empty_title'):
                _MkThemeEngine.apply_style_sheet(self.empty_title, "font-size: 18px; font-weight: bold; color: #f4f4f5;")
            if hasattr(self, 'lbl_cur_task'):
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 14px; color: #f4f4f5;")
            if hasattr(self, 'lbl_metrics_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_metrics_title, "font-size: 14px; font-weight: bold; color: #f4f4f5;")
        else:
            if hasattr(self, 'lbl_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 22px; font-weight: bold; color: #1e293b;")
            if hasattr(self, 'empty_title'):
                _MkThemeEngine.apply_style_sheet(self.empty_title, "font-size: 18px; font-weight: bold; color: #1e293b;")
            if hasattr(self, 'lbl_cur_task'):
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 14px; color: #1e293b;")
            if hasattr(self, 'lbl_metrics_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_metrics_title, "font-size: 14px; font-weight: 700; color: #0f172a;")


        # Sync splitters styling (clean invisible handles for seamless card spacing)
        spl_qss = """
            QSplitter {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: transparent;
                border: none;
            }
            QSplitter::handle:hover {
                background: transparent;
                border: none;
            }
            QSplitter::handle:pressed {
                background: transparent;
                border: none;
            }
        """
        if hasattr(self, "splitter"):
            _MkThemeEngine.apply_style_sheet(self.splitter, spl_qss)
        if hasattr(self, "rv_splitter"):
            _MkThemeEngine.apply_style_sheet(self.rv_splitter, spl_qss)

        if hasattr(self, 'chart_precision'):
            self.chart_precision.set_theme(is_dark)
        if hasattr(self, 'chart_recall'):
            self.chart_recall.set_theme(is_dark)
        if hasattr(self, 'chart_map50_95'):
            self.chart_map50_95.set_theme(is_dark)
        pass


