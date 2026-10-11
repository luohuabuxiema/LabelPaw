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
import csv
from typing import Any
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSplitter, QListWidget, QListWidgetItem,
                               QStackedWidget, QFrame, QToolButton,
                               QCheckBox, QSizePolicy, QScrollArea)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

from monkeyqt import (MkCard, MkButton, MkProgressBar, MkConsole, 
                      MkUpload, MkMessage, MkQWidget, ThemeEngine,
                      MkSelectCard, MkSelectItem, MkItemActionButton, MkComboBox, MkMessageBox)
from monkeyqt_icons import (PhPlus, PhPlay, PhStop, PhExport, 
                            PhArrowClockwise, PhTrash, PhFolder, PhCaretRight)

from TrainingPlatform.ui.components.thesis_chart import ThesisChartCard, ACADEMIC_COLORS
from TrainingPlatform.ui.components.terminal_formatter import (apply_terminal_console_style,
                                                               append_terminal_line_to_console)
from TrainingPlatform.ui.views.experiment_detail_view import ExperimentDetailView
from TrainingPlatform.ui.views.dialogs.train_create_dialog import TrainCreateDialog
from TrainingPlatform.ui.views.model_view import ExportModelDialog
from TrainingPlatform.core.checkpoint_metadata import (
    TASK_LABELS, config_dict, infer_task, metric_keys, normalize_history,
    load_checkpoint_metadata, experiment_data, model_experiment, checkpoint_path,
)
from TrainingPlatform.ui.components.custom_tooltip import enable_component_tooltips


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
        self._records = {}
        self._record_data = {}
        self._load_generation = 0
        self.checked_task_ids = None
        self._group_headers = []
        enable_component_tooltips(self)

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

        row_title.addStretch()

        v_info.addLayout(row_title)

        self.lbl_desc = QLabel("加载中...")
        _MkThemeEngine.apply_style_sheet(self.lbl_desc, "color: #64748b; font-size: 13px;")
        v_info.addWidget(self.lbl_desc)

        h_layout.addLayout(v_info, stretch=1)

        # Right Action Buttons
        self.btn_new_train = MkButton("新建训练任务", btn_type="primary")
        self.btn_new_train.setIcon(PhPlus.bold(size=18))
        _MkThemeEngine.apply_style_sheet(self.btn_new_train, "font-weight: bold; padding: 8px 16px;")
        self.btn_new_train.clicked.connect(self.open_train_dialog)
        h_layout.addWidget(self.btn_new_train)

        self.btn_refresh = MkButton("", btn_type="default")
        self.btn_refresh.setIcon(PhArrowClockwise.regular(size=18))
        self.btn_refresh.setToolTip("刷新项目状态")
        self.btn_refresh.setFixedSize(36, 36)
        self.btn_refresh.clicked.connect(lambda: self.load_project(self.project_id))
        h_layout.addWidget(self.btn_refresh)

        self.btn_delete_proj = MkButton("", btn_type="default")
        self.btn_delete_proj.setIcon(PhTrash.regular(size=18))
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

        self.metric_scope = MkComboBox()
        for label, value in (("任务主指标", "auto"), ("检测框 (B)", "B"), ("分割掩膜 (M)", "M"), ("关键点 (P)", "P")):
            self.metric_scope.addItem(label, value)
        self.metric_scope.currentIndexChanged.connect(lambda _: self._update_comparison_charts())
        metrics_hdr.addWidget(self.metric_scope)
        self.lbl_metrics_hint = QLabel("模型内嵌训练历史 · 训练中实时更新 · 悬停查看数值")
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
        if project_id != self.project_id:
            self.checked_task_ids = None
            self.current_task_id = None
            self._preferred_record_id = None
            self._record_data = {}
            self.content_stack.setCurrentWidget(self.dashboard_widget)
            self.main_layout.setContentsMargins(24, 20, 24, 20)
            self.main_layout.setSpacing(16)
        self._load_generation += 1
        self.project_id = project_id
        self.project_info = self.db.get_project_by_id(project_id)
        if not self.project_info:
            return

        # 1. Update Header (即时同步，耗时 < 0.5ms)
        self.lbl_title.setText(self.project_info["name"])
        summary = self.db.get_project_summary(project_id)
        desc = self.project_info.get("description") or "无描述"
        self.lbl_desc.setText(f"{desc} · 包含 {summary['task_count']} 个训练任务 · {summary['model_count']} 个产出模型")

        # 2. Check Tasks to Toggle Empty State vs Dashboard (即时同步，耗时 < 0.5ms)
        tasks = self.db.get_tasks(project_id=project_id)
        models = self.db.get_models(project_id=project_id)
        task_ids = {t["id"] for t in tasks}
        records = tasks + [model_experiment(m) for m in models if m.get("task_id") not in task_ids]
        self._records = {t["id"]: t for t in records}
        if not records:
            self.content_stack.setCurrentWidget(self.empty_widget)
            self.empty_upload.clear()
            self.header_card.setVisible(True)
        else:
            if self.content_stack.currentWidget() != getattr(self, "experiment_detail_view", None):
                self.content_stack.setCurrentWidget(self.dashboard_widget)
                self.header_card.setVisible(True)
                QTimer.singleShot(40, self._ensure_splitter_sizes)
            self._render_task_list(records, preserve_console=preserve_console, defer_heavy=defer_heavy)
            # Build the experiment inspector when it is opened, rather than
            # adding hundreds of hidden widgets to every theme update.
            generation = self._load_generation
            from TrainingPlatform.core.env_manager import EnvironmentManager
            python_exe = EnvironmentManager.get_instance().get_selected_python()

            def fetch():
                data = {t["id"]: experiment_data(t, models, python_exe) for t in records}
                for model in models:
                    record_id = model.get("task_id") if model.get("task_id") in task_ids else -model["id"]
                    parsed = data.get(record_id, {})
                    if parsed.get("signature"):
                        self.db.update_model_metadata(model["id"], parsed)
                return data

            def loaded(data):
                if generation != self._load_generation or project_id != self.project_id:
                    return
                self._record_data = data
                for record in records:
                    record["_experiment_data"] = data.get(record["id"], {})
                self._render_task_list(records, preserve_console=True)
                errors = [d["error"] for d in data.values() if d.get("error")]
                if errors:
                    self.console.warning(errors[0])

            self.run_async(fetch, on_success=loaded, auto_loading=False)

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
        if task_id < 0:
            exp_view.load_model(-task_id)
        else:
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
        collapsed = {header.group_id: header.is_collapsed() for header in self._group_headers}
        self.task_card.blockSignals(True)
        self.task_card.clear_items()
        for header in self._group_headers:
            self.task_card.list_layout.removeWidget(header)
            self.task_card._groups.pop(header.group_id, None)
            header.deleteLater()
        self._group_headers = []
        all_ids = {t["id"] for t in tasks}
        self.checked_task_ids = all_ids if self.checked_task_ids is None else self.checked_task_ids & all_ids
        groups = {}
        for record in tasks:
            data = self._record_data.get(record["id"], {})
            kind = data.get("task") or infer_task(record.get("model_name"), config_dict(record.get("config_json")))
            groups.setdefault(kind, []).append(record)
        for kind in TASK_LABELS:
            if kind not in groups:
                continue
            self._group_headers.append(self.task_card.add_group(kind, TASK_LABELS[kind], collapsed=collapsed.get(kind, False)))
            for record in groups[kind]:
                tid = record["id"]
                data = self._record_data.get(tid, {})
                color = ACADEMIC_COLORS[sorted(all_ids).index(tid) % len(ACADEMIC_COLORS)]
                subtitle = ""
                if record.get("status") == "RUNNING":
                    subtitle = f"{record.get('progress', 0):.1f}% · 训练中"
                else:
                    primary = metric_keys(kind)[0 if kind == "classify" else 2]
                    value = data.get("metrics", {}).get(primary)
                    if value is not None:
                        subtitle = f"{value * 100:.1f}%"
                    elif not record.get("is_imported"):
                        subtitle = record.get("model_name", "")
                size = data.get("file_size")
                if size is None:
                    path = checkpoint_path(record)
                    if path:
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            pass
                if size is not None:
                    size_text = f"{size} B"
                    for unit, divisor in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
                        if size >= divisor:
                            size_text = f"{size / divisor:.1f} {unit}"
                            break
                    subtitle = " · ".join(part for part in (subtitle, size_text) if part)
                self.task_card.add_item({
                    "id": tid, "title": record["task_name"], "subtitle": subtitle,
                    "dot_color": color, "checked": tid in self.checked_task_ids, "group_id": kind,
                    "actions": [{"key": "delete", "icon": "trash", "tooltip": "移除模型记录" if tid < 0 else "删除该训练任务", "danger": True}],
                })
        self.task_card.blockSignals(False)
        if tasks:
            preferred = getattr(self, "_preferred_record_id", None)
            selected = preferred if preferred in all_ids else prev_task_id if prev_task_id in all_ids else tasks[0]["id"]
            self._preferred_record_id = None
            self.select_task(selected, clear_console=not preserve_console and selected != prev_task_id, defer_logs=defer_heavy)
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
            if int(item_id) < 0:
                if MkMessageBox.confirm(self, "移除模型", "移除该模型记录？模型文件会保留。", danger=True, ok_text="移除", cancel_text="取消"):
                    self.db.delete_model(-int(item_id))
                    self.load_project(self.project_id)
                    self.project_updated.emit(self.project_id)
            else:
                self.handle_delete_task(int(item_id))

    def _update_comparison_charts(self):
        series = [{}, {}, {}]
        labels = [set(), set(), set()]
        scope = self.metric_scope.currentData() or "auto"
        all_ids = sorted(self._records)
        for tid in sorted(self.checked_task_ids or ()):
            record = self._records.get(tid)
            if not record:
                continue
            data = self._record_data.get(tid, {})
            history = data.get("history", {})
            folder = record.get("save_dir") or ""
            if record.get("status") == "RUNNING" or not history:
                csv_path = os.path.join(folder, "results.csv")
                if folder and os.path.isfile(csv_path):
                    try:
                        with open(csv_path, encoding="utf-8-sig", newline="") as source:
                            rows = list(csv.DictReader(source))
                        if rows:
                            history = normalize_history({key: [r[key] for r in rows] for key in rows[0]})
                    except (OSError, ValueError):
                        pass
            kind = data.get("task") or infer_task(record.get("model_name"), config_dict(record.get("config_json")))
            keys = metric_keys(kind, scope)
            if not history and tid > 0 and kind in ("detect", "obb") and scope in ("auto", "B"):
                rows = self.db.get_task_metrics(tid)
                if rows:
                    history = {"epoch": [r["epoch"] for r in rows]}
                    for key, field in zip(keys, ("precision", "recall", "map50_95")):
                        history[key] = [r[field] for r in rows]
            for i, key in enumerate(keys):
                if not key:
                    continue
                labels[i].add(key.removeprefix("metrics/"))
                values = history.get(key, [])
                epochs = history.get("epoch", list(range(1, len(values) + 1)))
                if not values or len(epochs) != len(values):
                    continue
                name = record["task_name"]
                if name in series[i]:
                    name += f" ({tid})"
                series[i][name] = {"epochs": epochs, "values": values,
                                   "color": ACADEMIC_COLORS[all_ids.index(tid) % len(ACADEMIC_COLORS)]}
        for i, chart in enumerate((self.chart_precision, self.chart_recall, self.chart_map50_95)):
            title = " / ".join(sorted(labels[i])) if labels[i] else ("Precision", "Recall", "mAP50-95")[i]
            chart.display_title = title
            chart.lbl_title.setText(title)
            chart.setVisible(i != 2 or bool(labels[i]) or not self.checked_task_ids)
            chart.update_series(series[i])
        count = 2 if self.checked_task_ids and not labels[2] else 3
        self.lbl_metrics_title.setText(f"核心评估指标对比 ({count}/{count})")

    def select_task(self, task_id, clear_console=True, defer_logs=False):
        switching_task = (self.current_task_id != task_id)
        self.current_task_id = task_id
        task = self._records.get(task_id) or self.db.get_task_by_id(task_id)
        if not task:
            return

        # Visual state in task_card
        if hasattr(self, "task_card") and self.task_card:
            for it in self.task_card.get_items():
                it._apply_style()

        # Update Status Bar
        self.lbl_cur_task.setText(f"{'模型' if task.get('is_imported') else '任务'}: {task['task_name']}")
        self.lbl_cur_model.setText(f"模型: {task['model_name']}")
        status = task.get("status", "WAITING")
        self.lbl_cur_status.setText("已导入" if status == "IMPORTED" else status)

        if status == "RUNNING":
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dbeafe; color: #1d4ed8;")
            self.btn_stop_task.setVisible(True)
            self.btn_resume_task.setVisible(False)
        elif status in ("SUCCESS", "COMPLETED"):
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #dcfce7; color: #15803d;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)
        elif status == "IMPORTED":
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #e2e8f0; color: #475569;")
        else:
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; background: #fee2e2; color: #b91c1c;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)

        if task.get("is_imported"):
            self.btn_resume_task.hide()
            self.btn_stop_task.hide()
        prog = task.get("progress", 0.0)
        self.progress_bar.setVisible(not task.get("is_imported"))
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
        task = self._records.get(self.current_task_id) or self.db.get_task_by_id(self.current_task_id)
        if not task:
            return

        best_pt = checkpoint_path(task)
        if not best_pt:
            MkMessage.warning(self, "未找到可导出的模型权重")
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
        from TrainingPlatform.core.env_manager import EnvironmentManager
        python_exe = EnvironmentManager.get_instance().get_selected_python()
        project_id = self.project_id
        self.mini_upload.clear()
        self.empty_upload.clear()

        def fetch():
            imported, errors = [], []
            for path in dict.fromkeys(file_paths):
                try:
                    if not path.lower().endswith(".pt") or not os.path.isfile(path):
                        raise ValueError("请选择有效的 .pt 模型文件")
                    metadata = load_checkpoint_metadata(path, python_exe)
                    model_id = self.db.import_model(project_id, path, metadata)
                    model = self.db.get_model_by_id(model_id)
                    linked_task = model.get("task_id")
                    imported.append(linked_task if linked_task and self.db.get_task_by_id(linked_task) else -model_id)
                except Exception as exc:
                    errors.append(f"{os.path.basename(path)}: {exc}")
            return imported, errors

        def loaded(result):
            imported, errors = result
            if project_id == self.project_id:
                if self.checked_task_ids is not None:
                    self.checked_task_ids.update(imported)
                if imported:
                    self._preferred_record_id = imported[-1]
                self.load_project(project_id)
            self.project_updated.emit(project_id)
            if imported:
                MkMessage.success(self, f"已导入 {len(imported)} 个模型")
            if errors:
                MkMessage.error(self, "\n".join(errors))

        self.run_async(fetch, on_success=loaded, auto_loading=False)

    def handle_delete_task(self, task_id):
        task = self.db.get_task_by_id(task_id)
        task_name = task.get("task_name", f"任务 #{task_id}") if task else f"任务 #{task_id}"

        if not MkMessageBox.confirm(
            self,
            "确认删除任务",
            f"确定要删除训练任务 [{task_name}] 吗？<br>该操作将删除该任务记录及其指标数据，且不可恢复。",
            danger=True,
            ok_text="删除任务",
            cancel_text="取消",
        ):
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
        if MkMessageBox.confirm(
            self,
            "确认删除项目",
            f"确定要删除项目 [{self.project_info['name']}] 吗？<br>该操作将同时清理项目下的训练历史记录。",
            danger=True,
            ok_text="删除项目",
            cancel_text="取消",
        ):
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


