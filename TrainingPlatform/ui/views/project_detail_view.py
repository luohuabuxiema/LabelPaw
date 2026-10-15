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
                               QCheckBox, QSizePolicy, QScrollArea, QApplication)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

from monkeyqt import (MkCard, MkButton, MkProgressBar, 
                      MkUpload, MkMessage, MkQWidget, ThemeEngine,
                      MkSelectCard, MkSelectItem, MkItemActionButton, MkComboBox, MkMessageBox,
                      MkSkeleton, MkSkeletonBlock, MkChartDashboard, MkChartGroup, MkLineChart, MkSeries)
from monkeyqt.components.layout.scroll_area import MkScrollArea
from monkeyqt.components.data.line_chart import DEFAULT_CHART_COLORS
from monkeyqt_icons import (PhPlus, PhPlay, PhStop, PhExport, 
                            PhArrowClockwise, PhTrash, PhFolder, PhCaretRight)

ACADEMIC_COLORS = DEFAULT_CHART_COLORS
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
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        main_layout = self.main_layout

        # ── 1. Top Project Header (integrated into scrollable dashboard) ──
        self.header_card = QWidget()
        self.header_card.setObjectName("ProjectHeaderCard")
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

        # ── 2. Content Stack (Skeleton State vs Active Dashboard vs Empty State vs Experiment Detail) ──
        self.content_stack = QStackedWidget()

        # State S: Skeleton Loading State (Fast 0ms shimmer placeholder)
        self.skeleton_widget = self._create_skeleton_widget()
        self.content_stack.addWidget(self.skeleton_widget)

        # State B: Dashboard Master-Visualizer Widget (Like Ultralytics Figure 1 & 2)
        self.dashboard_widget = self._create_dashboard_widget()
        self.content_stack.addWidget(self.dashboard_widget)

        # State A: Empty State Widget (Like Ultralytics Figure 4)
        self.empty_widget = self._create_empty_state_widget()
        self.content_stack.addWidget(self.empty_widget)

        # State C: Experiment Detail View (Lazy initialized on first demand)
        self.experiment_detail_view = None

        main_layout.addWidget(self.content_stack, stretch=1)

    def _create_skeleton_widget(self):
        """
        Creates a dedicated skeleton screen widget mirroring the ProjectDetailView layout:
        - Top Header skeleton
        - Left Pane (width ~280px): Task list skeleton with header, select-all pill, 5 simulated task items, upload box.
        - Right Pane: Top task status bar skeleton + responsive 2D chart cards grid skeleton.
        Uses MonkeyQt MkSkeleton shimmering wave animation inside a full-page scroll container.
        """
        skeleton_scroll = MkScrollArea(parent=self)
        skeleton_scroll.setObjectName("SkeletonScrollArea")
        skeleton_scroll.setFrameShape(QFrame.NoFrame)
        skeleton_scroll.setWidgetResizable(True)
        skeleton_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        skeleton_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _MkThemeEngine.apply_style_sheet(skeleton_scroll, "background: transparent; border: none;")

        skeleton_content = QWidget()
        skeleton_content.setObjectName("SkeletonContent")
        s_layout = QVBoxLayout(skeleton_content)
        s_layout.setContentsMargins(24, 20, 16, 20)
        s_layout.setSpacing(16)

        # ── Top Header Skeleton ──
        header_card = MkCard(title="")
        header_card_layout = header_card.content_layout
        header_card_layout.setContentsMargins(14, 12, 14, 12)
        header_blocks = [
            MkSkeletonBlock(x=12, y=10, width=160, height=24, radius=5),
            MkSkeletonBlock(x=12, y=42, width=320, height=14, radius=4),
            MkSkeletonBlock(x=0.82, y=14, width=0.15, height=32, radius=6),
        ]
        self.header_skeleton = MkSkeleton(blocks=header_blocks)
        self.header_skeleton.setFixedHeight(54)
        header_card_layout.addWidget(self.header_skeleton)
        s_layout.addWidget(header_card)

        spl = QSplitter(Qt.Horizontal)
        spl.setHandleWidth(12)
        spl.setChildrenCollapsible(False)
        spl.setCollapsible(0, False)
        spl.setCollapsible(1, False)
        spl_qss = """
            QSplitter {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: transparent;
                border: none;
            }
        """
        _MkThemeEngine.apply_style_sheet(spl, spl_qss)

        # ── Left Pane Skeleton: Mirrors MkSelectCard & Mini Upload ──
        left_pane = MkQWidget(role="transparent")
        left_pane.setMinimumWidth(260)
        left_pane.setMaximumWidth(360)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_card = MkCard(title="")
        left_card_layout = left_card.content_layout
        left_card_layout.setContentsMargins(12, 14, 12, 14)
        left_card_layout.setSpacing(10)

        left_skeleton_blocks = [
            # Header Title & Select All pill
            MkSkeletonBlock(x=12, y=10, width=0.55, height=20, radius=4),
            MkSkeletonBlock(x=12, y=42, width=0.92, height=28, radius=6),
            # 5 simulated task items (dot + title + subtitle)
            MkSkeletonBlock(x=12, y=86, width=12, height=12, radius=6, shape="circle"),
            MkSkeletonBlock(x=32, y=84, width=0.55, height=15, radius=4),
            MkSkeletonBlock(x=32, y=104, width=0.68, height=12, radius=4),

            MkSkeletonBlock(x=12, y=130, width=12, height=12, radius=6, shape="circle"),
            MkSkeletonBlock(x=32, y=128, width=0.60, height=15, radius=4),
            MkSkeletonBlock(x=32, y=148, width=0.50, height=12, radius=4),

            MkSkeletonBlock(x=12, y=174, width=12, height=12, radius=6, shape="circle"),
            MkSkeletonBlock(x=32, y=172, width=0.48, height=15, radius=4),
            MkSkeletonBlock(x=32, y=192, width=0.72, height=12, radius=4),

            MkSkeletonBlock(x=12, y=218, width=12, height=12, radius=6, shape="circle"),
            MkSkeletonBlock(x=32, y=216, width=0.52, height=15, radius=4),
            MkSkeletonBlock(x=32, y=236, width=0.58, height=12, radius=4),

            MkSkeletonBlock(x=12, y=262, width=12, height=12, radius=6, shape="circle"),
            MkSkeletonBlock(x=32, y=260, width=0.45, height=15, radius=4),
            MkSkeletonBlock(x=32, y=280, width=0.64, height=12, radius=4),

            # Mini upload dropzone placeholder
            MkSkeletonBlock(x=12, y=320, width=0.92, height=88, radius=8),
        ]
        self.left_skeleton = MkSkeleton(blocks=left_skeleton_blocks)
        self.left_skeleton.setMinimumHeight(440)
        left_card_layout.addWidget(self.left_skeleton)
        left_layout.addWidget(left_card, stretch=1)
        spl.addWidget(left_pane)

        # ── Right Pane Skeleton: Status Bar + Charts Grid ──
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # 1. Top Status Bar Skeleton
        status_card = MkCard(title="")
        status_card_layout = status_card.content_layout
        status_card_layout.setContentsMargins(14, 8, 14, 8)
        status_blocks = [
            MkSkeletonBlock(x=12, y=13, width=130, height=18, radius=4),
            MkSkeletonBlock(x=154, y=13, width=64, height=18, radius=4),
            MkSkeletonBlock(x=228, y=13, width=110, height=18, radius=4),
            MkSkeletonBlock(x=350, y=17, width=0.32, height=10, radius=5),
            MkSkeletonBlock(x=0.76, y=10, width=0.11, height=26, radius=6),
            MkSkeletonBlock(x=0.89, y=10, width=0.09, height=26, radius=6),
        ]
        self.status_skeleton = MkSkeleton(blocks=status_blocks)
        self.status_skeleton.setFixedHeight(46)
        status_card_layout.addWidget(self.status_skeleton)
        right_layout.addWidget(status_card)

        # 2. Charts Cards Grid Skeleton directly in layout (NO inner nested QScrollArea!)
        chart_blocks = [
            # Group 1: Metrics Title Pill
            MkSkeletonBlock(x=12, y=10, width=140, height=22, radius=5),
            # 6 Cards Grid (2 rows x 3 cols)
            MkSkeletonBlock(x=0.01, y=44, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.34, y=44, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.67, y=44, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.01, y=248, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.34, y=248, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.67, y=248, width=0.31, height=190, radius=10),

            # Group 2: Loss Title Pill
            MkSkeletonBlock(x=12, y=462, width=110, height=22, radius=5),
            # 3 Cards Grid (1 row x 3 cols)
            MkSkeletonBlock(x=0.01, y=496, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.34, y=496, width=0.31, height=190, radius=10),
            MkSkeletonBlock(x=0.67, y=496, width=0.31, height=190, radius=10),
        ]
        self.charts_skeleton = MkSkeleton(blocks=chart_blocks)
        self.charts_skeleton.setMinimumHeight(700)
        right_layout.addWidget(self.charts_skeleton, stretch=1)

        spl.addWidget(right_pane)

        spl.setStretchFactor(0, 0)
        spl.setStretchFactor(1, 1)
        spl.setSizes([280, 1100])
        self.skeleton_splitter = spl

        s_layout.addWidget(spl, stretch=1)
        skeleton_scroll.setWidget(skeleton_content)
        return skeleton_scroll

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

        # ── 1. Create the unified full-page scroll area (matching Ultralytics Platform) ──
        self.dashboard_scroll = MkScrollArea(parent=self)
        self.dashboard_scroll.setObjectName("DashboardScrollArea")
        self.dashboard_scroll.setFrameShape(QFrame.NoFrame)
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dashboard_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _MkThemeEngine.apply_style_sheet(self.dashboard_scroll, "background: transparent; border: none;")

        self.dashboard_content = QWidget()
        self.dashboard_content.setObjectName("DashboardContent")
        self.dashboard_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        d_layout = QVBoxLayout(self.dashboard_content)
        d_layout.setContentsMargins(24, 20, 16, 20)
        d_layout.setSpacing(16)

        # Top Project Header placed at the top of the scrollable page (flows naturally)
        d_layout.addWidget(self.header_card)

        # ── 2. Horizontal Splitter: Left Task List, Right Visualizer Dashboard ──
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

        # ── Left Pane: Models / Tasks Picker ──
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
        left_layout.addWidget(self.task_card)

        # Mini drop area for existing .pt models
        lbl_drop = QLabel("快速添加已有权重 (.pt):")
        _MkThemeEngine.apply_style_sheet(lbl_drop, "color: #94a3b8; font-size: 11px;")
        left_layout.addWidget(lbl_drop)

        self.mini_upload = MkUpload(accept_filters=[".pt", "*.pt"], multiple=True, max_size_mb=2048)
        self.mini_upload.filesSelected.connect(self.handle_weights_dropped)
        left_layout.addWidget(self.mini_upload)
        left_layout.addStretch()

        self.splitter.addWidget(left_pane)

        # ── Right Pane: Visualizer Dashboard ──
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        # 1. Selected Task Status & Actions Bar (compact card)
        self.task_status_card = MkCard(title="")
        sc_layout = self.task_status_card.content_layout
        sc_layout.setContentsMargins(14, 8, 14, 8)

        s_bar = QHBoxLayout()
        s_bar.setSpacing(10)

        self.lbl_cur_task = QLabel("任务: -")
        _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 13px; color: #1e293b;")
        s_bar.addWidget(self.lbl_cur_task)

        self.lbl_cur_status = QLabel("待定")
        _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #e2e8f0; color: #475569;")
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
        right_layout.addWidget(self.task_status_card)

        # 2. Add Modern MonkeyQt Chart Dashboard (60 FPS Native QPainter, 0 Matplotlib overhead)
        self.chart_dashboard = MkChartDashboard(parent=self, scrollable=False, spacing=14)
        right_layout.addWidget(self.chart_dashboard)

        # ── Group 1: Metrics (Image 2: Metrics (6/8)) ──
        self.group_metrics = self.chart_dashboard.add_group(
            title="Metrics",
            columns=3,
            is_expanded=True,
        )
        metric_configs = [
            ("precision(B)", "precision(B)", True),
            ("recall(B)", "recall(B)", True),
            ("mAP50-95(B)", "mAP50-95(B)", True),
            ("precision(M)", "precision(M)", True),
            ("recall(M)", "recall(M)", True),
            ("mAP50-95(M)", "mAP50-95(M)", True),
            ("mAP50(B)", "mAP50(B)", False),
            ("mAP50(M)", "mAP50(M)", False),
        ]
        for cid, title, vis in metric_configs:
            self.group_metrics.add_chart(
                chart_id=cid,
                title=title,
                default_visible=vis,
                x_label="Epoch",
                smooth=True,
                area=True,
                show_legend=False,
                chart_min_height=220,
            )
        self.group_metrics.visibilityChanged.connect(lambda *_: self._schedule_charts_update(delay_ms=15))

        # ── Group 2: Loss (Image 2 & 5: Loss (5/5) with Train solid / Val dashed) ──
        self.group_loss = self.chart_dashboard.add_group(
            title="Loss",
            columns=3,
            is_expanded=True,
            enable_series_toggle=True,
            series_options=["Train (solid)", "Val (dashed)"],
        )
        loss_configs = [
            ("box_loss", "box_loss", True),
            ("seg_loss", "seg_loss", True),
            ("cls_loss", "cls_loss", True),
            ("dfl_loss", "dfl_loss", True),
            ("sem_loss", "sem_loss", True),
        ]
        for cid, title, vis in loss_configs:
            self.group_loss.add_chart(
                chart_id=cid,
                title=title,
                default_visible=vis,
                x_label="Epoch",
                smooth=True,
                area=True,
                show_legend=False,
                chart_min_height=220,
            )
        self.group_loss.visibilityChanged.connect(lambda *_: self._schedule_charts_update(delay_ms=15))
        self.group_loss.seriesToggled.connect(lambda *_: self._schedule_charts_update(delay_ms=15))

        # ── Group 3: Learning Rate (Image 3 & 4: Learning Rate (4/4)) ──
        self.group_lr = self.chart_dashboard.add_group(
            title="Learning Rate",
            columns=3,
            is_expanded=True,
        )
        lr_configs = [
            ("pg0", "pg0", True),
            ("pg1", "pg1", True),
            ("pg2", "pg2", True),
            ("pg3", "pg3", True),
        ]
        for cid, title, vis in lr_configs:
            self.group_lr.add_chart(
                chart_id=cid,
                title=title,
                default_visible=vis,
                x_label="Epoch",
                smooth=True,
                area=True,
                show_legend=False,
                chart_min_height=220,
            )
        self.group_lr.visibilityChanged.connect(lambda *_: self._schedule_charts_update(delay_ms=15))

        right_layout.addStretch()

        self.splitter.addWidget(right_pane)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 1100])

        d_layout.addWidget(self.splitter, stretch=1)

        self.dashboard_scroll.setWidget(self.dashboard_content)
        self.dashboard_scroll.verticalScrollBar().valueChanged.connect(self._on_charts_scroll_scrolled)

        return self.dashboard_scroll

    def _ensure_splitter_sizes(self):
        if hasattr(self, "splitter") and self.splitter:
            total_w = self.splitter.width()
            if total_w > 600:
                left_w = min(320, max(260, int(total_w * 0.22)))
                self.splitter.setSizes([left_w, total_w - left_w])
        if hasattr(self, "skeleton_splitter") and self.skeleton_splitter:
            total_w = self.skeleton_splitter.width()
            if total_w > 600:
                left_w = min(320, max(260, int(total_w * 0.22)))
                self.skeleton_splitter.setSizes([left_w, total_w - left_w])

    def _on_charts_scroll_scrolled(self):
        """Scroll notification handler for dashboard scrollbar."""
        pass

    @property
    def chart_map50_95(self):
        """Backward compatibility accessor for test suites and parent views."""
        if hasattr(self, "group_metrics") and self.group_metrics:
            card_m = self.group_metrics.get_chart("mAP50-95(M)")
            card_b = self.group_metrics.get_chart("mAP50-95(B)")
            if card_m and getattr(card_m, "series_list", None):
                return card_m
            if card_b and getattr(card_b, "series_list", None):
                return card_b
            return card_m or card_b
        return None

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_splitter_sizes_initialized", False):
            self._splitter_sizes_initialized = True
            QTimer.singleShot(40, self._ensure_splitter_sizes)

    def _apply_console_style(self):
        pass

    def append_terminal_line(self, text: str):
        pass

    def _wire_task_manager_signals(self):
        self.task_mgr.task_started.connect(self.on_task_started)
        self.task_mgr.task_log.connect(self.on_task_log)
        self.task_mgr.task_progress.connect(self.on_task_progress)
        self.task_mgr.task_finished.connect(self.on_task_finished)

    def load_project(self, project_id, preserve_console=False, defer_heavy=False, show_skeleton=True):
        if not project_id:
            return
        is_project_switch = (project_id != self.project_id)
        self.project_id = project_id
        self._load_generation += 1
        generation = self._load_generation

        if is_project_switch:
            self.checked_task_ids = None
            self.current_task_id = None
            self._preferred_record_id = None
            self._record_data = {}
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(0)
            self.header_card.setVisible(True)

            if show_skeleton and hasattr(self, "skeleton_widget"):
                # 0ms 瞬间切换到骨架屏，丝滑不卡顿
                self.content_stack.setCurrentWidget(self.skeleton_widget)
                p_cached = self.db.get_project_by_id(project_id)
                pname = p_cached["name"] if p_cached else "项目看板"
                self.lbl_title.setText(pname)
                self.lbl_desc.setText("正在加载任务与模型数据...")
                QTimer.singleShot(40, self._ensure_splitter_sizes)

        # 异步读取数据：避免在主线程阻塞 SQLite 与文件系统 I/O
        from TrainingPlatform.core.env_manager import EnvironmentManager
        python_exe = EnvironmentManager.get_instance().get_selected_python()

        def fetch_project():
            p_info = self.db.get_project_by_id(project_id)
            if not p_info:
                return None
            summary = self.db.get_project_summary(project_id)
            tasks = self.db.get_tasks(project_id=project_id)
            models = self.db.get_models(project_id=project_id)
            task_ids = {t["id"] for t in tasks}
            records = tasks + [model_experiment(m) for m in models if m.get("task_id") not in task_ids]
            # 后台线程预先读取文件大小，彻底消除主线程磁盘 I/O 停顿
            for r in records:
                path = checkpoint_path(r)
                if path:
                    try:
                        r["_file_size"] = os.path.getsize(path)
                    except OSError:
                        r["_file_size"] = None
                else:
                    r["_file_size"] = None

            # 后台线程解析实验历史与指标数据
            data = {t["id"]: experiment_data(t, models, python_exe) for t in records}
            for model in models:
                record_id = model.get("task_id") if model.get("task_id") in task_ids else -model["id"]
                parsed = data.get(record_id, {})
                if parsed.get("signature"):
                    self.db.update_model_metadata(model["id"], parsed)

            return {
                "project_info": p_info,
                "summary": summary,
                "records": records,
                "models": models,
                "task_ids": task_ids,
                "data": data,
            }

        def on_project_loaded(result):
            if generation != self._load_generation or project_id != self.project_id:
                return
            if not result:
                return

            self.project_info = result["project_info"]
            self.lbl_title.setText(self.project_info["name"])
            summary = result["summary"]
            desc = self.project_info.get("description") or "无描述"
            self.lbl_desc.setText(f"{desc} · 包含 {summary['task_count']} 个训练任务 · {summary['model_count']} 个产出模型")

            records = result["records"]
            data = result["data"]
            self._records = {t["id"]: t for t in records}
            self._record_data = data
            for record in records:
                record["_experiment_data"] = data.get(record["id"], {})

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
                if hasattr(self, "chart_dashboard"):
                    self.chart_dashboard.set_loading(False)
                errors = [d["error"] for d in data.values() if d.get("error")]
                if errors:
                    MkMessage.warning(self, errors[0])

        if hasattr(self, "chart_dashboard"):
            self.chart_dashboard.set_loading(True)

        self.run_async(fetch_project, on_success=on_project_loaded, auto_loading=False)

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
        # 0 毫秒立即切换页面，用户屏幕即刻响应切入
        self.content_stack.setCurrentWidget(exp_view)
        exp_view.scroll_to_top()
        if task_id < 0:
            exp_view.load_model(-task_id)
        else:
            exp_view.load_task(task_id)

    def show_project_dashboard(self):
        """Navigate back from Experiment Detail View to Project Dashboard instantly without lag."""
        self.header_card.setVisible(True)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.content_stack.setCurrentWidget(self.dashboard_widget)
        QTimer.singleShot(40, self._ensure_splitter_sizes)
        # 静默在后台 150ms 检查是否有正在运行的训练状态更新，完全不阻塞当前 UI 渲染和返回动作
        if self.project_id:
            QTimer.singleShot(150, self._sync_dashboard_state)

    def _sync_dashboard_state(self):
        """Quietly check and sync task states without resetting items or freezing UI."""
        if not self.project_id or not self.isVisible():
            return
        if self.content_stack.currentWidget() != self.dashboard_widget:
            return
        try:
            tasks = self.db.get_tasks(project_id=self.project_id)
            models = self.db.get_models(project_id=self.project_id)
            task_ids = {t["id"] for t in tasks}
            records = tasks + [model_experiment(m) for m in models if m.get("task_id") not in task_ids]
            changed = False
            if len(records) != len(self._records):
                changed = True
            else:
                for r in records:
                    old = self._records.get(r["id"])
                    if not old or old.get("status") != r.get("status") or old.get("progress") != r.get("progress"):
                        changed = True
                        break
            if changed:
                self._records = {t["id"]: t for t in records}
                self._render_task_list(records, preserve_console=True, defer_heavy=True)
        except Exception:
            pass

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
                size = data.get("file_size", record.get("_file_size"))
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
        # 释放当前 Qt 鼠标点击事件循环，保证复选框打勾和高亮动画立即呈现（<1ms），
        # 并通过轻量级 25ms 定时器防抖合并高频点击，彻底消除主线程阻塞。
        self._schedule_charts_update(delay_ms=25)

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

    def _schedule_charts_update(self, delay_ms: int = 15):
        """Debounce comparison charts update with microsecond-level native QPainter rendering."""
        if not hasattr(self, "_chart_update_timer") or self._chart_update_timer is None:
            self._chart_update_timer = QTimer(self)
            self._chart_update_timer.setSingleShot(True)
            self._chart_update_timer.timeout.connect(self._update_comparison_charts)
        self._chart_update_timer.start(delay_ms)

    def _update_comparison_charts(self):
        # If dashboard is hidden or currently navigated to task details, skip expensive Matplotlib draws
        if not self.isVisible() or (hasattr(self, "content_stack") and self.content_stack.currentWidget() != getattr(self, "dashboard_widget", None)):
            return

        all_ids = sorted(self._records)
        checked_ids = sorted(self.checked_task_ids or ())

        # Collect history and color for each checked task
        records_history = []
        for tid in checked_ids:
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
                        mtime = os.path.getmtime(csv_path)
                        cached_mtime, cached_hist = getattr(self, "_csv_cache", {}).get(csv_path, (0, None))
                        if cached_hist and cached_mtime == mtime:
                            history = cached_hist
                        else:
                            with open(csv_path, encoding="utf-8-sig", newline="") as source:
                                rows = list(csv.DictReader(source))
                            if rows:
                                history = normalize_history({key: [r[key] for r in rows] for key in rows[0]})
                                if not hasattr(self, "_csv_cache"):
                                    self._csv_cache = {}
                                self._csv_cache[csv_path] = (mtime, history)
                                data["history"] = history
                    except (OSError, ValueError):
                        pass
            if not history and tid > 0:
                rows = self.db.get_task_metrics(tid)
                if rows:
                    history = {
                        "epoch": [r["epoch"] for r in rows],
                        "metrics/precision(B)": [r.get("precision", 0) for r in rows],
                        "metrics/recall(B)": [r.get("recall", 0) for r in rows],
                        "metrics/mAP50-95(B)": [r.get("map50_95", 0) for r in rows],
                        "metrics/mAP50(B)": [r.get("map50", 0) for r in rows],
                    }
                    data["history"] = history
            elif history and "history" not in data:
                data["history"] = history
            color = ACADEMIC_COLORS[all_ids.index(tid) % len(ACADEMIC_COLORS)]
            name = record["task_name"]
            records_history.append((tid, name, color, history))

        # ── 1. Group: Metrics ──
        if getattr(self.group_metrics, "is_expanded", True):
            metric_key_map = {
                "precision(B)": ["metrics/precision(B)", "precision(B)", "metrics/precision", "precision"],
                "recall(B)": ["metrics/recall(B)", "recall(B)", "metrics/recall", "recall"],
                "mAP50-95(B)": ["metrics/mAP50-95(B)", "mAP50-95(B)", "metrics/mAP50-95", "map50_95"],
                "precision(M)": ["metrics/precision(M)", "precision(M)"],
                "recall(M)": ["metrics/recall(M)", "recall(M)"],
                "mAP50-95(M)": ["metrics/mAP50-95(M)", "mAP50-95(M)"],
                "mAP50(B)": ["metrics/mAP50(B)", "mAP50(B)", "metrics/mAP50", "map50"],
                "mAP50(M)": ["metrics/mAP50(M)", "mAP50(M)"],
            }
            for cid, card in self.group_metrics.chart_cards.items():
                if not self.group_metrics.visible_charts.get(cid, True):
                    continue
                series = {}
                candidate_keys = metric_key_map.get(cid, [cid])
                for tid, name, color, hist in records_history:
                    val = []
                    for k in candidate_keys:
                        if k in hist and len(hist[k]) > 0:
                            val = hist[k]
                            break
                    epochs = hist.get("epoch", list(range(1, len(val) + 1)))
                    if val and len(epochs) == len(val):
                        display_name = name if name not in series else f"{name} ({tid})"
                        series[display_name] = {
                            "epochs": epochs,
                            "values": val,
                            "color": color,
                            "linestyle": "-"
                        }
                card.update_series(series)

        # ── 2. Group: Loss (Image 5: Train solid, Val dashed) ──
        if getattr(self.group_loss, "is_expanded", True):
            loss_key_map = {
                "box_loss": (["train/box_loss", "box_loss"], ["val/box_loss"]),
                "seg_loss": (["train/seg_loss", "seg_loss"], ["val/seg_loss"]),
                "cls_loss": (["train/cls_loss", "cls_loss"], ["val/cls_loss"]),
                "dfl_loss": (["train/dfl_loss", "dfl_loss", "train/l1_loss", "l1_loss"], ["val/dfl_loss", "val/l1_loss"]),
                "sem_loss": (["train/sem_loss", "sem_loss"], ["val/sem_loss"]),
            }
            show_train = self.group_loss.show_train_series
            show_val = self.group_loss.show_val_series
            for cid, card in self.group_loss.chart_cards.items():
                if not self.group_loss.visible_charts.get(cid, True):
                    continue
                series = {}
                train_keys, val_keys = loss_key_map.get(cid, ([f"train/{cid}", cid], [f"val/{cid}"]))
                for tid, name, color, hist in records_history:
                    # Train line (solid)
                    if show_train:
                        t_val = []
                        for k in train_keys:
                            if k in hist and len(hist[k]) > 0:
                                t_val = hist[k]
                                break
                        epochs = hist.get("epoch", list(range(1, len(t_val) + 1)))
                        if t_val and len(epochs) == len(t_val):
                            t_name = name if not show_val else f"{name} (train)"
                            if len(records_history) == 1 and show_val:
                                t_name = "Train"
                            series[t_name] = {
                                "epochs": epochs,
                                "values": t_val,
                                "color": color,
                                "linestyle": "-"
                            }
                    # Val line (dashed)
                    if show_val:
                        v_val = []
                        for k in val_keys:
                            if k in hist and len(hist[k]) > 0:
                                v_val = hist[k]
                                break
                        epochs = hist.get("epoch", list(range(1, len(v_val) + 1)))
                        if v_val and len(epochs) == len(v_val):
                            v_name = f"{name} (val)" if show_train else name
                            if len(records_history) == 1 and show_train:
                                v_name = "Val"
                            series[v_name] = {
                                "epochs": epochs,
                                "values": v_val,
                                "color": color,
                                "linestyle": "--"
                            }
                card.update_series(series)

        # ── 3. Group: Learning Rate (Image 3, 4) ──
        if getattr(self.group_lr, "is_expanded", True):
            lr_key_map = {
                "pg0": ["lr/pg0", "pg0"],
                "pg1": ["lr/pg1", "pg1"],
                "pg2": ["lr/pg2", "pg2"],
                "pg3": ["lr/pg3", "pg3"],
            }
            for cid, card in self.group_lr.chart_cards.items():
                if not self.group_lr.visible_charts.get(cid, True):
                    continue
                series = {}
                candidate_keys = lr_key_map.get(cid, [cid])
                for tid, name, color, hist in records_history:
                    val = []
                    for k in candidate_keys:
                        if k in hist and len(hist[k]) > 0:
                            val = hist[k]
                            break
                    epochs = hist.get("epoch", list(range(1, len(val) + 1)))
                    if val and len(epochs) == len(val):
                        display_name = name if name not in series else f"{name} ({tid})"
                        series[display_name] = {
                            "epochs": epochs,
                            "values": val,
                            "color": color,
                            "linestyle": "-"
                        }
                card.update_series(series)

    def select_task(self, task_id, clear_console=True, defer_logs=False):
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
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #dbeafe; color: #1d4ed8;")
            self.btn_stop_task.setVisible(True)
            self.btn_resume_task.setVisible(False)
        elif status in ("SUCCESS", "COMPLETED"):
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #dcfce7; color: #15803d;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)
        elif status == "IMPORTED":
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #e2e8f0; color: #475569;")
        else:
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #fee2e2; color: #b91c1c;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)

        if task.get("is_imported"):
            self.btn_resume_task.hide()
            self.btn_stop_task.hide()
        prog = task.get("progress", 0.0)
        self.progress_bar.setVisible(not task.get("is_imported"))
        self.progress_bar.percentage = int(prog)

        # Highlight corresponding curves across all charts (nearestLine)
        task_name = task.get("task_name")
        for grp in (getattr(self, "group_metrics", None), getattr(self, "group_loss", None), getattr(self, "group_lr", None)):
            if grp:
                for chart in grp.charts().values():
                    if hasattr(chart, "set_highlight_series"):
                        chart.set_highlight_series(task_name)

    def open_train_dialog(self):
        if not self.project_info or getattr(self, "_is_opening_train_dialog", False):
            return
        self._is_opening_train_dialog = True
        self.btn_new_train.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dlg = TrainCreateDialog(self.project_info, self.db, self.task_mgr, self)
            dlg.task_started.connect(self.on_new_task_started)
            main_win = self.window()
            QApplication.restoreOverrideCursor()
            if hasattr(main_win, "show_dialog_with_blur"):
                main_win.show_dialog_with_blur(dlg)
            else:
                dlg.exec()
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_new_train.setEnabled(True)
            self._is_opening_train_dialog = False

    def on_new_task_started(self, task_id):
        MkMessage.success(self, "新训练任务已成功启动！")
        self.current_task_id = task_id
        self.load_project(self.project_id)
        self.project_updated.emit(self.project_id)

    def handle_stop_task(self):
        if not self.current_task_id:
            return
        success, msg = self.task_mgr.stop_task(self.current_task_id)
        if success:
            MkMessage.info(self, "已向训练任务发送终止信号...")
            self.load_project(self.project_id)
        else:
            MkMessage.error(self, f"操作失败: {msg}")

    def handle_resume_task(self):
        if not self.current_task_id:
            return
        success, msg = self.task_mgr.start_task(self.current_task_id)
        if success:
            MkMessage.success(self, "训练任务已恢复运行！")
            self.load_project(self.project_id)
        else:
            MkMessage.error(self, f"启动失败: {msg}")

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
                out_path = m.export(**args)
                MkMessage.success(self, f"模型已成功导出为 {fmt} 格式！")
            except Exception as e:
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
            _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #dbeafe; color: #1d4ed8;")
            self.btn_stop_task.setVisible(True)
            self.btn_resume_task.setVisible(False)

    @Slot(int, str)
    def on_task_log(self, task_id, msg):
        pass  # Logs belong to task detail page (ExperimentDetailView)

    @Slot(int, dict)
    def on_task_progress(self, task_id, metrics):
        if self.current_task_id == task_id:
            prog = metrics.get("progress", 0.0)
            self.progress_bar.percentage = int(prog)
            if hasattr(self, "checked_task_ids") and task_id in self.checked_task_ids:
                self._schedule_charts_update(600)

    @Slot(int, bool, str)
    def on_task_finished(self, task_id, success, msg):
        if self.current_task_id == task_id:
            if success:
                self.lbl_cur_status.setText("SUCCESS")
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #dcfce7; color: #15803d;")
            else:
                self.lbl_cur_status.setText("STOPPED")
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_status, "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #fee2e2; color: #b91c1c;")
            self.btn_stop_task.setVisible(False)
            self.btn_resume_task.setVisible(True)
            self.load_project(self.project_id)

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            if hasattr(self, 'lbl_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 22px; font-weight: bold; color: #f4f4f5;")
            if hasattr(self, 'empty_title'):
                _MkThemeEngine.apply_style_sheet(self.empty_title, "font-size: 18px; font-weight: bold; color: #f4f4f5;")
            if hasattr(self, 'lbl_cur_task'):
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 13px; color: #f4f4f5;")
        else:
            if hasattr(self, 'lbl_title'):
                _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 22px; font-weight: bold; color: #1e293b;")
            if hasattr(self, 'empty_title'):
                _MkThemeEngine.apply_style_sheet(self.empty_title, "font-size: 18px; font-weight: bold; color: #1e293b;")
            if hasattr(self, 'lbl_cur_task'):
                _MkThemeEngine.apply_style_sheet(self.lbl_cur_task, "font-weight: bold; font-size: 13px; color: #1e293b;")

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

        if hasattr(self, 'group_metrics') and hasattr(self.group_metrics, "set_theme"):
            self.group_metrics.set_theme(is_dark)
        if hasattr(self, 'group_loss') and hasattr(self.group_loss, "set_theme"):
            self.group_loss.set_theme(is_dark)
        if hasattr(self, 'group_lr') and hasattr(self.group_lr, "set_theme"):
            self.group_lr.set_theme(is_dark)


