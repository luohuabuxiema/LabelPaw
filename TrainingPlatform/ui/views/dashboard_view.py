# -*- coding: utf-8 -*-
"""
Dashboard View for AI Vision Training Platform
Refactored to use MonkeyQt MkQWidget, MkCard, MkProTable.
Zero-bleed rounded table headers, native theme adaptation, millisecond theme toggles.
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt

from monkeyqt import MkCard, MkButton, MkAlert, ThemeEngine, MkQWidget, MkProTable
from TrainingPlatform.ui.components.gpu_card import GPUCard

# Safeguard MkPageSizeComboBox against RuntimeError on popup deletion during style changes
try:
    from monkeyqt.components.data.pro_table import MkPageSizeComboBox
    _orig_hide = MkPageSizeComboBox.hidePopup
    def _safe_hide(self):
        try:
            if hasattr(self, "_popup") and self._popup is not None:
                self._popup.hide()
        except (RuntimeError, AttributeError):
            pass
    MkPageSizeComboBox.hidePopup = _safe_hide
except Exception:
    pass

class DashboardView(MkQWidget):
    def __init__(self, db_manager, task_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager
        self.task_mgr = task_manager

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        self.title_lbl = QLabel("平台总览 Dashboard")
        main_layout.addWidget(self.title_lbl)

        # Top 3 Statistic Cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.card_projects = self._create_stat_card("训练项目数", "0", "#409eff")
        self.card_tasks = self._create_stat_card("训练任务数", "0", "#67c23a")
        self.card_models = self._create_stat_card("产出模型数", "0", "#e6a23c")

        stats_layout.addWidget(self.card_projects["card"])
        stats_layout.addWidget(self.card_tasks["card"])
        stats_layout.addWidget(self.card_models["card"])

        main_layout.addLayout(stats_layout)

        # GPU Card
        self.gpu_card = GPUCard()
        main_layout.addWidget(self.gpu_card)

        # Recent Tasks Section — using MonkeyQt MkProTable (fixes Image 1 corner clipping)
        cols = [
            {"key": "id", "label": "ID", "width": 60, "align": "center"},
            {"key": "task_name", "label": "任务名称", "align": "left"},
            {"key": "project_name", "label": "关联项目", "align": "left"},
            {"key": "model_name", "label": "模型架构", "align": "left"},
            {"key": "status", "label": "状态", "width": 110, "align": "center", "type": "status"},
            {"key": "progress_str", "label": "进度", "width": 110, "align": "center"},
            {"key": "map50_str", "label": "mAP50", "width": 110, "align": "center"},
        ]
        self.pro_table = MkProTable(
            columns=cols,
            data=[],
            title="最近训练任务",
            searchable=False,
            selectable=False,
            page_size=10,
            border_radius=12,
            parent=self
        )
        main_layout.addWidget(self.pro_table)
        main_layout.addStretch()

        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)

        self.refresh_data()

    def _create_stat_card(self, title, val, color):
        card = MkCard(title=title)
        l = card.content_layout
        v_lbl = QLabel(val)
        _MkThemeEngine.apply_style_sheet(v_lbl, f"font-size: 26px; font-weight: bold; color: {color};")
        l.addWidget(v_lbl)
        return {"card": card, "val_lbl": v_lbl}

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 20px; font-weight: bold; color: #f8fafc;")
        else:
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 20px; font-weight: bold; color: #0f172a;")

    def refresh_data(self, auto_loading: bool = True):
        def _fetch():
            stats = self.db.get_dashboard_stats()
            tasks = self.db.get_tasks()[:10]  # top 10
            table_data = []
            for t in tasks:
                map50_val = t.get('best_map50', 0.0) or 0.0
                if t.get('save_dir'):
                    try:
                        from TrainingPlatform.core.metrics_parser import parse_final_experiment_metrics
                        fm = parse_final_experiment_metrics(t['save_dir'])
                        if fm.get("map50", 0.0) > 0.0:
                            map50_val = fm["map50"]
                            if map50_val != t.get('best_map50', 0.0):
                                self.db.update_task_status(t['id'], t['status'], best_map50=map50_val, best_map50_95=fm["map50_95"])
                    except Exception:
                        pass

                table_data.append({
                    "id": t["id"],
                    "task_name": str(t.get("task_name", "-")),
                    "project_name": str(t.get("project_name", "-")),
                    "model_name": str(t.get("model_name", "-")),
                    "status": str(t.get("status", "-")),
                    "progress_str": f"{t.get('progress', 0.0):.1f}%",
                    "map50_str": f"{map50_val:.4f}"
                })
            return stats, table_data

        def _on_loaded(res):
            stats, table_data = res
            self.card_projects["val_lbl"].setText(str(stats["projects"]))
            self.card_tasks["val_lbl"].setText(str(stats["tasks"]))
            self.card_models["val_lbl"].setText(str(stats["models"]))
            self.pro_table.set_data(table_data)

        self.run_async(_fetch, on_success=_on_loaded, auto_loading=auto_loading)
