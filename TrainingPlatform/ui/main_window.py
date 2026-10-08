# -*- coding: utf-8 -*-
"""
Main Window Shell for AI Vision Platform
Refactored to a modern "Project-Centric" architecture (similar to Ultralytics Platform):
- Sidebar: Dashboard + Train (Submenu with "+ 新建项目" & dynamic project list) + Dataset Tools
- Workspaces:
  - DashboardView (Overview & GPU stats)
  - ProjectDetailView (Master-Visualizer: Tasks/Models, Empty State, Metric Charts, MkConsole)
  - DatasetToolView (Standalone utility toolbox)
- Modals:
  - ProjectCreateDialog (Lightweight project creator)
  - TrainCreateDialog (Local YOLO training launcher)
"""
import os
import sys
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                               QStackedWidget, QLabel, QPushButton,
                               QGraphicsBlurEffect, QDialog, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QObject, QEvent, Signal, QTimer, QRect
from PySide6.QtGui import QPainter, QColor, QCursor, QBrush, QGuiApplication

from monkeyqt import (MkWindow, MkMenu, MkButton, MkMessage, MkQWidget, use_theme, ThemeEngine)
from monkeyqt_icons import (PhSun, PhMoon, PhFolderPlus, PhPlus,
                            PhGauge, PhEye, PhDatabase, PhFolder, PhTrash)


class ModalBackdropOverlay(MkQWidget):
    """
    Semi-transparent backdrop overlay displayed over the blurred main content area.
    Provides modern frosted glass modal tinting in both light and dark modes.
    """
    clicked = Signal()

    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent=parent, role="transparent")
        self.is_dark = is_dark
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.is_dark:
            # Dark mode: deeper dark backdrop tint
            painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        else:
            # Light mode: clean, elegant slate tint for high contrast with white dialog
            painter.fillRect(self.rect(), QColor(15, 23, 42, 45))
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


class SidebarAddButton(QPushButton):
    """
    Transparent icon button for the sidebar submenu header.
    Uses QPainter to paint the '+' icon with precise optical centering,
    ensuring it is perfectly horizontally level with the adjacent Chinese menu text.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarBtnAddProject")
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("新建项目")
        self.setStyleSheet("""
            QPushButton#SidebarBtnAddProject,
            QPushButton#SidebarBtnAddProject:hover,
            QPushButton#SidebarBtnAddProject:pressed,
            QPushButton#SidebarBtnAddProject:focus {
                border: none;
                outline: none;
                background: transparent;
                background-color: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        self._is_hovered = False

    def enterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        active_fg = ThemeEngine.get("--sidebar-active-fg", "#FFFFFF" if ThemeEngine.is_dark() else "#0F172A")
        muted = ThemeEngine.get("--sidebar-text-muted", "#A1A1AA" if ThemeEngine.is_dark() else "#64748B")
        color = active_fg if self._is_hovered else muted
        dpr = self.devicePixelRatio() if hasattr(self, "devicePixelRatio") else 2.0
        pix = PhPlus.pixmap(size=14, color=color, dpr=dpr)
        if pix and not pix.isNull():
            pw = pix.width() / dpr
            ph = pix.height() / dpr
            x = (self.width() - pw) / 2.0
            y = (self.height() - ph) / 2.0
            painter.drawPixmap(int(round(x)), int(round(y)), pix)
        painter.end()


class SidebarProjectDeleteButton(QPushButton):
    """
    Delete button displayed on the right edge of a sidebar project submenu item.
    - Hidden by default.
    - Shown when parent sub_item is hovered (Fig 1).
    - Hover state: vibrant danger-red icon (#ef4444) with light-red rounded background (#fee2e2),
      matching Fig 2 benchmark.
    """
    def __init__(self, project_id: int, project_name: str, parent=None):
        super().__init__(parent)
        self.project_id = project_id
        self.project_name = project_name
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("删除项目")
        self._is_hovered = False
        self._is_pressed = False
        self.hide()
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)

    def enterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        is_dark = ThemeEngine.is_dark()

        # 1. Background on hover
        if self._is_hovered:
            if is_dark:
                bg_color = QColor(239, 68, 68, 65 if self._is_pressed else 45)
            else:
                bg_color = QColor("#fecaca" if self._is_pressed else "#fee2e2")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(self.rect(), 5, 5)

        # 2. Trash icon (red on hover, neutral gray when item hovered)
        color = "#ef4444" if self._is_hovered else ("#a1a1aa" if is_dark else "#94a3b8")
        dpr = self.devicePixelRatio() if hasattr(self, "devicePixelRatio") else 2.0
        pix = PhTrash.pixmap(size=14, color=color, dpr=dpr)
        if pix and not pix.isNull():
            pw = pix.width() / dpr
            ph = pix.height() / dpr
            x = (self.width() - pw) / 2.0
            y = (self.height() - ph) / 2.0
            painter.drawPixmap(int(round(x)), int(round(y)), pix)
        painter.end()


class SubItemHoverFilter(QObject):
    """
    Event filter to show/hide delete button on hover.
    """
    def __init__(self, item: QWidget, btn: QWidget):
        super().__init__(item)
        self.item = item
        self.btn = btn

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.Enter:
            self.btn.show()
        elif t == QEvent.Type.Leave:
            QTimer.singleShot(15, self._check_leave)
        return False

    def _check_leave(self):
        if not self.item.isVisible():
            self.btn.hide()
            return
        local_pos = self.item.mapFromGlobal(QCursor.pos())
        if not self.item.rect().contains(local_pos):
            self.btn.hide()

from TrainingPlatform.db.database import DatabaseManager
from TrainingPlatform.core.task_manager import TaskManager
from TrainingPlatform.ui.views import (DashboardView, DatasetToolView, 
                                        ProjectDetailView, ProjectCreateDialog)

class MainWindow(MkWindow):
    def __init__(self, parent=None):
        super().__init__(
            use_custom_title_bar=True,
            sidebar_full_height=True,
            parent=parent
        )
        # Center the window on screen for initial restore state
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1350, int(avail.width() * 0.9))
            h = min(880, int(avail.height() * 0.88))
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
            self._normal_geometry = QRect(x, y, w, h)
        else:
            self.resize(1350, 880)

        # Initialize Database & Task Manager
        self.db = DatabaseManager()
        self.task_mgr = TaskManager(self.db)
        self.is_dark_mode = False
        self.current_project_id = None
        self._is_navigating = False
        self.current_nav_id = "dashboard"

        # Ensure global theme stylesheet is active
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and not app.styleSheet():
            use_theme("默认")

        self.init_ui()

    def init_ui(self):
        # 1. Central Container Widget & Layout
        self.central_container = MkQWidget(self, role="transparent")
        self.central_container.setObjectName("CentralContainer")
        self.central_layout = QHBoxLayout(self.central_container)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)

        # 2. Left Sidebar Navigation Menu
        self.sidebar = MkMenu(title="智能训练与标注系统", collapse_mode="header", parent=self)
        self.sidebar.set_border_right("none")
        self.sidebar.setFixedWidth(225)
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        # ── 平台总览 (Top Level) ──
        self.item_dashboard = self.sidebar.add_item("dashboard", "平台总览", icon=PhGauge)

        # ── 视觉模型训练 (Project-Centric Dynamic Tree) ──
        self.sub_train = self.sidebar.add_submenu("视觉模型训练", icon=PhEye)
        self.sub_train.title_btn.clicked.connect(self.on_sub_train_clicked)

        # 头部右侧快捷「+」新建项目按钮 (水平精准对齐，纯图标无背景)
        self.btn_add_project = SidebarAddButton(self.sub_train.title_btn)
        self.btn_add_project.clicked.connect(self.on_action_new_project)
        self.sub_train.title_layout.addWidget(self.btn_add_project)

        # ── 数据集处理工具 (Standalone Utilities) ──
        self.item_dataset_tools = self.sidebar.add_item("dataset_tools", "数据集处理工具", icon=PhDatabase)

        # 侧边栏折叠联动控制：收起时不显示「+」号
        orig_toggle_collapse = self.sidebar.toggle_collapse
        def custom_toggle_collapse():
            orig_toggle_collapse()
            is_collapsed = self.sidebar._is_collapsed
            self.btn_add_project.setVisible(not is_collapsed)
            self.sidebar.setFixedWidth(64 if is_collapsed else 225)
        self.sidebar.toggle_collapse = custom_toggle_collapse

        self.central_layout.addWidget(self.sidebar)

        # 3. Stacked Area for Views
        self.stacked_widget = QStackedWidget(self)

        # Page 1: Dashboard View
        self.view_dashboard = DashboardView(self.db, self.task_mgr)
        self.view_dashboard.setObjectName("dashboard")

        # Page 2: Unified Project Detail View (Workspace for any active project)
        self.view_project_detail = ProjectDetailView(self.db, self.task_mgr)
        self.view_project_detail.setObjectName("project_detail")
        self.view_project_detail.project_deleted.connect(self.on_project_deleted)
        self.view_project_detail.project_updated.connect(self.on_project_updated)

        # Page 3: Dataset Tools View
        self.view_dataset_tool = DatasetToolView()
        self.view_dataset_tool.setObjectName("dataset_tools")

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.view_dashboard)
        self.stacked_widget.addWidget(self.view_project_detail)
        self.stacked_widget.addWidget(self.view_dataset_tool)

        self.central_layout.addWidget(self.stacked_widget, stretch=1)
        self.setCentralWidget(self.central_container)

        # 4. Clear decorative titlebar labels & remove bottom border
        self.titlebar._border_bottom = "none"
        self.titlebar.apply_theme_colors()
        while self.titlebar.center_layout.count():
            item = self.titlebar.center_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 5. Enable MonkeyQt Built-in Navigation Extras (< > history navigation with silky 200ms slide animation)
        self.enable_titlebar_extras(
            avatar=False,
            history_nav=True,
            animation_duration=200,
            stack=self.stacked_widget
        )
        self.stacked_widget = self._extras_stack

        # Add Theme Toggle Switcher on the right side of titlebar
        self.btn_theme_toggle = QPushButton()
        self.btn_theme_toggle.setObjectName("TitlebarThemeToggle")
        self.btn_theme_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_theme_toggle.setToolTip("切换默认/暗黑主题")
        self.btn_theme_toggle.setFixedSize(28, 28)
        self.btn_theme_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme_toggle.clicked.connect(self.toggle_light_dark_theme)
        self.titlebar.center_layout.addWidget(self.btn_theme_toggle)
        self._update_theme_toggle_style()

        # 6. Connect navigation signals
        self.sidebar.itemClicked.connect(self.on_sidebar_item_clicked)
        if hasattr(self, "titlebar_history_nav") and self.titlebar_history_nav:
            self.titlebar_history_nav.pageChanged.connect(self.on_page_changed)

        # 7. Initial Load: Populate dynamic projects & set active page
        self.sub_train.toggle() # Auto-expand sub_train
        self.refresh_sidebar_projects()
        self._sync_sidebar_selection("dashboard")

    def refresh_sidebar_projects(self, select_project_id=None):
        """Dynamic project list loader for the sidebar submenu."""
        # 1. Clear previous project items
        to_remove = []
        for item in list(self.sub_train._items):
            if item.item_id.startswith("project_"):
                to_remove.append(item)

        for item in to_remove:
            self.sub_train._items.remove(item)
            if item in self.sidebar._all_items:
                self.sidebar._all_items.remove(item)
            self.sub_train.content_layout.removeWidget(item)
            item.deleteLater()

        # 2. Re-query projects from DB
        projects = self.db.get_projects()
        for p in projects:
            sub_item = self.sidebar.add_submenu_item(
                self.sub_train,
                f"project_{p['id']}",
                p['name'],
                icon=PhFolder
            )

            # Delete button (hidden by default, shown on hover)
            btn_del = SidebarProjectDeleteButton(p['id'], p['name'], parent=sub_item)
            btn_del.clicked.connect(lambda checked=False, pid=p['id'], pname=p['name']: self.confirm_delete_project(pid, pname))
            sub_item.layout().addWidget(btn_del)

            hover_filter = SubItemHoverFilter(sub_item, btn_del)
            sub_item.installEventFilter(hover_filter)
            btn_del.installEventFilter(hover_filter)
            sub_item._hover_filter = hover_filter
            sub_item._delete_btn = btn_del

        # 3. Select project if requested
        if select_project_id:
            self.open_project(select_project_id)
        elif self.current_project_id and any(p['id'] == self.current_project_id for p in projects):
            self.view_project_detail.load_project(self.current_project_id)
            self._sync_sidebar_selection(f"project_{self.current_project_id}")

    def show_dialog_with_blur(self, dialog: QDialog, target_widget: QWidget = None, backdrop_dismiss: bool = True):
        """
        Show a modal dialog with smooth Gaussian blur and a modern frosted glass backdrop overlay 
        over the right-hand main content area (stacked_widget).
        """
        if target_widget is None:
            target_widget = self.stacked_widget

        # 1. Apply Gaussian blur to the right-hand main content area
        blur_effect = QGraphicsBlurEffect(target_widget)
        blur_effect.setBlurRadius(20)
        target_widget.setGraphicsEffect(blur_effect)

        # 2. Add semi-transparent backdrop overlay over target_widget geometry
        overlay = ModalBackdropOverlay(self.central_container, is_dark=self.is_dark_mode)
        overlay.setGeometry(target_widget.geometry())
        overlay.show()
        overlay.raise_()

        # Keep overlay in sync with target_widget during resizing
        class _ResizeFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
                    overlay.setGeometry(target_widget.geometry())
                return False

        resize_filter = _ResizeFilter()
        target_widget.installEventFilter(resize_filter)

        # 3. Center the dialog over the right-hand main content area
        target_center = target_widget.mapToGlobal(target_widget.rect().center())
        dialog.move(target_center.x() - dialog.width() // 2,
                    target_center.y() - dialog.height() // 2)

        # 4. Backdrop click handling: dismiss gracefully if user hasn't started entering data
        if backdrop_dismiss:
            def on_backdrop_clicked():
                if hasattr(dialog, "input_name") and dialog.input_name.text().strip():
                    return
                dialog.reject()
            overlay.clicked.connect(on_backdrop_clicked)

        try:
            return dialog.exec()
        finally:
            target_widget.removeEventFilter(resize_filter)
            overlay.deleteLater()
            target_widget.setGraphicsEffect(None)

    def on_action_new_project(self):
        """Triggered from the '+' button on the right of the Train submenu header."""
        dlg = ProjectCreateDialog(self.db, self)
        dlg.project_created.connect(lambda new_id: self.refresh_sidebar_projects(select_project_id=new_id))
        self.show_dialog_with_blur(dlg)


    def on_sidebar_item_clicked(self, item_id):
        if self._is_navigating:
            return

        if item_id == "action_new_project":
            self.on_action_new_project()
            return

        if item_id.startswith("project_"):
            proj_id = int(item_id.replace("project_", ""))
            self.open_project(proj_id)
            return

        # Regular views (dashboard, dataset_tools)
        self._is_navigating = True
        try:
            if hasattr(self, "titlebar_history_nav") and self.titlebar_history_nav:
                self.titlebar_history_nav.navigate(item_id)
            else:
                widget = self.findChild(QWidget, item_id)
                if widget:
                    self.stacked_widget.setCurrentWidget(widget)
            if item_id == "dashboard":
                QTimer.singleShot(210, self.view_dashboard.refresh_data)
        finally:
            self._is_navigating = False
        self._sync_sidebar_selection(item_id)

    def on_sub_train_clicked(self):
        """Handle clicking the '视觉模型训练' submenu header.
        Only expand/collapse the sub-menu items without changing or popping up right-hand views.
        """
        if self.sidebar._is_collapsed:
            self.sidebar.toggle_collapse()
            if not self.sub_train._is_expanded:
                self.sub_train.toggle()

    def open_project(self, project_id):
        self.current_project_id = project_id
        self._sync_sidebar_selection(f"project_{project_id}")

        # 1. 立即准备项目视图结构（更新标题、类型标签、任务列表与展示栈，耗时 < 3ms），确保滑入视口的正是目标看板，绝不露出空项目状态
        self.view_project_detail.load_project(project_id, defer_heavy=True)

        # 2. 0 毫秒立即启动页面滑入切换动画，用户点击的一瞬间立马看到切换动画！
        if hasattr(self, "titlebar_history_nav") and self.titlebar_history_nav:
            self._is_navigating = True
            try:
                self.titlebar_history_nav.navigate("project_detail")
            finally:
                self._is_navigating = False
        else:
            self.stacked_widget.setCurrentWidget(self.view_project_detail)

    def on_page_changed(self, page_id):
        if self._is_navigating:
            return

        if page_id == "project_detail":
            if self.current_project_id:
                self._sync_sidebar_selection(f"project_{self.current_project_id}")
                self.view_project_detail.load_project(self.current_project_id, defer_heavy=True)
            else:
                self._sync_sidebar_selection("project_detail")
            return

        self._sync_sidebar_selection(page_id)
        if page_id == "dashboard":
            # 切入总览页面时执行静默无感更新，配合 Loading Debounce 彻底消除骨架屏闪烁
            QTimer.singleShot(50, lambda: self.view_dashboard.refresh_data(auto_loading=False))

    def confirm_delete_project(self, project_id: int, project_name: str):
        reply = QMessageBox.question(
            self,
            "确认删除项目",
            f"确定要删除项目 [{project_name}] 吗？\n该操作将同时清理该项目下的所有训练历史记录与任务。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_project(project_id)
            MkMessage.success(self, f"项目 [{project_name}] 已成功删除！")
            self.on_project_deleted(project_id)

    def on_project_updated(self, project_id):
        self.refresh_sidebar_projects()

    def on_project_deleted(self, project_id):
        if self.current_project_id == project_id:
            self.current_project_id = None
            if hasattr(self, "titlebar_history_nav") and self.titlebar_history_nav:
                self.titlebar_history_nav.navigate("dashboard")
            else:
                self.stacked_widget.setCurrentWidget(self.view_dashboard)
            self._sync_sidebar_selection("dashboard")
        self.refresh_sidebar_projects()

    def _sync_sidebar_selection(self, current_nav_id: str):
        self.current_nav_id = current_nav_id
        if current_nav_id in ("project_detail", "train", "sub_train"):
            target_id = f"project_{self.current_project_id}" if self.current_project_id else "dashboard"
        else:
            target_id = current_nav_id

        self.sidebar.blockSignals(True)
        try:
            found = False
            for item in self.sidebar._all_items:
                if item.item_id == target_id:
                    found = True
                    break
            if found:
                self.sidebar.set_active(target_id)
            else:
                for item in self.sidebar._all_items:
                    item.setChecked(False)
        finally:
            self.sidebar.blockSignals(False)

    def _update_theme_toggle_style(self):
        hover_bg = "rgba(255, 255, 255, 0.12)" if self.is_dark_mode else "rgba(0, 0, 0, 0.06)"
        press_bg = "rgba(255, 255, 255, 0.20)" if self.is_dark_mode else "rgba(0, 0, 0, 0.12)"
        self.btn_theme_toggle.setStyleSheet(f"""
            QPushButton#TitlebarThemeToggle {{
                border: none;
                outline: none;
                background-color: transparent;
                border-radius: 6px;
                padding: 0px;
                margin: 0px;
            }}
            QPushButton#TitlebarThemeToggle:hover {{
                border: none;
                outline: none;
                background-color: {hover_bg};
            }}
            QPushButton#TitlebarThemeToggle:pressed {{
                border: none;
                outline: none;
                background-color: {press_bg};
            }}
            QPushButton#TitlebarThemeToggle:focus {{
                border: none;
                outline: none;
            }}
        """)
        icon_color = "#cbd5e1" if self.is_dark_mode else "#606266"
        icon_class = PhMoon if self.is_dark_mode else PhSun
        self.btn_theme_toggle.setIcon(icon_class.regular(size=16, color=icon_color))

    def update_style(self, is_max: bool | None = None):
        if not getattr(self, "use_custom_title_bar", False):
            return
        if is_max is None:
            is_max = self.isMaximized()
        self._current_is_max_state = is_max

        theme_name = ThemeEngine.current_theme()
        last_theme = getattr(self, "_last_styled_theme", None)
        if last_theme != theme_name or not getattr(self, "_window_container_styled", False):
            self._last_styled_theme = theme_name
            self._window_container_styled = True

            tokens = ThemeEngine.current_tokens()
            is_dark = ThemeEngine.is_dark()
            window_bg = tokens.get("--bg", "#ffffff" if not is_dark else "#18181b")
            border_color = tokens.get("--border", "#303030" if is_dark else "#e2e8f0")
            normal_radius = getattr(self, "_border_radius", 10)

            if self.container_frame:
                self.container_frame.setStyleSheet(f"""
                    QWidget#MkWindowContainer, QFrame#MkWindowContainer {{
                        background-color: {window_bg};
                        border: 1px solid {border_color};
                        border-radius: {normal_radius}px;
                    }}
                    QWidget#MkWindowContainer[mk_maximized="true"], QFrame#MkWindowContainer[mk_maximized="true"] {{
                        border: none;
                        border-radius: 0px;
                    }}
                """)

        # Fast dynamic property update on container_frame only — zero recursive restyling lag
        max_str = "true" if is_max else "false"
        if self.container_frame:
            self.container_frame.setProperty("mk_maximized", max_str)
            st = self.container_frame.style()
            st.unpolish(self.container_frame)
            st.polish(self.container_frame)
        if self.titlebar:
            self.titlebar.set_maximized_state(is_max)
        if hasattr(self, "_enable_native_corners"):
            self._enable_native_corners(is_max)

    def setStyleSheet(self, qss: str):
        # When app.setStyleSheet has already set the global application stylesheet,
        # re-applying the identical global QSS directly on MainWindow causes Qt
        # to redundantly traverse and restyle all child widgets in the window tree a second time.
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and app.styleSheet() and qss == app.styleSheet():
            return
        super().setStyleSheet(qss)

    def toggle_light_dark_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self._last_styled_theme = None
        use_theme("暗黑模式" if self.is_dark_mode else "默认")
        self.sidebar.set_border_right("none")
        self.titlebar._border_bottom = "none"
        self.titlebar.apply_theme_colors()
        self._update_theme_toggle_style()
        self.btn_add_project.update()
        self._sync_sidebar_selection(self.current_nav_id)
