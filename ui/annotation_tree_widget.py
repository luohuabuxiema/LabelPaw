# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QColorDialog, QLineEdit, QHBoxLayout, QMenu,
    QApplication, QStyle, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QPoint
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QFont, QIcon, QPixmap
import os
import json
import random


class ClassTreeItemWidget(QWidget):
    def __init__(self, item, class_name, color, count, parent_tree):
        super().__init__()
        self.item = item
        self.class_name = class_name
        self.color = color
        self.count = count
        self.parent_tree = parent_tree
        self.class_visible = True
        self.editor = None

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # 1. Caret expand button
        self.btnCaret = QPushButton()
        self.btnCaret.setFixedSize(16, 16)
        self.btnCaret.setFlat(True)
        self.btnCaret.setStyleSheet("border: none; background: transparent; padding: 0;")
        self.btnCaret.setCursor(Qt.PointingHandCursor)
        self.btnCaret.clicked.connect(self.toggle_expansion)

        # 2. Color dot button
        self.btnColorDot = QPushButton()
        self.btnColorDot.setFixedSize(12, 12)
        self.btnColorDot.setCursor(Qt.PointingHandCursor)
        self.update_color(self.color)
        self.btnColorDot.clicked.connect(self.pick_color)

        # 3. Class Name
        self.lblClassName = QLabel(self.class_name)
        self.lblClassName.setStyleSheet("font-weight: bold; background: transparent;")

        # 4. Count badge
        self.lblCount = QLabel(f"({self.count})" if self.count > 0 else "")
        self.lblCount.setStyleSheet("color: #64748B; font-size: 9pt; background: transparent;")

        # 5. Hide All button
        self.btnHideAll = QPushButton()
        self.btnHideAll.setFixedSize(20, 20)
        self.btnHideAll.setFlat(True)
        self.btnHideAll.setCursor(Qt.PointingHandCursor)
        self.btnHideAll.setStyleSheet("border: none; background: transparent; padding: 0;")
        self.update_eye_icon()
        self.btnHideAll.clicked.connect(self.toggle_visibility)
        self.btnHideAll.hide()  # Hover only

        layout.addWidget(self.btnCaret)
        layout.addWidget(self.btnColorDot)
        layout.addWidget(self.lblClassName)
        layout.addWidget(self.lblCount)
        layout.addStretch()
        layout.addWidget(self.btnHideAll)

        self.update_caret_icon()

    def update_color(self, color):
        self.color = color
        self.btnColorDot.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {color.darker(130).name()};
                border-radius: 6px;
                background-color: {color.name()};
            }}
        """)

    def pick_color(self):
        new_color = QColorDialog.getColor(self.color, self, f"选择 '{self.class_name}' 的颜色")
        if new_color.isValid():
            self.update_color(new_color)
            self.parent_tree.color_changed.emit(self.class_name, new_color)
            self.parent_tree._class_colors[self.class_name] = new_color
            self.parent_tree._save_colors()
            self.parent_tree.refresh_canvas_shape_colors(self.class_name, new_color)

    def toggle_expansion(self):
        is_expanded = self.item.isExpanded()
        self.item.setExpanded(not is_expanded)
        self.update_caret_icon()

    def update_caret_icon(self):
        is_expanded = self.item.isExpanded()
        caret_path = "ui/icon/caret-down.svg" if is_expanded else "ui/icon/caret-right.svg"
        icon_color = QColor("#CBD5E1") if self.parent_tree.is_dark else QColor("#475569")
        colored_icon = self.parent_tree.get_theme_icon(caret_path, icon_color)
        self.btnCaret.setIcon(colored_icon)
        self.btnCaret.setVisible(self.count > 0)

    def update_eye_icon(self):
        eye_path = "ui/icon/eye.svg" if self.class_visible else "ui/icon/eye-slash.svg"
        icon_color = QColor("#CBD5E1") if self.parent_tree.is_dark else QColor("#475569")
        colored_icon = self.parent_tree.get_theme_icon(eye_path, icon_color)
        self.btnHideAll.setIcon(colored_icon)

    def toggle_visibility(self):
        self.class_visible = not self.class_visible
        self.update_eye_icon()
        self.parent_tree.toggle_class_shapes_visibility(self.class_name, self.class_visible)

    def start_edit(self):
        self.lblClassName.hide()
        self.btnCaret.hide()
        self.btnColorDot.hide()
        self.lblCount.hide()

        self.editor = QLineEdit(self.class_name)
        bg = "#334155" if self.parent_tree.is_dark else "#F1F5F9"
        fg = "white" if self.parent_tree.is_dark else "black"
        self.editor.setStyleSheet(f"font-weight: bold; background: {bg}; color: {fg}; border: 1px solid #475569; padding: 2px;")
        self.editor.returnPressed.connect(self.finish_edit)
        self.editor.focusOutEvent = lambda event: self.finish_edit()

        self.layout().insertWidget(2, self.editor)
        self.editor.setFocus()
        self.editor.selectAll()

    def finish_edit(self):
        if not self.editor:
            return
        new_name = self.editor.text().strip()
        self.editor.deleteLater()
        self.editor = None

        self.lblClassName.show()
        self.btnCaret.show()
        self.btnColorDot.show()
        self.lblCount.show()

        if new_name and new_name != self.class_name:
            old_name = self.class_name
            self.class_name = new_name
            self.lblClassName.setText(new_name)
            self.parent_tree.rename_class_in_data(old_name, new_name)

    def set_active(self, active):
        text_color = "#E2E8F0" if self.parent_tree.is_dark else "#1E293B"
        if active:
            bg_selected = "#334155" if self.parent_tree.is_dark else "#CBD5E1"
            self.setStyleSheet(f"background-color: {bg_selected}; border-radius: 4px;")
            self.lblClassName.setStyleSheet(f"font-weight: bold; color: {text_color}; background: transparent;")
        else:
            self.setStyleSheet("background-color: transparent;")
            self.lblClassName.setStyleSheet(f"font-weight: bold; color: {text_color}; background: transparent;")

    def set_hovered(self, hovered):
        if hovered:
            bg_hover = "#1E293B" if self.parent_tree.is_dark else "#E2E8F0"
            self.setStyleSheet(f"background-color: {bg_hover}; border-radius: 4px;")
        else:
            selected_items = self.parent_tree.treeWidget.selectedItems()
            selected_item = selected_items[0] if selected_items else None
            is_active = False
            if selected_item:
                if selected_item.parent():
                    is_active = (selected_item.parent() == self.item)
                else:
                    is_active = (selected_item == self.item)
            self.set_active(is_active)

    def enterEvent(self, event):
        self.set_hovered(True)
        self.btnHideAll.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hovered(False)
        self.btnHideAll.hide()
        super().leaveEvent(event)


class AnnotationTreeItemWidget(QWidget):
    def __init__(self, item, shape, idx, parent_tree):
        super().__init__()
        self.item = item
        self.shape = shape
        self.idx = idx
        self.parent_tree = parent_tree
        self.is_visible = shape.isVisible()

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 2)  # No left margin, indicator bar handles it
        layout.setSpacing(6)

        # 0. Left Indicator Bar (keeps layout aligned and consistent)
        self.indicatorBar = QWidget()
        self.indicatorBar.setFixedWidth(3)
        self.indicatorBar.setFixedHeight(14)
        self.indicatorBar.setStyleSheet("background-color: transparent;")

        # 1. Text Label
        text = self.parent_tree.get_shape_info_text(self.idx, self.shape)
        self.lblText = QLabel(text)
        self.lblText.setStyleSheet("color: #94A3B8; background: transparent; padding-left: 4px;")

        # 2. Hide button
        self.btnHide = QPushButton()
        self.btnHide.setFixedSize(16, 16)
        self.btnHide.setFlat(True)
        self.btnHide.setCursor(Qt.PointingHandCursor)
        self.btnHide.setStyleSheet("border: none; background: transparent; padding: 0;")
        self.update_eye_icon()
        self.btnHide.clicked.connect(self.toggle_visibility)
        self.btnHide.hide()

        # 3. Delete button
        self.btnDelete = QPushButton()
        self.btnDelete.setFixedSize(16, 16)
        self.btnDelete.setFlat(True)
        self.btnDelete.setCursor(Qt.PointingHandCursor)
        self.btnDelete.setStyleSheet("border: none; background: transparent; padding: 0;")
        trash_color = QColor("#EF4444")
        self.btnDelete.setIcon(self.parent_tree.get_theme_icon("ui/icon/trash.svg", trash_color))
        self.btnDelete.clicked.connect(self.delete_shape)
        self.btnDelete.hide()

        layout.addWidget(self.indicatorBar)
        layout.addWidget(self.lblText)
        layout.addStretch()
        layout.addWidget(self.btnHide)
        layout.addWidget(self.btnDelete)

    def update_eye_icon(self):
        eye_path = "ui/icon/eye.svg" if self.is_visible else "ui/icon/eye-slash.svg"
        icon_color = QColor("#CBD5E1") if self.parent_tree.is_dark else QColor("#475569")
        colored_icon = self.parent_tree.get_theme_icon(eye_path, icon_color)
        self.btnHide.setIcon(colored_icon)

    def toggle_visibility(self):
        self.is_visible = not self.is_visible
        self.update_eye_icon()
        self.shape.setVisible(self.is_visible)
        if hasattr(self.shape, '_update_handle_visibility'):
            self.shape._update_handle_visibility()

    def delete_shape(self):
        self.parent_tree.delete_shape(self.shape)

    def set_selected(self, selected):
        if selected:
            border_color = self.parent_tree.get_class_color(self.shape.label).name()
            self.indicatorBar.setStyleSheet(f"background-color: {border_color}; border-radius: 1.5px;")
            bg_selected = "#334155" if self.parent_tree.is_dark else "#CBD5E1"
            self.setStyleSheet(f"background-color: {bg_selected}; border-radius: 4px;")
            
            text_color = "#E2E8F0" if self.parent_tree.is_dark else "#1E293B"
            self.lblText.setStyleSheet(f"color: {text_color}; background: transparent; font-weight: bold; padding-left: 4px;")
        else:
            self.indicatorBar.setStyleSheet("background-color: transparent;")
            self.setStyleSheet("background-color: transparent;")
            self.lblText.setStyleSheet("color: #94A3B8; background: transparent; font-weight: normal; padding-left: 4px;")

    def set_hovered(self, hovered):
        if hovered:
            bg_hover = "#1E293B" if self.parent_tree.is_dark else "#E2E8F0"
            self.setStyleSheet(f"background-color: {bg_hover}; border-radius: 4px;")
            self.btnHide.show()
            self.btnDelete.show()
        else:
            self.setStyleSheet("background-color: transparent;")
            is_selected = self.item.isSelected()
            self.set_selected(is_selected)
            if not is_selected:
                self.btnHide.hide()
                self.btnDelete.hide()

    def enterEvent(self, event):
        self.set_hovered(True)
        self.parent_tree.highlight_shape_on_canvas(self.shape, True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hovered(False)
        self.parent_tree.highlight_shape_on_canvas(self.shape, False)
        super().leaveEvent(event)


class AnnotationTreeWidget(QWidget):
    """带搜索、添加、分级管理的标注层级结构树组件"""

    class_added = Signal(str)
    class_renamed = Signal(str, str)
    class_removed = Signal(str)
    color_changed = Signal(str, object)
    item_changed = Signal(object)  # 兼容旧 item_changed 信号
    shape_class_reassigned = Signal(object, str)  # (shape, new_class_name) 当点击父类别修改选中图形的类别

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._class_list = []
        self._class_colors = {}
        self._color_index = 0
        self._current_dir = None
        self.is_dark = True
        self._hovered_tree_items = []
        self._class_items = {}  # {class_name: QTreeWidgetItem}
        self._updating = False

        # 兼容 main.py 的 listWidget 接口引用
        self.listWidget = self

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        self.setObjectName("annotationTreeCard")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Search bar
        self.searchInput = QLineEdit()
        self.searchInput.setObjectName("classSearchInput")
        self.searchInput.setPlaceholderText("搜索类别...")
        self.searchInput.setClearButtonEnabled(True)
        layout.addWidget(self.searchInput)

        # 2. Add class bar
        add_bar = QHBoxLayout()
        add_bar.setContentsMargins(0, 0, 0, 0)
        add_bar.setSpacing(4)

        self.btnAdd = QPushButton("+")
        self.btnAdd.setObjectName("classAddBtn")
        self.btnAdd.setFixedSize(30, 30)
        self.btnAdd.setCursor(Qt.PointingHandCursor)
        self.btnAdd.setToolTip("添加新类别")

        self.addInput = QLineEdit()
        self.addInput.setObjectName("classAddInput")
        self.addInput.setPlaceholderText("输入新类别名...")

        add_bar.addWidget(self.btnAdd)
        add_bar.addWidget(self.addInput, 1)
        layout.addLayout(add_bar)

        # 3. Tree Widget
        self.treeWidget = QTreeWidget()
        self.treeWidget.setObjectName("classListWidget")
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setRootIsDecorated(False)  # Hide default branch arrow
        self.treeWidget.setMouseTracking(True)
        self.treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.treeWidget.setExpandsOnDoubleClick(False)  # 双击不展开/收缩，展开收缩由前面的 caret 图标控制
        self.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        layout.addWidget(self.treeWidget, 1)

    def _connect_signals(self):
        self.searchInput.textChanged.connect(self._on_search_text_changed)
        self.addInput.returnPressed.connect(self._on_add_class)
        self.btnAdd.clicked.connect(self._on_add_class)
        self.treeWidget.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.treeWidget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.treeWidget.customContextMenuRequested.connect(self._show_context_menu)

    def set_theme(self, is_dark):
        self.is_dark = is_dark
        self.update_theme_style(is_dark)
        
        # 逐个更新已存在项的图标和样式，避免清空树导致已标框子项丢失
        for i in range(self.treeWidget.topLevelItemCount()):
            parent_item = self.treeWidget.topLevelItem(i)
            parent_widget = self.treeWidget.itemWidget(parent_item, 0)
            if isinstance(parent_widget, ClassTreeItemWidget):
                parent_widget.update_caret_icon()
                parent_widget.update_eye_icon()
                
            for j in range(parent_item.childCount()):
                child_item = parent_item.child(j)
                child_widget = self.treeWidget.itemWidget(child_item, 0)
                if isinstance(child_widget, AnnotationTreeItemWidget):
                    child_widget.update_eye_icon()
                    
        self.update_widgets_selection_state()


    def update_theme_style(self, is_dark):
        text_color = "#E2E8F0" if is_dark else "#1E293B"

        self.treeWidget.setStyleSheet(f"""
            QTreeWidget {{
                border: none;
                background-color: transparent;
                outline: none;
            }}
            QTreeWidget::item {{
                background: transparent;
                color: {text_color};
                border: none;
                outline: none;
            }}
            QTreeWidget::item:hover {{
                background-color: transparent;
            }}
            QTreeWidget::item:selected {{
                background-color: transparent;
            }}
        """)

    # ======================== 搜索过滤 ========================

    def _on_search_text_changed(self, text):
        keyword = text.strip().lower()
        for i in range(self.treeWidget.topLevelItemCount()):
            item = self.treeWidget.topLevelItem(i)
            cls_name = item.data(0, Qt.UserRole)
            item.setHidden(keyword != "" and keyword not in cls_name.lower())

    # ======================== 添加类别 ========================

    def _on_add_class(self):
        text = self.addInput.text().strip()
        if not text:
            return
        if text in self._class_list:
            if text in self._class_items:
                self.treeWidget.setCurrentItem(self._class_items[text])
            self.addInput.clear()
            return

        self.add_class(text)
        self.addInput.clear()
        self.class_added.emit(text)

    # ======================== 重命名 ========================

    def _on_item_double_clicked(self, item, column):
        # Double-click Class item to edit inline
        if not item.parent():
            widget = self.treeWidget.itemWidget(item, 0)
            if isinstance(widget, ClassTreeItemWidget):
                widget.start_edit()

    def _show_context_menu(self, position):
        item = self.treeWidget.itemAt(position)
        if item is None:
            return
        if item.parent() is not None:
            item = item.parent()

        class_name = item.data(0, Qt.UserRole)
        if not class_name:
            return

        menu = QMenu(self.treeWidget)
        delete_action = menu.addAction("删除类别")
        selected_action = menu.exec(self.treeWidget.viewport().mapToGlobal(position))
        if selected_action == delete_action:
            self.class_removed.emit(str(class_name))

    def rename_class_in_data(self, old_name, new_name):
        if old_name not in self._class_list:
            return
        if new_name in self._class_list:
            return

        idx = self._class_list.index(old_name)
        self._class_list[idx] = new_name
        if old_name in self._class_colors:
            self._class_colors[new_name] = self._class_colors.pop(old_name)
            self._save_colors()

        self.class_renamed.emit(old_name, new_name)

        if old_name in self._class_items:
            item = self._class_items.pop(old_name)
            self._class_items[new_name] = item
            item.setData(0, Qt.UserRole, new_name)

    # ======================== 双向选中机制 ========================

    def update_widgets_selection_state(self):
        selected_items = self.treeWidget.selectedItems()
        selected_item = selected_items[0] if selected_items else None
        
        active_parent_cls = None
        if selected_item:
            if selected_item.parent():
                active_parent_cls = selected_item.parent().data(0, Qt.UserRole)
            else:
                active_parent_cls = selected_item.data(0, Qt.UserRole)
                
        for i in range(self.treeWidget.topLevelItemCount()):
            parent = self.treeWidget.topLevelItem(i)
            cls_name = parent.data(0, Qt.UserRole)
            p_selected = parent.isSelected()
            is_parent_active = (p_selected or cls_name == active_parent_cls)
            
            p_widget = self.treeWidget.itemWidget(parent, 0)
            if isinstance(p_widget, ClassTreeItemWidget):
                p_widget.set_active(is_parent_active)
                
            for j in range(parent.childCount()):
                child = parent.child(j)
                c_selected = child.isSelected()
                c_widget = self.treeWidget.itemWidget(child, 0)
                if isinstance(c_widget, AnnotationTreeItemWidget):
                    c_widget.set_selected(c_selected)

    def _on_tree_selection_changed(self):
        if self._updating:
            return
        self._updating = True
        try:
            self.update_widgets_selection_state()
            
            selected = self.treeWidget.selectedItems()
            if not selected:
                return
            item = selected[0]
            shape = item.data(0, Qt.UserRole)
            # If a child shape item is selected in tree, select it on the canvas!
            if shape and hasattr(shape, 'scene') and shape.scene():
                scene = shape.scene()
                scene.blockSignals(True)
                for s_item in scene.selectedItems():
                    if s_item != shape:
                        s_item.setSelected(False)
                shape.setSelected(True)
                scene.blockSignals(False)
            else:
                # 点击的是父类别项：检查画布上是否有已选中的图形，如果有则将其类别修改为当前点击的类别
                cls_name = item.data(0, Qt.UserRole)
                from PySide6.QtWidgets import QApplication
                from labelpaw.graphics.shapes import BaseShape
                for widget in QApplication.topLevelWidgets():
                    if hasattr(widget, 'scene') and widget.scene:
                        scene = widget.scene
                        selected_shapes = [s for s in scene.selectedItems()
                                           if isinstance(s, BaseShape) and not getattr(s, 'is_temp', False)]
                        if selected_shapes and cls_name:
                            # 有选中的图形 → 修改其类别
                            for s in selected_shapes:
                                self.shape_class_reassigned.emit(s, cls_name)
                        else:
                            # 没有选中的图形 → 取消所有选中
                            scene.blockSignals(True)
                            for s_item in scene.selectedItems():
                                s_item.setSelected(False)
                            scene.blockSignals(False)
                        break
        finally:
            self._updating = False

    def select_item_by_shape(self, shape):
        if self._updating:
            return
        self._updating = True
        try:
            self.treeWidget.blockSignals(True)
            for i in range(self.treeWidget.topLevelItemCount()):
                parent = self.treeWidget.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.data(0, Qt.UserRole) == shape:
                        self.treeWidget.clearSelection()
                        child.setSelected(True)
                        self.treeWidget.setCurrentItem(child)
                        # Automatically expand the parent to show selection
                        parent.setExpanded(True)
                        # Sync caret icon status
                        p_widget = self.treeWidget.itemWidget(parent, 0)
                        if isinstance(p_widget, ClassTreeItemWidget):
                            p_widget.update_caret_icon()
                        
                        self.update_widgets_selection_state()
                        self.treeWidget.blockSignals(False)
                        return
            self.treeWidget.blockSignals(False)
        finally:
            self._updating = False

    def clear_tree_selection(self):
        if self._updating:
            return
        self._updating = True
        try:
            self.treeWidget.blockSignals(True)
            self.treeWidget.clearSelection()
            self.update_widgets_selection_state()
            self.treeWidget.blockSignals(False)
        finally:
            self._updating = False

    # ======================== 双向悬停高亮 ========================

    def highlight_item_by_shape(self, shape, is_hovered):
        self.clear_tree_hover_highlights()
        if not is_hovered:
            return

        for i in range(self.treeWidget.topLevelItemCount()):
            parent = self.treeWidget.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.data(0, Qt.UserRole) == shape:
                    if parent.isExpanded():
                        self.set_item_hover_highlight(child, True)
                    else:
                        self.set_item_hover_highlight(parent, True)
                    return

    def set_item_hover_highlight(self, item, highlight):
        widget = self.treeWidget.itemWidget(item, 0)
        if widget and hasattr(widget, 'set_hovered'):
            widget.set_hovered(highlight)
            if highlight:
                self._hovered_tree_items.append(item)
        else:
            bg_hover = QColor("#1E293B" if self.is_dark else "#E2E8F0")
            if highlight:
                item.setBackground(0, QBrush(bg_hover))
                self._hovered_tree_items.append(item)
            else:
                item.setBackground(0, QBrush(Qt.transparent))

    def clear_tree_hover_highlights(self):
        for item in self._hovered_tree_items:
            widget = self.treeWidget.itemWidget(item, 0)
            if widget and hasattr(widget, 'set_hovered'):
                widget.set_hovered(False)
            else:
                item.setBackground(0, QBrush(Qt.transparent))
        self._hovered_tree_items.clear()

    def highlight_shape_on_canvas(self, shape, is_hovered):
        if is_hovered:
            if hasattr(shape, 'apply_hover_enter'):
                shape.apply_hover_enter(shape)
            if hasattr(shape, '_update_handle_visibility'):
                shape._hovered = True
                shape._update_handle_visibility()
        else:
            if hasattr(shape, 'apply_hover_leave'):
                shape.apply_hover_leave(shape)
            if hasattr(shape, '_update_handle_visibility'):
                shape._hovered = False
                shape._update_handle_visibility()

    # ======================== 公共 API ========================

    def get_selected_class(self):
        selected = self.treeWidget.selectedItems()
        if not selected:
            return None
        item = selected[0]
        if item.parent():
            return item.parent().data(0, Qt.UserRole)
        return item.data(0, Qt.UserRole)

    def add_class(self, cls_name, color=None):
        if cls_name in self._class_list:
            return
        self._class_list.append(cls_name)

        # Allocate color
        from ui.class_list_widget import get_palette_color
        if color and isinstance(color, QColor):
            self._class_colors[cls_name] = color
        elif cls_name not in self._class_colors:
            self._class_colors[cls_name] = get_palette_color(self._color_index)
            
        self._color_index += 1  # 无论是否有预设颜色，都推进颜色索引，确保新加类别颜色不复用初始颜色

        self.refresh_tree_items()

    def remove_class(self, cls_name):
        if cls_name not in self._class_list:
            return
        self._class_list.remove(cls_name)
        self._class_colors.pop(cls_name, None)
        self.refresh_tree_items()

    def clear_classes(self):
        self._class_list.clear()
        self._class_colors.clear()
        self._color_index = 0
        self.treeWidget.clear()
        self._class_items.clear()

    def get_class_list(self):
        return list(self._class_list)

    def select_class_by_label(self, label):
        """按类别名称精确选中指定类别。"""
        item = self._class_items.get(str(label))
        if item is not None:
            self.treeWidget.clearSelection()
            item.setSelected(True)
            self.treeWidget.setCurrentItem(item)

    def get_class_color(self, cls_name):
        return self._class_colors.get(cls_name, QColor("#3B82F6"))

    def get_all_colors(self):
        return dict(self._class_colors)

    def set_class_color(self, cls_name, color):
        self._class_colors[cls_name] = color
        self.refresh_tree_items()

    # ======================== 标注层级更新逻辑 ========================

    def update_annotations(self, shapes):
        """主入口：当画布状态更新时，由 main.py 触发以重新装填子标注项"""
        # 1. 暂存所有大分类折叠展开状态，保证视觉连续性
        expanded_states = {}
        for cls_name, item in self._class_items.items():
            expanded_states[cls_name] = item.isExpanded()

        # 阻止树信号，防止 takeChildren/addChild 等操作触发 itemSelectionChanged 连锁反应
        self.treeWidget.blockSignals(True)

        # 2. 清空所有大类的子节点
        for item in self._class_items.values():
            item.takeChildren()

        # 3. 按类别将标注进行归类
        grouped_shapes = {cls: [] for cls in self._class_list}
        for s in shapes:
            label = getattr(s, 'label', 'Unknown')
            if label not in grouped_shapes:
                grouped_shapes[label] = []
            grouped_shapes[label].append(s)

        # 4. 重新挂载子节点并更新 parent 栏数据
        for cls_name in self._class_list:
            item = self._class_items.get(cls_name)
            if not item:
                continue

            shapes_in_class = grouped_shapes.get(cls_name, [])
            count = len(shapes_in_class)

            # 更新父大类的 itemWidget (优先在原地更新已有控件，防止旧控件未销毁而重叠，并保证 count 计数正确)
            existing_widget = self.treeWidget.itemWidget(item, 0)
            if isinstance(existing_widget, ClassTreeItemWidget):
                existing_widget.count = count
                existing_widget.lblCount.setText(f"({count})" if count > 0 else "")
                existing_widget.update_caret_icon()
                parent_widget = existing_widget
            else:
                parent_widget = ClassTreeItemWidget(
                    item=item,
                    class_name=cls_name,
                    color=self.get_class_color(cls_name),
                    count=count,
                    parent_tree=self
                )
                self.treeWidget.setItemWidget(item, 0, parent_widget)

            # 挂载子标注项
            for idx, s in enumerate(shapes_in_class, 1):
                child_item = QTreeWidgetItem(item)
                child_item.setData(0, Qt.UserRole, s)
                
                child_widget = AnnotationTreeItemWidget(
                    item=child_item,
                    shape=s,
                    idx=idx,
                    parent_tree=self
                )
                self.treeWidget.setItemWidget(child_item, 0, child_widget)

            # 恢复之前的展开折叠状态
            was_expanded = expanded_states.get(cls_name, False)
            item.setExpanded(was_expanded)
            # 同步更新 caret 图标
            parent_widget.update_caret_icon()

        # 恢复树信号
        self.treeWidget.blockSignals(False)
            
        # 恢复画布上当前选中图形在树中的选中状态
        active_shape = next((s for s in shapes if s.isSelected()), None)
        if active_shape:
            self.select_item_by_shape(active_shape)
        else:
            self.update_widgets_selection_state()

    def refresh_tree_items(self):
        """主要在类列表发生纯类目变动（加载、增删类目等）时重构 Top-level Class 类目项"""
        self.treeWidget.clear()
        self._class_items.clear()

        for cls_name in self._class_list:
            item = QTreeWidgetItem(self.treeWidget)
            item.setData(0, Qt.UserRole, cls_name)
            self._class_items[cls_name] = item

            parent_widget = ClassTreeItemWidget(
                item=item,
                class_name=cls_name,
                color=self.get_class_color(cls_name),
                count=0,
                parent_tree=self
            )
            self.treeWidget.setItemWidget(item, 0, parent_widget)

    # ======================== 子控件业务桥接 ========================

    def delete_shape(self, shape):
        """子删除图标触发"""
        scene = shape.scene()
        if scene:
            scene.removeItem(shape)
            scene.state_changed.emit()

    def toggle_class_shapes_visibility(self, class_name, visible):
        """大类一键隐藏/显示"""
        for i in range(self.treeWidget.topLevelItemCount()):
            parent = self.treeWidget.topLevelItem(i)
            if parent.data(0, Qt.UserRole) == class_name:
                # 级联更新所有子节点的眼睛显示状态
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    c_widget = self.treeWidget.itemWidget(child, 0)
                    if isinstance(c_widget, AnnotationTreeItemWidget):
                        c_widget.is_visible = visible
                        c_widget.update_eye_icon()
                        c_widget.shape.setVisible(visible)
                        if hasattr(c_widget.shape, '_update_handle_visibility'):
                            c_widget.shape._update_handle_visibility()

    def refresh_canvas_shape_colors(self, class_name, color):
        """颜色盘更改后实时重绘 canvas 上同一类的所有形状颜色"""
        # We can find the active main window or query scene shapes
        from PySide6.QtWidgets import QApplication
        for widget in QApplication.topLevelWidgets():
            if hasattr(widget, 'scene') and widget.scene:
                for item in widget.scene.items():
                    if hasattr(item, 'label') and item.label == class_name:
                        if hasattr(item, 'set_color'):
                            item.set_color(color)
                if hasattr(widget, 'auto_save_annotation'):
                    widget.auto_save_annotation()
                break

    # ======================== 基础辅助 ========================

    def get_theme_icon(self, icon_path, color):
        icon = QIcon(icon_path)
        pixmap = icon.pixmap(32, 32)
        if pixmap.isNull():
            return icon
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()

        new_icon = QIcon()
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        return new_icon

    def get_shape_info_text(self, idx, shape):
        from labelpaw.graphics.shapes import RectShape, PolyShape, RotatedRectShape, PoseShape, PointShape
        if isinstance(shape, PoseShape):
            return f"{idx} {len(shape.kps)} 个关键点"
        elif isinstance(shape, PolyShape):
            return f"{idx} {shape.polygon().count()} 个点"
        elif isinstance(shape, RotatedRectShape):
            w = int(shape.box_w)
            h = int(shape.box_h)
            angle = int(shape.rotation())
            return f"{idx} {w} x {h} @{angle}°"
        elif isinstance(shape, RectShape):
            r = shape.rect()
            w = int(r.width())
            h = int(r.height())
            return f"{idx} {w} x {h}"
        elif isinstance(shape, PointShape):
            return f"{idx} 关键点"
        return f"{idx} 标注框"

    # ======================== 持久化代理 ========================

    def set_working_dir(self, dir_path):
        self._current_dir = dir_path

    def load_classes(self, dir_path):
        self.clear_classes()
        self._current_dir = dir_path
        self._color_index = 0

        color_file = os.path.join(dir_path, "class_colors.json")
        saved_colors = {}
        if os.path.exists(color_file):
            try:
                with open(color_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    saved_colors = {k: QColor(v) for k, v in raw.items()}
            except Exception:
                pass

        class_file = os.path.join(dir_path, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                for line in f:
                    cls_name = line.strip()
                    if cls_name:
                        color = saved_colors.get(cls_name, None)
                        self.add_class(cls_name, color)

    def save_classes(self):
        if not self._current_dir:
            return
        class_file = os.path.join(self._current_dir, "classes.txt")
        with open(class_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self._class_list))
        self._save_colors()

    def _save_colors(self):
        if not self._current_dir:
            return
        color_file = os.path.join(self._current_dir, "class_colors.json")
        try:
            data = {k: v.name() for k, v in self._class_colors.items()}
            with open(color_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
