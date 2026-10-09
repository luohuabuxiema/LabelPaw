import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QToolBar, QListWidget, QGraphicsView,
                               QLabel, QLineEdit, QPushButton, QStatusBar, QMenu, QComboBox, QSizePolicy, QAbstractItemView, QSplitter, QCheckBox, QFrame)
from ui.annotation_tree_widget import AnnotationTreeWidget
from PySide6.QtCore import Qt, Signal, QRect, QSize
from PySide6.QtGui import QAction, QActionGroup, QPainter, QColor, QFont, QIcon, QPixmap

def create_text_icon(text, color="#94A3B8"):
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QColor(color))
    font = QFont("Segoe UI Emoji", 14)
    painter.setFont(font)
    painter.drawText(QRect(0, 0, 24, 24), Qt.AlignCenter, text)
    painter.end()
    return QIcon(pixmap)


class FormatSelectorWidget(QWidget):
    format_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 5, 5, 5)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.btn = QPushButton()
        self.btn.setIcon(QIcon("ui/icon/格式.svg"))
        self.btn.setIconSize(QSize(20, 20))  # 放大图标
        self.btn.setText("　JSON 格式 ▾")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setToolTip("选择标注格式")
        self.btn.setObjectName("formatBtn")

        # 下拉菜单
        self.menu = QMenu(self)
        self.menu.setWindowFlag(Qt.FramelessWindowHint)
        self.menu.setAttribute(Qt.WA_TranslucentBackground)
        self.menu.setObjectName("formatMenu")

        self.act_json = QAction("JSON 格式", self)
        self.act_yolo = QAction("YOLO 格式", self)
        self.act_xml = QAction("XML 格式", self)

        self.menu.addAction(self.act_json)
        self.menu.addAction(self.act_yolo)
        self.menu.addAction(self.act_xml)

        self.btn.setMenu(self.menu)

        layout.addWidget(self.btn)

        self.act_json.triggered.connect(lambda: self._on_format_selected("json", "　JSON 格式 ▾"))
        self.act_yolo.triggered.connect(lambda: self._on_format_selected("yolo", "　YOLO 格式 ▾"))
        self.act_xml.triggered.connect(lambda: self._on_format_selected("xml", "　XML 格式 ▾"))

    def _on_format_selected(self, fmt, text):
        self.btn.setText(text)
        self.format_changed.emit(fmt)

    def set_yolo_enabled(self, enabled):
        self.act_yolo.setEnabled(enabled)
        if not enabled and self.btn.text().strip() == "YOLO 格式 ▾":
            self._on_format_selected("json", "　JSON 格式 ▾")

    def set_format(self, fmt):
        if fmt == "json":
            self.btn.setText("　JSON 格式 ▾")
        elif fmt == "yolo":
            self.btn.setText("　YOLO 格式 ▾")
        elif fmt == "xml":
            self.btn.setText("　XML 格式 ▾")
            
    def set_icon_only(self, icon_only):
        if icon_only:
            # 记住当前文字以便恢复
            self._cached_text = self.btn.text()
            self.btn.setText("")
            self.btn.setFixedSize(34, 34)
            self.btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid transparent;
                    color: #F8FAFC;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #1E293B;
                    color: #22C55E;
                }
                QPushButton::menu-indicator {
                    image: none; /* 强制隐藏默认箭头 */
                }
            """)
        else:
            if hasattr(self, '_cached_text'):
                self.btn.setText(self._cached_text)
            self.btn.setMaximumWidth(16777215)  # 释放宽度限制
            self.btn.setFixedHeight(34)
            self.btn.setStyleSheet("")


class TemplateSelectorWidget(QWidget):
    template_changed = Signal(str)

    edit_template = Signal(str)
    delete_template = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self.btn = QPushButton("Person (COCO) ▾")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setObjectName("templateBtn")

        # 下拉菜单
        self.menu = QMenu(self)
        self.menu.setWindowFlag(Qt.FramelessWindowHint)
        self.menu.setAttribute(Qt.WA_TranslucentBackground)
        self.menu.setObjectName("templateMenu")
        self.btn.setMenu(self.menu)

        layout.addWidget(self.btn)

    def update_templates(self, templates, main_window=None):
        self.menu.clear()
        
        fixed_templates = ["Person (COCO)", "Hand", "Face (68 pts)", "Rectangle", "Triangle"]

        for t_name in templates:
            if t_name in fixed_templates:
                act = QAction(t_name, self)
                act.triggered.connect(lambda checked=False, name=t_name: self._on_template_selected(name, f"{name} ▾"))
                self.menu.addAction(act)
            else:
                from PySide6.QtWidgets import QWidgetAction, QWidget, QHBoxLayout, QPushButton, QToolButton
                from PySide6.QtGui import QIcon, QColor
                from PySide6.QtCore import Qt

                action = QWidgetAction(self)
                widget = QWidget()
                widget.setStyleSheet("QWidget { background: transparent; }")
                layout = QHBoxLayout(widget)
                layout.setContentsMargins(10, 4, 10, 4)
                layout.setSpacing(5)

                btn_select = QPushButton(t_name)
                btn_select.setStyleSheet("text-align: left; background: transparent; border: none; padding: 2px;")
                btn_select.setCursor(Qt.PointingHandCursor)
                btn_select.clicked.connect(lambda checked=False, name=t_name: self._on_template_selected(name, f"{name} ▾"))
                btn_select.clicked.connect(self.menu.close)

                btn_edit = QToolButton()
                btn_delete = QToolButton()
                
                if main_window:
                    try:
                        btn_edit.setIcon(main_window.set_icon_color(QIcon("ui/icon/编辑.svg"), main_window.current_icon_color))
                        btn_delete.setIcon(main_window.set_icon_color(QIcon("ui/icon/trash.svg"), QColor("#EF4444")))
                    except:
                        pass

                btn_edit.setStyleSheet("QToolButton { background: transparent; border: none; } QToolButton:hover { background-color: rgba(128,128,128,0.2); border-radius: 4px; }")
                btn_delete.setStyleSheet("QToolButton { background: transparent; border: none; } QToolButton:hover { background-color: rgba(128,128,128,0.2); border-radius: 4px; }")
                btn_edit.setCursor(Qt.PointingHandCursor)
                btn_delete.setCursor(Qt.PointingHandCursor)

                btn_edit.clicked.connect(lambda checked=False, name=t_name: self.edit_template.emit(name))
                btn_edit.clicked.connect(self.menu.close)

                btn_delete.clicked.connect(lambda checked=False, name=t_name: self.delete_template.emit(name))
                btn_delete.clicked.connect(self.menu.close)

                layout.addWidget(btn_select, 1)
                layout.addWidget(btn_edit)
                layout.addWidget(btn_delete)
                action.setDefaultWidget(widget)
                self.menu.addAction(action)
            
        self.menu.addSeparator()
        
        act_new = QAction("+ New Template...", self)
        act_new.triggered.connect(lambda: self._on_template_selected("+ New Template...", self.btn.text()))
        self.menu.addAction(act_new)

    def _on_template_selected(self, template_name, btn_text):
        if template_name != "+ New Template...":
            self.btn.setText(btn_text)
        self.template_changed.emit(template_name)
        
    def set_current_template_text(self, text):
        self.btn.setText(f"{text} ▾")


class SwitchControl(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._checked = False
        self._vertical = False  # 竖向模式

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRect(0, 0, self.width(), self.height())

        if self._checked:
            p.setBrush(QColor("#22C55E"))
        else:
            p.setBrush(QColor("#334155"))

        p.setPen(Qt.NoPen)
        radius = min(self.width(), self.height()) // 2
        p.drawRoundedRect(rect, radius, radius)

        p.setBrush(QColor("#FFFFFF"))
        if self._vertical:
            # 竖向模式：圆球上下滑动
            circle_size = self.width() - 4
            if self._checked:
                p.drawEllipse(2, self.height() - circle_size - 2, circle_size, circle_size)
            else:
                p.drawEllipse(2, 2, circle_size, circle_size)
        else:
            # 横向模式：圆球左右滑动
            if self._checked:
                p.drawEllipse(self.width() - 24, 2, 22, 22)
            else:
                p.drawEllipse(2, 2, 22, 22)


class CanvasView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.CrossCursor)

        self._is_panning = False
        self._pan_start_pos = None

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl + 滚轮：缩放
            zoom_in_factor = 1.15
            zoom_out_factor = 1.0 / zoom_in_factor
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.scale(zoom_out_factor, zoom_out_factor)
        else:
            # 普通滚轮：滚动画布（仅放大后才生效，未放大时滚动条范围为 0 自动无效）
            # 垂直滚动
            if event.angleDelta().y() != 0:
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - event.angleDelta().y()
                )
            # 水平滚动（支持鼠标左右滚轮）
            if event.angleDelta().x() != 0:
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - event.angleDelta().x()
                )

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = event.position().toPoint()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.viewport().setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """鼠标离开画布时隐藏十字虚线"""
        scene = self.scene()
        if scene and hasattr(scene, 'h_line'):
            scene.h_line.hide()
            scene.v_line.hide()
        super().leaveEvent(event)

    def enterEvent(self, event):
        """鼠标进入画布时显示十字虚线"""
        scene = self.scene()
        if scene and hasattr(scene, 'h_line') and scene.img_item:
            scene.h_line.show()
            scene.v_line.show()
        super().enterEvent(event)

    def resizeEvent(self, event):
        """窗口大小变化时自适应图片（仅改变视图变换，不影响标注坐标）"""
        super().resizeEvent(event)
        scene = self.scene()
        if scene and scene.sceneRect().width() > 0:
            self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)


class Ui_MainWindow(object):
    def set_icon_color(self, icon, color):
        from PySide6.QtGui import QPainter, QIcon, QPixmap, QColor
        pixmap = icon.pixmap(100, 100)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        
        # 核心逻辑：自己创建一个可以响应 enable/disable 的 QIcon
        # 我们把这个纯色图作为 Normal 状态
        new_icon = QIcon()
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        
        # 为了让禁用状态变灰，我们用 QPainter 绘制一个半透明的版本作为 Disabled 状态
        disabled_pixmap = icon.pixmap(100, 100)
        dpainter = QPainter(disabled_pixmap)
        dpainter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        
        # 深色主题下禁用颜色为半透明白，浅色主题下为半透明黑
        if color.lightness() > 128:
            disabled_color = QColor(255, 255, 255, 60)
        else:
            disabled_color = QColor(15, 23, 42, 60)
            
        dpainter.fillRect(disabled_pixmap.rect(), disabled_color)
        dpainter.end()
        
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.On)
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.Off)
        
        return new_icon

    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("LabelPaw - 基于SAM3的智能标注系统")
        MainWindow.resize(1280, 800)

        # ================= Top Nav buttons (will be placed in central layout) =================
        font_icon = QFont("Segoe UI", 16, QFont.Bold)

        # Collapse Button
        self.btnCollapse = QPushButton("≡")
        self.btnCollapse.setFont(font_icon)
        self.btnCollapse.setToolTip("折叠/展开侧边栏")
        self.btnCollapse.setFixedSize(36, 36)
        self.btnCollapse.setObjectName("btnTopBar")

        # Theme Toggle Button
        self.btnThemeToggle = QPushButton("☀")
        self.btnThemeToggle.setFont(font_icon)
        self.btnThemeToggle.setToolTip("切换亮/暗色模式")
        self.btnThemeToggle.setFixedSize(36, 36)
        self.btnThemeToggle.setObjectName("btnTopBar")

        # Author Info Button
        self.btnAuthorInfo = QPushButton("ⓘ")
        self.btnAuthorInfo.setFont(font_icon)
        self.btnAuthorInfo.setToolTip("作者信息")
        self.btnAuthorInfo.setFixedSize(36, 36)
        self.btnAuthorInfo.setObjectName("btnTopBar")

        self.btnDatasetTool = QPushButton("数据集处理")
        self.btnDatasetTool.setStyleSheet("""
            QPushButton {
                background-color: #22C55E;
                color: #020617;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4ade80; }
        """)

        self.centralWidget = QWidget(MainWindow)
        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        # ================= Custom Top Bar (full width, above everything) =================
        self.topBar = QWidget()
        self.topBar.setObjectName("topBar")
        self.topBar.setFixedHeight(50)
        topBarLayout = QHBoxLayout(self.topBar)
        topBarLayout.setContentsMargins(10, 0, 10, 0)
        topBarLayout.setSpacing(8)
        topBarLayout.addWidget(self.btnCollapse)
        topBarLayout.addStretch()
        topBarLayout.addWidget(self.btnAuthorInfo)
        topBarLayout.addWidget(self.btnThemeToggle)
        self.mainLayout.addWidget(self.topBar)
        
        # ================= Content Area (Splitter: Canvas + Right Panel) =================
        from PySide6.QtWidgets import QSplitter
        self.contentSplitter = QSplitter(Qt.Horizontal)
        self.contentSplitter.setObjectName("contentSplitter")
        self.contentSplitter.setStyleSheet("QSplitter::handle { background: transparent; width: 1px; }")
        
        # --- Left: Canvas Area ---
        self.canvasArea = QWidget()
        self.canvasArea.setObjectName("canvasArea")
        canvasLayout = QVBoxLayout(self.canvasArea)
        canvasLayout.setContentsMargins(0, 5, 0, 0)
        canvasLayout.setSpacing(5)
        
        # ================= Annotation Toolbar =================
        self.annotationToolbar = QWidget()
        self.annotationToolbar.setObjectName("annotationToolbar")
        self.annotationToolbar.setFixedHeight(40)
        # 居中显示
        tb_layout_wrap = QHBoxLayout()
        tb_layout_wrap.setContentsMargins(0, 0, 0, 0)
        tb_layout_wrap.addStretch()
        
        tb_layout = QHBoxLayout(self.annotationToolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)
        tb_layout.setSpacing(10)
        
        self.btnDrawMode = QPushButton("✍ 手动标注")
        self.btnDrawMode.setCheckable(True)
        self.btnDrawMode.setAutoExclusive(True)
        self.btnDrawMode.setChecked(True)

        self.btnSmartMode = QPushButton("✨ 智能")
        self.btnSmartMode.setCheckable(True)
        self.btnSmartMode.setAutoExclusive(True)        
        # 模型选择器按钮（Smart 模式时显示）
        self.btnModelSelector = QPushButton(" SAM 3 ▾")
        self.btnModelSelector.setObjectName("btnModelSelector")
        self.btnModelSelector.setCursor(Qt.PointingHandCursor)
        self.btnModelSelector.setToolTip("切换 SAM 模型")
        self.btnModelSelector.hide()  # 默认隐藏，Smart 激活时显示

        self.btnPredict = QPushButton(" 预测")
        self.btnPredict.setObjectName("btnPredict")
        self.btnPredict.setCursor(Qt.PointingHandCursor)
        self.btnPredict.setToolTip("使用模型进行预测 (快捷键: M)")
        self.btnPredict.hide()  # 仅在非SAM模型下显示

        self.btnClassFilter = QPushButton(" 全部类别 ▾")
        self.btnClassFilter.setObjectName("btnClassFilter")
        self.btnClassFilter.setCursor(Qt.PointingHandCursor)
        self.btnClassFilter.setToolTip("选择YOLO模型预测的类别过滤")
        self.btnClassFilter.hide()  # 仅在加载YOLO模型下显示
        
        # Segmented toggle effect
        self.btnDrawMode.toggled.connect(lambda checked: self.btnSmartMode.setChecked(not checked) if checked else None)
        self.btnSmartMode.toggled.connect(lambda checked: self.btnDrawMode.setChecked(not checked) if checked else None)
        
        self.templateWidget = TemplateSelectorWidget()
        self.templateWidget.hide() # Only show when Keypoint mode is active
        
        from PySide6.QtWidgets import QToolButton
        
        # 默认明亮主题的颜色 (深灰色)
        self.current_icon_color = QColor(15, 23, 42)
        
        self.btnUndo = QToolButton()
        self.btnUndo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-left.svg"), self.current_icon_color))
        self.btnUndo.setToolTip("撤销 (Ctrl+Z)")
        self.btnUndo.setFixedSize(36, 36)
        
        self.btnRedo = QToolButton()
        self.btnRedo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-right.svg"), self.current_icon_color))
        self.btnRedo.setToolTip("重做 (Ctrl+Shift+Z)")
        self.btnRedo.setFixedSize(36, 36)
        
        self.btnDelete = QToolButton()
        self.btnDelete.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
        self.btnDelete.setToolTip("删除 (Del)")
        self.btnDelete.setFixedSize(36, 36)
        
        self.btnSave = QToolButton()
        self.btnSave.setIcon(self.set_icon_color(QIcon("ui/icon/floppy-disk.svg"), self.current_icon_color))
        self.btnSave.setToolTip("保存 (Ctrl+S)")
        self.btnSave.setFixedSize(36, 36)
        
        self.btnKeyboard = QToolButton()
        self.btnKeyboard.setIcon(self.set_icon_color(QIcon("ui/icon/keyboard.svg"), self.current_icon_color))
        self.btnKeyboard.setToolTip("快捷键大全 (F1)")
        self.btnKeyboard.setFixedSize(36, 36)
        
        for btn in [self.btnUndo, self.btnRedo, self.btnDelete, self.btnSave, self.btnKeyboard]:
            btn.setStyleSheet("QToolButton { border: none; background: transparent; border-radius: 6px; } QToolButton:hover { background-color: rgba(128, 128, 128, 0.2); }")

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #475569; font-weight: bold;")
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #475569; font-weight: bold;")
        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #475569; font-weight: bold;")
        
        tb_layout.addWidget(self.btnDrawMode)
        tb_layout.addWidget(self.btnSmartMode)
        tb_layout.addWidget(self.btnModelSelector)
        tb_layout.addWidget(self.btnPredict)
        tb_layout.addWidget(self.btnClassFilter)
        tb_layout.addWidget(sep1)
        tb_layout.addWidget(self.templateWidget)
        self.sepTemplate = QLabel("|")
        self.sepTemplate.setStyleSheet("color: #475569; font-weight: bold;")
        self.sepTemplate.hide()
        tb_layout.addWidget(self.sepTemplate)
        tb_layout.addWidget(self.btnUndo)
        tb_layout.addWidget(self.btnRedo)
        tb_layout.addWidget(self.btnDelete)
        tb_layout.addWidget(self.btnSave)
        tb_layout.addWidget(sep3)
        tb_layout.addWidget(self.btnKeyboard)
        
        tb_layout_wrap.addWidget(self.annotationToolbar)
        tb_layout_wrap.addStretch()
        canvasLayout.addLayout(tb_layout_wrap)
        # =======================================================

        self.view = CanvasView()
        canvasLayout.addWidget(self.view)
        
        self.contentSplitter.addWidget(self.canvasArea)
        
        # --- Right: Annotation Panel ---
        self.rightPanel = QWidget()
        self.rightPanel.setObjectName("rightPanel")
        self.rightPanel.setMinimumWidth(180)
        self.rightPanel.setMaximumWidth(300)
        self.dockLayout = QVBoxLayout(self.rightPanel)
        self.dockLayout.setContentsMargins(8, 8, 8, 8)

        # 标注管理标题栏
        rightTitleBar = QHBoxLayout()
        self.rightPanelTitle = QLabel("标注管理")
        self.rightPanelTitle.setObjectName("rightPanelTitle")
        titleFont = QFont("Microsoft YaHei", 10, QFont.Bold)
        self.rightPanelTitle.setFont(titleFont)
        rightTitleBar.addWidget(self.rightPanelTitle)
        rightTitleBar.addStretch()
        self.dockLayout.addLayout(rightTitleBar)

        # QSplitter to allow resizing historical categories and file list
        self.rightSplitter = QSplitter(Qt.Vertical)
        self.rightSplitter.setObjectName("rightSplitter")

        # Container for classes
        self.classesContainer = QWidget()
        self.classesContainer.setObjectName("classesContainer")
        classesLayout = QVBoxLayout(self.classesContainer)
        classesLayout.setContentsMargins(0, 0, 0, 8)
        self.labelClasses = QLabel("历史类别:")
        self.classListWidget = AnnotationTreeWidget()
        self.listClasses = self.classListWidget.listWidget  # 兼容旧引用
        classesLayout.addWidget(self.labelClasses)
        classesLayout.addWidget(self.classListWidget)

        # Container for files
        self.filesContainer = QWidget()
        self.filesContainer.setObjectName("filesContainer")
        filesLayout = QVBoxLayout(self.filesContainer)
        filesLayout.setContentsMargins(0, 0, 0, 0)
        filesLayout.setSpacing(6)
        
        # 文件列表头部布局 (复选框 + 计数标签)
        self.filesHeader = QWidget()
        self.filesHeader.setObjectName("filesHeader")
        filesHeaderLayout = QHBoxLayout(self.filesHeader)
        filesHeaderLayout.setContentsMargins(0, 0, 0, 4)
        filesHeaderLayout.setSpacing(6)
        
        self.chkSelectAll = QCheckBox("全选")
        self.chkSelectAll.setObjectName("chkSelectAll")
        self.chkSelectAll.setCursor(Qt.PointingHandCursor)
        self.chkSelectAll.setTristate(False)

        self.btnOverwrite = QPushButton("覆盖")
        self.btnOverwrite.setObjectName("btnOverwrite")
        self.btnOverwrite.setCheckable(True)
        self.btnOverwrite.setCursor(Qt.PointingHandCursor)
        self.btnOverwrite.setToolTip("覆盖已有标注框")
        self.btnOverwrite.setFixedHeight(22)

        self.btnDeleteFiles = QPushButton()
        self.btnDeleteFiles.setObjectName("btnDeleteFiles")
        self.btnDeleteFiles.setCursor(Qt.PointingHandCursor)
        self.btnDeleteFiles.setIcon(QIcon("ui/icon/trash.svg"))
        self.btnDeleteFiles.setIconSize(QSize(13, 13))
        self.btnDeleteFiles.setToolTip("批量删除选中的图片")
        self.btnDeleteFiles.setFixedSize(22, 22)
        self.btnDeleteFiles.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.2);
            }
        """)
        
        self.labelFiles = QLabel("文件列表")
        self.labelFiles.setObjectName("labelFiles")
        self.labelFiles.setStyleSheet("font-weight: bold;")
        
        self.labelSelectedCount = QLabel("(已选 0/0)")
        self.labelSelectedCount.setObjectName("labelSelectedCount")
        self.labelSelectedCount.setStyleSheet("color: #64748B; font-size: 11px;")
        
        filesHeaderLayout.addWidget(self.chkSelectAll)
        filesHeaderLayout.addWidget(self.btnOverwrite)
        filesHeaderLayout.addWidget(self.btnDeleteFiles)
        filesHeaderLayout.addWidget(self.labelFiles)
        filesHeaderLayout.addWidget(self.labelSelectedCount)
        filesHeaderLayout.addStretch()
        
        self.listFiles = QListWidget()
        self.listFiles.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listFiles.setContextMenuPolicy(Qt.CustomContextMenu)
        
        filesLayout.addWidget(self.filesHeader)
        filesLayout.addWidget(self.listFiles)

        # Add both to splitter
        self.rightSplitter.addWidget(self.classesContainer)
        self.rightSplitter.addWidget(self.filesContainer)
        
        # Set initial stretch/sizes
        self.rightSplitter.setStretchFactor(0, 3)
        self.rightSplitter.setStretchFactor(1, 2)

        self.dockLayout.addWidget(self.rightSplitter)

        # ==== 右下角区域 ====
        # 提示词输入与提取按钮的精美卡片式聊天框 (DeepSeek / Gemini 3 Style)
        self.samTextGroup = QFrame()
        self.samTextGroup.setObjectName("samTextGroup")
        
        textLayout = QVBoxLayout(self.samTextGroup)
        textLayout.setContentsMargins(12, 10, 10, 8)
        textLayout.setSpacing(6)

        # 顶部输入框：无边框，透明背景
        self.samPromptInput = QLineEdit()
        self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog)")
        self.samPromptInput.setObjectName("samPromptInput")
        self.samPromptInput.setFrame(False)
        textLayout.addWidget(self.samPromptInput)

        # 底部操作栏
        bottomLayout = QHBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)

        bottomLayout.addStretch()

        self.samPromptBtn = QPushButton()
        self.samPromptBtn.setObjectName("samPromptBtn")
        self.samPromptBtn.setCursor(Qt.PointingHandCursor)
        self.samPromptBtn.setFixedSize(28, 28)
        bottomLayout.addWidget(self.samPromptBtn)

        textLayout.addLayout(bottomLayout)

        self.dockLayout.addWidget(self.samTextGroup)

        self.contentSplitter.addWidget(self.rightPanel)
        
        # 设置 splitter 初始比例 (canvas 占大部分)
        self.contentSplitter.setStretchFactor(0, 1)
        self.contentSplitter.setStretchFactor(1, 0)
        
        self.mainLayout.addWidget(self.contentSplitter)
        MainWindow.setCentralWidget(self.centralWidget)

        self.statusBar = QStatusBar()
        MainWindow.setStatusBar(self.statusBar)
        self.coordLabel = QLabel("坐标: X: 0, Y: 0")
        self.statusBar.addPermanentWidget(self.coordLabel)

        self.toolBar = QToolBar("工具栏")
        self.toolBar.setOrientation(Qt.Vertical)
        self.toolBar.setMovable(False)
        self.toolBar.setFixedWidth(190)  # 固定宽度
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolBar.setIconSize(QSize(24, 24))  # 统一图标尺寸
        MainWindow.addToolBar(Qt.LeftToolBarArea, self.toolBar)

        # 初始图标颜色 (浅色主题用深色图标)
        _ic = QColor(15, 23, 42)
        self.actionOpen = QAction(self.set_icon_color(QIcon("ui/icon/folder.svg"), _ic), "打开目录", MainWindow)
        self.actionRect = QAction(self.set_icon_color(QIcon("ui/icon/rectangle.svg"), _ic), "矩形标注 (R)", MainWindow)
        self.actionPoly = QAction(self.set_icon_color(QIcon("ui/icon/polygon.svg"), _ic), "多边形标注 (P)", MainWindow)
        self.actionPoint = QAction(self.set_icon_color(QIcon("ui/icon/关键点.svg"), _ic), "关键点标注 (T)", MainWindow)
        self.actionRBox = QAction(self.set_icon_color(QIcon("ui/icon/手机旋转1.svg"), _ic), "旋转框标注 (O)", MainWindow)

        self.modeGroup = QActionGroup(MainWindow)
        for act in [self.actionRect, self.actionPoly, self.actionPoint, self.actionRBox]:
            act.setCheckable(True)
            self.modeGroup.addAction(act)

        self.actionRect.setChecked(True)

        # Brand Logo (顶部，在左侧工具栏内)
        self.logoWidget = QWidget()
        self.logoWidget.setObjectName("logoWidget")
        logoLayout = QHBoxLayout(self.logoWidget)
        logoLayout.setContentsMargins(8, 8, 4, 8)
        logoLayout.setSpacing(5)

        self.logoIcon = QLabel()
        self.logoIcon.setObjectName("logoIcon")
        self.logoIcon.setFixedSize(28, 28)
        self.logoIcon.setAlignment(Qt.AlignCenter)
        
        self.logoLabel = QLabel("LabelPaw")
        self.logoLabel.setObjectName("logoLabel")
        self.logoLabel.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        
        # 尝试初始化加载 Logo 图标
        logo_path = "ui/icon/logo.png"
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                self.logoIcon.setPixmap(pix.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        logoLayout.addWidget(self.logoIcon)
        logoLayout.addWidget(self.logoLabel)
        logoLayout.addStretch()
        self.logoWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolBar.addWidget(self.logoWidget)

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addSeparator()

        self.formatWidget = FormatSelectorWidget()
        self.formatWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.formatWidget.btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toolBar.addWidget(self.formatWidget)
        self.toolBar.addSeparator()

        self.toolBar.addAction(self.actionRect)
        self.toolBar.addAction(self.actionPoly)
        self.toolBar.addAction(self.actionPoint)
        self.toolBar.addAction(self.actionRBox)

        # 统一所有 QToolButton 宽度（类似前端 width:100%），使图标垂直对齐
        self._actionButtons = []
        for action in [self.actionOpen, self.actionRect, self.actionPoly, self.actionPoint, self.actionRBox]:
            btn = self.toolBar.widgetForAction(action)
            if btn:
                btn.setMinimumWidth(180)  # 强制撑满工具栏宽度
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                self._actionButtons.append(btn)

        self.toolBar.addSeparator()

        # SAM 组件（图标 + 开关）
        self.samWidget = QWidget()
        self.samWidget.setStyleSheet("background-color: transparent;")
        samOuterLayout = QHBoxLayout(self.samWidget)
        samOuterLayout.setContentsMargins(8, 5, 4, 5)
        samOuterLayout.setSpacing(10)
        samOuterLayout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # SAM 图标 (左侧)
        self.samIcon = QLabel()
        self.samIcon.setFixedSize(24, 24)
        self.samIcon.setAlignment(Qt.AlignCenter)
        self.samIcon.setPixmap(
            self.set_icon_color(QIcon("ui/icon/魔法-copy.svg"), _ic).pixmap(24, 24)
        )
        self.samIcon.setToolTip("SAM 智能辅助")

        # SAM 开关 (右侧)
        self.samSwitch = SwitchControl()
        self.samSwitch.setToolTip("开启/关闭 SAM 智能辅助")

        samOuterLayout.addWidget(self.samIcon)
        samOuterLayout.addWidget(self.samSwitch)
        
        self.samWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolBar.addWidget(self.samWidget)

        # ==========================================
        # 数据集处理按钮
        # ==========================================
        self.toolBar.addSeparator()

        self.btnDatasetTool = QPushButton(self.set_icon_color(QIcon("ui/icon/wrench.svg"), _ic), "一键训练平台")
        self.btnDatasetTool.setCursor(Qt.PointingHandCursor)
        self.btnDatasetTool.setObjectName("btnDatasetTool")
        self.btnDatasetTool.setToolTip("一键训练平台")
        self.btnDatasetTool.setIconSize(QSize(24, 24))
        self.btnDatasetTool.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolBar.addWidget(self.btnDatasetTool)