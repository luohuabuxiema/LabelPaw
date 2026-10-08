# -*- coding: utf-8 -*-
"""
Interactive Prediction & Inference UI Components for AI Vision Training Platform
Benchmark: Images 1, 2, 4, 5:
- Dropzone upload card with dashed border, drag-and-drop, and click-to-browse (Image 5)
- Rotating circular loading spinner overlay with Chinese text (Image 4)
- Built-in example thumbnail cards (bus.jpg, zidane.jpg, browse card) with click-to-predict (Images 1, 2, 5)
- Floating action buttons on top-right of preview (download, re-upload, clear) (Image 1)
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QToolButton, QFileDialog, QSizePolicy, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF, QSize, QEvent
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QPixmap, QImage,
    QCursor, QBrush, QLinearGradient
)
from monkeyqt_icons import (
    PhUploadSimple, PhDownloadSimple, PhX, PhArrowsOutSimple,
    PhImage, PhFolderOpen, PhSparkle
)
from monkeyqt import ThemeEngine, MkQWidget
from TrainingPlatform.ui.components.custom_tooltip import attach_tooltip


class LoadingOverlayWidget(MkQWidget):
    """
    Semi-transparent loading overlay with smooth animated circular spinner
    and Chinese prompt text (Matching Figure 4).
    """
    def __init__(self, text="正在进行模型推理预测...", parent=None):
        super().__init__(parent=parent, role="transparent")
        self.text = text
        self.angle = 0
        self.is_dark = ThemeEngine.is_dark()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        if parent:
            parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self.update()

    def start(self, text=None):
        if text:
            self.text = text
        self.angle = 0
        self.timer.start(25)  # 40 FPS smooth rotation
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self.update()

    def stop(self):
        self.timer.stop()
        self.hide()

    def _rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Semi-transparent frosted backdrop (Figure 4)
        bg = QColor(15, 23, 42, 195) if self.is_dark else QColor(255, 255, 255, 210)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)

        # 2. Spinner coordinates
        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 14.0
        r = 22.0

        # Background track
        track_color = QColor(71, 85, 105, 80) if self.is_dark else QColor(203, 213, 225, 140)
        painter.setPen(QPen(track_color, 3.8))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Active rotating arc
        arc_color = QColor(96, 165, 250) if self.is_dark else QColor(37, 99, 235)
        pen_arc = QPen(arc_color, 3.8)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)

        # 100 degree rotating arc
        start_angle = int(-self.angle * 16)
        span_angle = int(-100 * 16)
        arc_rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        painter.drawArc(arc_rect, start_angle, span_angle)

        # 3. Chinese Prompt Label (Matching Figure 4)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        text_color = QColor(226, 232, 240) if self.is_dark else QColor(30, 41, 59)
        painter.setPen(text_color)
        text_rect = QRectF(0, cy + r + 16.0, self.width(), 26.0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, self.text)


class DropzoneWidget(MkQWidget):
    """
    Dropzone upload card with dashed border, drag-and-drop, and click-to-browse (Image 5).
    """
    imageSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent, role="card")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_dark = ThemeEngine.is_dark()
        self.is_drag_hover = False
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("DropzoneWidget")
        self.setMinimumHeight(380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self.lbl_icon = QLabel()
        self.lbl_icon.setPixmap(PhUploadSimple.regular(size=40, color="#3b82f6").pixmap(40, 40))
        _MkThemeEngine.apply_style_sheet(self.lbl_icon, "background: transparent;")
        layout.addWidget(self.lbl_icon, alignment=Qt.AlignmentFlag.AlignCenter)

        # Main Title
        self.lbl_title = QLabel("拖拽图片至此处 (Drop an image here)")
        self.lbl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        # Secondary action hint
        self.lbl_sub = QLabel("或点击浏览本地文件 (or click to browse)")
        self.lbl_sub.setFont(QFont("Segoe UI", 10))
        self.lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_sub)

        # Supported formats
        self.lbl_formats = QLabel("支持格式: JPG, JPEG, PNG, WebP, BMP 等常见图像格式 (上传后立即预测)")
        self.lbl_formats.setFont(QFont("Segoe UI", 9))
        self.lbl_formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_formats)

        self._apply_style()

    def _apply_style(self):
        if self.is_drag_hover:
            border_col = "#2563eb"
            bg_col = "#eff6ff" if not self.is_dark else "#1e293b"
        else:
            border_col = "#3b82f6" if self.underMouse() else ("#334155" if self.is_dark else "#cbd5e1")
            bg_col = "#1e293b" if self.is_dark else "#f8fafc"

        title_color = "#f1f5f9" if self.is_dark else "#0f172a"
        sub_color = "#94a3b8" if self.is_dark else "#475569"
        format_color = "#64748b" if self.is_dark else "#94a3b8"

        _MkThemeEngine.apply_style_sheet(self, f"""
            QFrame#DropzoneWidget {{
                background-color: {bg_col};
                border: 2px dashed {border_col};
                border-radius: 12px;
            }}
        """)
        _MkThemeEngine.apply_style_sheet(self.lbl_title, f"color: {title_color}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_sub, f"color: {sub_color}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_formats, f"color: {format_color}; background: transparent;")

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self._apply_style()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._apply_style()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择待预测图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
            )
            if path:
                self.imageSelected.emit(path)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                        self.is_drag_hover = True
                        self._apply_style()
                        event.acceptProposedAction()
                        return
        super().dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        self.is_drag_hover = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.is_drag_hover = False
        self._apply_style()
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    p = url.toLocalFile()
                    ext = os.path.splitext(p)[1].lower()
                    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                        self.imageSelected.emit(p)
                        event.acceptProposedAction()
                        return
        super().dropEvent(event)


class ExampleThumbCard(MkQWidget):
    """
    Built-in Example Thumbnail Card (bus.jpg, zidane.jpg, or Upload Card).
    Matching Figure 1, 2, 5. Clicking triggers immediate prediction.
    """
    clicked = Signal(str)

    def __init__(self, image_path: str = "", title: str = "", is_browse_card: bool = False, parent=None):
        super().__init__(parent=parent, role="card")
        self.image_path = image_path
        self.title = title or os.path.basename(image_path)
        self.is_browse_card = is_browse_card
        self.is_selected = False
        self.is_dark = ThemeEngine.is_dark()
        self.setFixedSize(140, 92)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("ExampleThumbCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if self.is_browse_card:
            # Browse action card (like Camera/Webcam card in Figure 5)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_icon = QLabel()
            self.lbl_icon.setPixmap(PhFolderOpen.regular(size=24, color="#64748b").pixmap(24, 24))
            layout.addWidget(self.lbl_icon, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addSpacing(4)

            self.lbl_title = QLabel("选择本地图片")
            self.lbl_title.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.lbl_title, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            # Thumbnail with dark bottom gradient & filename
            self.thumb_label = QLabel()
            self.thumb_label.setScaledContents(True)
            layout.addWidget(self.thumb_label, stretch=1)

            # Bottom gradient bar for filename
            self.bottom_bar = QFrame()
            self.bottom_bar.setFixedHeight(26)
            b_layout = QHBoxLayout(self.bottom_bar)
            b_layout.setContentsMargins(8, 0, 8, 0)

            self.lbl_title = QLabel(self.title)
            self.lbl_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            _MkThemeEngine.apply_style_sheet(self.lbl_title, "color: #ffffff; background: transparent;")
            b_layout.addWidget(self.lbl_title)
            layout.addWidget(self.bottom_bar)

            self._load_thumbnail()

        self._apply_style()

    def _load_thumbnail(self):
        if not self.image_path or not os.path.exists(self.image_path):
            return
        pix = QPixmap(self.image_path)
        if not pix.isNull():
            scaled = pix.scaled(
                QSize(140, 92),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            self.thumb_label.setPixmap(scaled)

    def set_selected(self, selected: bool):
        self.is_selected = selected
        self._apply_style()

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self._apply_style()

    def _apply_style(self):
        border = "#2563eb" if (self.is_selected or self.underMouse()) else ("#334155" if self.is_dark else "#e2e8f0")
        border_w = "2px" if (self.is_selected or self.underMouse()) else "1px"
        bg = "#1e293b" if self.is_dark else "#f8fafc"

        if self.is_browse_card:
            _MkThemeEngine.apply_style_sheet(self, f"""
                QFrame#ExampleThumbCard {{
                    background-color: {bg};
                    border: {border_w} dashed {border};
                    border-radius: 8px;
                }}
            """)
            if hasattr(self, "lbl_title"):
                col = "#94a3b8" if self.is_dark else "#475569"
                _MkThemeEngine.apply_style_sheet(self.lbl_title, f"color: {col}; background: transparent;")
        else:
            _MkThemeEngine.apply_style_sheet(self, f"""
                QFrame#ExampleThumbCard {{
                    background-color: #0f172a;
                    border: {border_w} solid {border};
                    border-radius: 8px;
                }}
            """)
            if hasattr(self, "bottom_bar"):
                _MkThemeEngine.apply_style_sheet(self.bottom_bar, """
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 rgba(0, 0, 0, 0),
                                                stop:1 rgba(0, 0, 0, 180));
                    border-bottom-left-radius: 7px;
                    border-bottom-right-radius: 7px;
                """)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._apply_style()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_browse_card:
                path, _ = QFileDialog.getOpenFileName(
                    self, "选择本地图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
                )
                if path:
                    self.clicked.emit(path)
            else:
                self.clicked.emit(self.image_path)
        super().mousePressEvent(event)


class FloatingToolbarWidget(MkQWidget):
    """
    Floating action pill toolbar on the top-right corner of prediction image preview (Figures 1 & 2).
    Buttons:
    - Download / Save result image
    - Replace / Upload new image
    - Clear / Reset to dropzone
    """
    downloadClicked = Signal()
    replaceClicked = Signal()
    clearClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent, role="card")
        self.setObjectName("FloatingToolbar")
        self.is_dark = ThemeEngine.is_dark()
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Download Button
        self.btn_download = QToolButton()
        self.btn_download.setIcon(PhDownloadSimple.regular(size=18, color="#f8fafc" if self.is_dark else "#334155"))
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_download, "保存当前推理结果图片")
        self.btn_download.clicked.connect(self.downloadClicked.emit)
        layout.addWidget(self.btn_download)

        # Replace / Re-upload Button
        self.btn_replace = QToolButton()
        self.btn_replace.setIcon(PhUploadSimple.regular(size=18, color="#f8fafc" if self.is_dark else "#334155"))
        self.btn_replace.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_replace, "更换/上传新图片")
        self.btn_replace.clicked.connect(self.replaceClicked.emit)
        layout.addWidget(self.btn_replace)

        # Clear / Close Button
        self.btn_clear = QToolButton()
        self.btn_clear.setIcon(PhX.regular(size=18, color="#f8fafc" if self.is_dark else "#334155"))
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_clear, "清除图片并返回上传界面")
        self.btn_clear.clicked.connect(self.clearClicked.emit)
        layout.addWidget(self.btn_clear)

        self._apply_style()

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        icon_col = "#f8fafc" if is_dark else "#334155"
        self.btn_download.setIcon(PhDownloadSimple.regular(size=18, color=icon_col))
        self.btn_replace.setIcon(PhUploadSimple.regular(size=18, color=icon_col))
        self.btn_clear.setIcon(PhX.regular(size=18, color=icon_col))
        self._apply_style()

    def _apply_style(self):
        bg = "rgba(15, 23, 42, 0.88)" if self.is_dark else "rgba(255, 255, 255, 0.95)"
        border = "rgba(51, 65, 85, 0.7)" if self.is_dark else "rgba(203, 213, 225, 0.9)"
        btn_hover = "rgba(51, 65, 85, 0.9)" if self.is_dark else "rgba(241, 245, 249, 1.0)"
        _MkThemeEngine.apply_style_sheet(self, f"""
            QFrame#FloatingToolbar {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QToolButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: {btn_hover};
            }}
        """)

