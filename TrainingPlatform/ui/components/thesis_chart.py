# -*- coding: utf-8 -*-
"""
Thesis / Academic Style Plot Component for AI Vision Training Platform
Strictly benchmarked against Master's thesis / publication figures and modern frontend standards (Images 1, 2, 3):
- Modern rounded card container (border-radius: 12px, border: 1px solid #e2e8f0)
- Action buttons: Fullscreen and More with dropdown menu
- Clean white background with subtle horizontal dashed grid lines
- Despined top & right borders (academic standard)
- Classical publication color palettes (vibrant blue, terracotta, emerald, purple)
- Interactive hover effects:
  - Vertical crosshair guide line following cursor epoch
  - Halo dot marker highlighting data point on line
  - Modern floating tooltip bubble with shadow and rich text (x: 145, ● yolo26n: 0.639)
- Fullscreen expand modal with zoom/pan and 300 DPI high-res export
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
from typing import Dict, List, Optional, Tuple
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QToolButton, QDialog, QFileDialog, QSizePolicy,
                               QFrame, QGraphicsDropShadowEffect, QMenu, QApplication,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QRectF, QTimer
from PySide6.QtGui import QFont, QColor, QCursor, QPainter, QImage, QBrush, QLinearGradient

import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from monkeyqt import MkQWidget
from monkeyqt_icons import (PhArrowsOutSimple, PhDownloadSimple, PhX, PhChartLine, PhDotsThree,
                            PhCamera, PhCornersOut, PhCopy)


# Modern publication colors matching Ultralytics / Tailwind / Academic papers
ACADEMIC_COLORS = [
    "#ef4444",  # Crimson Red (matching Figure 1)
    "#2563eb",  # Vibrant Ultralytics Blue
    "#10b981",  # Emerald Green
    "#f97316",  # Warm Orange / Terracotta
    "#8b5cf6",  # Purple Violet
    "#ec4899",  # Modern Rose Pink
    "#06b6d4",  # Cyan Teal
    "#eab308",  # Amber Gold
    "#64748b",  # Slate Gray
]


class ChartHoverTooltip(QFrame):
    """Modern floating tooltip card benchmarked against Ultralytics HUB / ECharts (Image 2 & 3)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setObjectName("ChartHoverTooltip")
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(0, 0, 0, 32))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)
        
        self.lbl_content = QLabel(self)
        self.lbl_content.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.lbl_content)
        
        self.set_theme(False)

    def set_theme(self, is_dark: bool):
        bg = "#18181b" if is_dark else "#ffffff"
        border = "#27272a" if is_dark else "#e2e8f0"
        _MkThemeEngine.apply_style_sheet(self, f"""
            QFrame#ChartHoverTooltip {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel {{
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Microsoft YaHei", sans-serif;
            }}
        """)


class ThesisChartCanvas(FigureCanvas):
    """Matplotlib canvas styled strictly according to master's thesis standards (Image 2) with interactive hover effects (Image 3)."""

    def __init__(self, title: str = "", y_label: str = "", is_dark: bool = False, parent=None):
        self._theme_frames = {}
        self._theme_frame = None
        self._theme_redraw_pending = False
        self._rendered_frame = None
        self._resize_layout_pending = False
        self.is_dark = is_dark
        self.plot_title = title
        self.y_label_text = y_label
        self.series_data: Dict[str, Dict] = {}
        self.hover_dots: List = []
        self.hover_vline = None
        
        # Colors
        self._update_color_scheme(is_dark)

        self.fig = Figure(figsize=(4.8, 3.2), dpi=100, facecolor=self.bg_color)
        super().__init__(self.fig)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self._finish_resize)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        _MkThemeEngine.apply_style_sheet(self, f"background-color: {self.bg_color};")

        self.ax = self.fig.add_subplot(111)
        self._apply_thesis_style()
        self._update_layout_for_size()

        # Child tooltip overlay
        self.tooltip = ChartHoverTooltip(self)
        self.tooltip.hide()

        self.DEFAULT_LINEWIDTH = 1.3
        self.HOVER_LINEWIDTH = 2.4
        self._is_chart_hovered = False
        self.series_lines = []

        # Connect mouse events
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("axes_leave_event", self._on_mouse_leave)
        self.mpl_connect("figure_leave_event", self._on_mouse_leave)

    def wheelEvent(self, event):
        """Forward mouse wheel events to parent QScrollArea so dashboard can scroll naturally."""
        try:
            super().wheelEvent(event)
        except Exception:
            pass

        p = self.parent()
        scroll_area = None
        while p:
            if isinstance(p, QScrollArea):
                scroll_area = p
                break
            p = p.parent()

        if scroll_area is not None:
            QApplication.sendEvent(scroll_area.viewport(), event)
        else:
            event.ignore()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._set_curves_highlighted(True)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            self._set_curves_highlighted(False)
            self._on_mouse_leave(None)

    def _set_curves_highlighted(self, highlighted: bool):
        if self._is_chart_hovered == highlighted:
            return
        self._is_chart_hovered = highlighted
        target_lw = self.HOVER_LINEWIDTH if highlighted else self.DEFAULT_LINEWIDTH
        changed = False
        for _, line, _ in self.series_lines:
            try:
                if abs(line.get_linewidth() - target_lw) > 0.05:
                    line.set_linewidth(target_lw)
                    changed = True
            except Exception:
                pass
        if changed:
            self.draw_idle()

    def _update_color_scheme(self, is_dark: bool):
        self.is_dark = is_dark
        self.bg_color = "#171717" if is_dark else "#ffffff"
        self.text_color = "#f4f4f5" if is_dark else "#0f172a"
        self.muted_text = "#a1a1aa" if is_dark else "#64748b"
        self.spine_color = "#27272a" if is_dark else "#e2e8f0"
        self.grid_color = "#222225" if is_dark else "#f1f5f9"
        self.crosshair_color = "#3f3f46" if is_dark else "#cbd5e1"

    def _update_layout_for_size(self):
        """Dynamically adapt tick padding, label font sizes, and figure margins based on canvas height so x-axis & Epoch are never clipped."""
        if self._resize_layout_pending:
            return
        try:
            dpr = getattr(self, "device_pixel_ratio", 1.0)
            if hasattr(self, "devicePixelRatioF"):
                dpr = self.devicePixelRatioF()
            w = self.width() * dpr
            h = self.height() * dpr
            if w > 0 and h > 0 and self.fig.dpi > 0:
                self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi, forward=False)

            # Adapt based on height
            logical_h = self.height()
            if logical_h > 0 and logical_h < 200:
                self.ax.tick_params(axis="both", which="both", colors=self.muted_text, labelsize=8, width=0.7, length=2, pad=2)
                self.ax.set_xlabel("Epoch", fontsize=8.5, color=self.muted_text, fontweight="medium", labelpad=2)
                if self.y_label_text:
                    self.ax.set_ylabel(self.y_label_text, fontsize=8.5, color=self.muted_text, fontweight="medium", labelpad=2)
                self.fig.tight_layout(pad=0.5)
            elif logical_h > 0 and logical_h < 260:
                self.ax.tick_params(axis="both", which="both", colors=self.muted_text, labelsize=8.5, width=0.8, length=2.5, pad=3)
                self.ax.set_xlabel("Epoch", fontsize=9.0, color=self.muted_text, fontweight="medium", labelpad=3)
                if self.y_label_text:
                    self.ax.set_ylabel(self.y_label_text, fontsize=9.0, color=self.muted_text, fontweight="medium", labelpad=3)
                self.fig.tight_layout(pad=0.8)
            else:
                self.ax.tick_params(axis="both", which="both", colors=self.muted_text, labelsize=9, width=0.8, length=3, pad=4)
                self.ax.set_xlabel("Epoch", fontsize=9.5, color=self.muted_text, fontweight="medium", labelpad=4)
                if self.y_label_text:
                    self.ax.set_ylabel(self.y_label_text, fontsize=9.5, color=self.muted_text, fontweight="medium", labelpad=4)
                self.fig.tight_layout(pad=1.0)

            for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                label.set_fontfamily(["Segoe UI", "Microsoft YaHei", "Arial", "sans-serif"])

            self.draw_idle()
        except Exception:
            pass

    def resizeEvent(self, event):
        # Qt needs to finish resizing the window chrome before expensive Agg
        # rendering. Keep the last owned frame while consecutive sizes arrive.
        # The first frame still uses the normal immediate drawing path.
        if self._rendered_frame is not None:
            self._resize_layout_pending = True
            self._theme_frames.clear()
            self._theme_frame = None
            if self.isVisible() and not self.visibleRegion().isEmpty():
                self._resize_timer.start()
        super().resizeEvent(event)
        if not self._resize_layout_pending:
            self._update_layout_for_size()

    def _finish_resize(self):
        if not self._resize_layout_pending:
            return
        if not self.isVisible() or self.visibleRegion().isEmpty():
            # A chart below the scroll viewport can wait until it is exposed.
            return
        self._resize_layout_pending = False
        # A previously queued Matplotlib idle draw may have been deferred.
        self._draw_pending = False
        self._update_layout_for_size()
        self.draw_idle()

    def _draw_idle(self):
        if self._resize_layout_pending:
            return
        super()._draw_idle()

    def paintEvent(self, event):
        resizing = self._resize_layout_pending
        if not resizing:
            self._draw_idle()

        painter = QPainter(self)
        try:
            # Completely fill canvas background to prevent ghost trails during moving / resizing
            painter.fillRect(self.rect(), QColor(self.bg_color))
            if resizing:
                if self._rendered_frame is not None:
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    painter.drawImage(QRectF(self.rect()), self._rendered_frame)
                if not self._resize_timer.isActive():
                    self._resize_timer.start()
                return
            if not hasattr(self, 'renderer'):
                return
            qimage = self._theme_frame
            if qimage is None:
                buf = self.buffer_rgba()
                qimage = QImage(buf, buf.shape[1], buf.shape[0], QImage.Format.Format_RGBA8888)
                qimage.setDevicePixelRatio(self.device_pixel_ratio)
            painter.drawImage(QPoint(0, 0), qimage)
            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def _theme_frame_key(self):
        return self.is_dark, self.width(), self.height(), self.device_pixel_ratio

    def draw(self):
        self._theme_frame = None
        if not self._theme_redraw_pending:
            self._theme_frames.clear()
        self._theme_redraw_pending = False
        super().draw()
        # Own the pixels: Agg reuses its buffer on the next render. Keep at most
        # one light and one dark frame, invalidated by data/layout/hover updates.
        buf = self.buffer_rgba()
        frame = QImage(buf, buf.shape[1], buf.shape[0], QImage.Format.Format_RGBA8888).copy()
        frame.setDevicePixelRatio(self.device_pixel_ratio)
        self._rendered_frame = frame
        key = self._theme_frame_key()
        self._theme_frames = {k: v for k, v in self._theme_frames.items() if k[1:] == key[1:]}
        self._theme_frames[key] = frame

    def draw_idle(self):
        # All non-theme changes must discard both snapshots to avoid stale data.
        self._theme_frames.clear()
        self._theme_frame = None
        self._theme_redraw_pending = False
        super().draw_idle()

    def _draw_theme(self):
        self._theme_frame = self._theme_frames.get(self._theme_frame_key())
        if self._theme_frame is not None and not self._draw_pending:
            self._rendered_frame = self._theme_frame
            self.update()
        else:
            self._theme_redraw_pending = True
            super().draw_idle()

    def _apply_thesis_style(self):
        """Apply academic thesis despined formatting."""
        self.ax.set_facecolor(self.bg_color)
        self.fig.set_facecolor(self.bg_color)

        # Despine top and right borders (Image 2 characteristic)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color(self.spine_color)
        self.ax.spines["bottom"].set_color(self.spine_color)
        self.ax.spines["left"].set_linewidth(1.0)
        self.ax.spines["bottom"].set_linewidth(1.0)

        # Subtle horizontal grid lines only
        self.ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=self.grid_color, alpha=1.0)
        self.ax.grid(False, axis="x")

        # Tick styling
        self.ax.tick_params(axis="both", which="both", colors=self.muted_text, labelsize=9, width=0.8, length=3)
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_fontfamily(["Segoe UI", "Microsoft YaHei", "Arial", "sans-serif"])

        # Axis labels
        self.ax.set_xlabel("Epoch", fontsize=9.5, color=self.muted_text, fontweight="medium", labelpad=4)
        if self.y_label_text:
            self.ax.set_ylabel(self.y_label_text, fontsize=9.5, color=self.muted_text, fontweight="medium", labelpad=4)

    def _apply_thesis_colors(self):
        """Quickly update colors on theme switch without expensive tight_layout recalculation."""
        self.ax.set_facecolor(self.bg_color)
        self.fig.set_facecolor(self.bg_color)
        self.ax.spines["left"].set_color(self.spine_color)
        self.ax.spines["bottom"].set_color(self.spine_color)
        self.ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=self.grid_color, alpha=1.0)
        self.ax.tick_params(axis="both", which="both", colors=self.muted_text)
        self.ax.xaxis.label.set_color(self.muted_text)
        if self.y_label_text:
            self.ax.yaxis.label.set_color(self.muted_text)
        for text in self.ax.texts:
            text.set_color(self.muted_text)
        if self.hover_vline is not None:
            self.hover_vline.set_color(self.crosshair_color)
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_color(self.muted_text)
        leg = self.ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(self.bg_color)
            leg.get_frame().set_edgecolor(self.spine_color)
            for text in leg.get_texts():
                text.set_color(self.text_color)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_pending_theme", None) is not None:
            self._pending_theme = None
            self._apply_thesis_colors()
            self._draw_theme()

    def set_theme(self, is_dark: bool):
        if getattr(self, "_applied_dark_mode", None) == is_dark:
            return
        self._applied_dark_mode = is_dark
        self._update_color_scheme(is_dark)
        self.tooltip.set_theme(is_dark)
        if not self.isVisible():
            self._pending_theme = is_dark
            return
        self._pending_theme = None
        self._apply_thesis_colors()
        self._draw_theme()

    def render_series(self, series_dict: Dict[str, Dict]):
        """
        Render multiple model series on the same axis with halo dot highlight capability.
        """
        self._theme_frames.clear()
        self._theme_frame = None
        self.series_data = series_dict or {}
        self.ax.clear()
        self._apply_thesis_style()
        self.hover_dots.clear()
        self.hover_vline = None
        self.tooltip.hide()
        self.series_lines.clear()

        if not self.series_data or all(len(d.get("values", [])) == 0 for d in self.series_data.values()):
            self.ax.text(0.5, 0.5, "暂无训练数据或未勾选模型",
                         horizontalalignment="center", verticalalignment="center",
                         transform=self.ax.transAxes, color=self.muted_text, fontsize=10)
            self.draw_idle()
            return

        has_valid_data = False
        idx = 0
        min_y, max_y = float("inf"), float("-inf")

        for name, data in self.series_data.items():
            epochs = data.get("epochs", [])
            values = data.get("values", [])
            if not epochs or not values:
                continue

            color = data.get("color") or ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)]
            lw = self.HOVER_LINEWIDTH if self._is_chart_hovered else self.DEFAULT_LINEWIDTH
            linestyle = data.get("linestyle", "-")
            line, = self.ax.plot(epochs, values, label=name, color=color, linewidth=lw,
                                 linestyle=linestyle,
                                 solid_capstyle="round", solid_joinstyle="round", antialiased=True)
            self.series_lines.append((name, line, color))
            
            # Hover highlight halo dot for this curve
            dot, = self.ax.plot([epochs[0]], [values[0]], marker="o", markersize=7,
                                markerfacecolor="white", markeredgecolor=color,
                                markeredgewidth=2.2, visible=False, zorder=10)
            self.hover_dots.append((name, dot, epochs, values, color))

            min_y = min(min_y, min(values))
            max_y = max(max_y, max(values))
            has_valid_data = True
            idx += 1

        if has_valid_data:
            # Vertical crosshair guide line
            self.hover_vline = self.ax.axvline(x=0, color=self.crosshair_color, linestyle="-", linewidth=1.2, alpha=0.9, visible=False, zorder=5)

            # Clean legend only if more than 1 series
            if idx > 1:
                legend = self.ax.legend(loc="upper right", frameon=False, fontsize=8.5)
                if legend:
                    for text in legend.get_texts():
                        text.set_color(self.text_color)
            
            # Set Y limits with clean padding
            if min_y != float("inf") and max_y != float("-inf"):
                y_pad = max((max_y - min_y) * 0.1, 0.05)
                self.ax.set_ylim(bottom=max(0.0, min_y - y_pad * 0.5), top=min(1.0, max_y + y_pad))

        self.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or not self.series_data or not self.hover_dots:
            self._on_mouse_leave(event)
            return

        # Dynamically thicken curves when cursor is inside the plot
        if not self._is_chart_hovered:
            self._set_curves_highlighted(True)

        if event.xdata is None:
            self._on_mouse_leave(event)
            return

        target_epoch = int(round(event.xdata))

        # Check matched series data
        matched_items = []
        for name, dot, epochs, values, color in self.hover_dots:
            if not epochs or not values:
                dot.set_visible(False)
                continue
            
            # Find nearest epoch
            nearest_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - target_epoch))
            if abs(epochs[nearest_idx] - target_epoch) <= 4:
                ep = epochs[nearest_idx]
                val = values[nearest_idx]
                dot.set_data([ep], [val])
                dot.set_visible(True)
                matched_items.append((name, val, color, ep))
            else:
                dot.set_visible(False)

        if not matched_items:
            self._on_mouse_leave(event)
            return

        # Show and update vertical crosshair
        display_epoch = matched_items[0][3]
        if self.hover_vline:
            self.hover_vline.set_xdata([display_epoch, display_epoch])
            self.hover_vline.set_visible(True)

        self.draw_idle()

        # Build floating tooltip HTML (Image 2 & 3 format)
        html_lines = [
            f"<div style='color: {self.muted_text}; font-size: 11px; font-weight: 500;'>x: <b style='color: {self.text_color};'>{display_epoch}</b></div>"
        ]
        for name, val, color, _ in matched_items:
            html_lines.append(
                f"<div style='margin-top: 3px; font-size: 12px; white-space: nowrap;'>"
                f"<span style='color: {color}; font-size: 13px;'>●</span> "
                f"<span style='color: {self.muted_text}; font-weight: 600;'>{name}:</span> "
                f"<span style='color: {self.text_color}; font-weight: 700; margin-left: 3px;'>{val:.3f}</span>"
                f"</div>"
            )
        self.tooltip.lbl_content.setText("".join(html_lines))
        self.tooltip.adjustSize()

        # Calculate position inside canvas bounds using native Qt logical coordinates
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            g = event.guiEvent
            if hasattr(g, "position"):
                px = int(g.position().x())
                py = int(g.position().y())
            elif hasattr(g, "pos"):
                px = int(g.pos().x())
                py = int(g.pos().y())
            else:
                dpr = getattr(self, "device_pixel_ratio", 1.0)
                px = int(event.x / dpr)
                py = int(self.height() - event.y / dpr)
        else:
            dpr = getattr(self, "device_pixel_ratio", 1.0)
            px = int(event.x / dpr)
            py = int(self.height() - event.y / dpr)
        tip_w = self.tooltip.width()
        tip_h = self.tooltip.height()

        tx = px + 14
        ty = py - tip_h - 8
        if tx + tip_w > self.width() - 8:
            tx = px - tip_w - 14
        if tx < 8:
            tx = 8
        if ty < 8:
            ty = py + 14
        if ty + tip_h > self.height() - 8:
            ty = self.height() - tip_h - 8

        self.tooltip.move(tx, ty)
        self.tooltip.show()
        self.tooltip.raise_()

    def _on_mouse_leave(self, event):
        need_redraw = False
        pos = self.mapFromGlobal(QCursor.pos())
        # Restore curves to thin default when leaving canvas boundaries
        if not self.rect().contains(pos) and self._is_chart_hovered:
            self._is_chart_hovered = False
            for _, line, _ in self.series_lines:
                try:
                    if abs(line.get_linewidth() - self.DEFAULT_LINEWIDTH) > 0.05:
                        line.set_linewidth(self.DEFAULT_LINEWIDTH)
                        need_redraw = True
                except Exception:
                    pass

        if self.hover_vline and self.hover_vline.get_visible():
            self.hover_vline.set_visible(False)
            need_redraw = True

        for _, dot, _, _, _ in self.hover_dots:
            if dot.get_visible():
                dot.set_visible(False)
                need_redraw = True

        if self.tooltip.isVisible():
            self.tooltip.hide()

        if need_redraw:
            self.draw_idle()


class FullscreenChartDialog(QDialog):
    """High-resolution expanded chart modal with interactive controls and 300 DPI export."""

    def __init__(self, title: str, y_label: str, series_dict: Dict[str, Dict], is_dark: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"曲线全屏查看 - {title}")
        self.resize(1060, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        bg = "#171717" if is_dark else "#ffffff"
        fg = "#f4f4f5" if is_dark else "#09090b"
        border = "#27272a" if is_dark else "#e2e8f0"
        _MkThemeEngine.apply_style_sheet(self, f"background-color: {bg}; color: {fg};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header Bar
        hdr = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(PhChartLine.regular(size=20, color="#2563eb").pixmap(20, 20))
        hdr.addWidget(icon_lbl)

        t_lbl = QLabel(f"{title} - 训练评估指标对比曲线")
        t_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        hdr.addWidget(t_lbl)

        hdr.addStretch()

        btn_export = QToolButton()
        btn_export.setText(" 导出高清论文图 (300 DPI)")
        btn_export.setIcon(PhDownloadSimple.regular(size=15, color="#2563eb"))
        btn_export.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        _MkThemeEngine.apply_style_sheet(btn_export, """
            QToolButton {
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                background-color: #eff6ff;
                color: #2563eb;
            }
            QToolButton:hover {
                background-color: #dbeafe;
            }
        """)
        btn_export.clicked.connect(lambda: self._export_chart(title))
        hdr.addWidget(btn_export)

        btn_close = QToolButton()
        btn_close.setIcon(PhX.regular(size=16, color="#64748b"))
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        _MkThemeEngine.apply_style_sheet(btn_close, """
            QToolButton {
                border: none;
                border-radius: 6px;
                padding: 6px;
                background: transparent;
            }
            QToolButton:hover {
                background-color: #fee2e2;
            }
        """)
        btn_close.clicked.connect(self.close)
        hdr.addWidget(btn_close)

        layout.addLayout(hdr)

        # Canvas with full interactive hover
        self.canvas = ThesisChartCanvas(title=title, y_label=y_label, is_dark=is_dark, parent=self)
        self.canvas.render_series(series_dict)
        layout.addWidget(self.canvas, stretch=1)

    def _export_chart(self, title: str):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存高分辨率论文曲线图",
            f"{title}_thesis_plot.png",
            "PNG Image (*.png);;PDF Vector Document (*.pdf);;SVG Vector (*.svg)"
        )
        if path:
            self.canvas.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=self.canvas.bg_color)


class ThesisChartCard(MkQWidget):
    """
    Card-wrapped Thesis Chart Widget with header, metric title, and modern action buttons.
    Matching Image 1, 2, 3 cards with rounded corners and frontend aesthetics.
    """
    def __init__(self, metric_key: str = "", display_title: str = "", is_dark: bool = False, parent=None, title: str = None):
        super().__init__(parent=parent, role="card", radius=12)
        if title and not display_title:
            display_title = title
        if not metric_key:
            metric_key = display_title
        self.metric_key = metric_key
        self.display_title = display_title
        self.is_dark = is_dark
        self.series_data: Dict[str, Dict] = {}

        self._init_ui()

    def _init_ui(self):
        self.setObjectName("ThesisChartCard")
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 8)
        main_layout.setSpacing(4)

        # Header bar with Title & Modern Action Buttons (Image 3)
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 0, 2, 0)
        top_bar.setSpacing(6)

        self.lbl_title = QLabel(self.display_title)
        self.lbl_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        top_bar.addWidget(self.lbl_title)
        top_bar.addStretch()

        # 1. Fullscreen expand button
        self.btn_fullscreen = QToolButton()
        self.btn_fullscreen.setToolTip("全屏放大图表 (论文模式)")
        self.btn_fullscreen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fullscreen.setAutoRaise(True)
        self.btn_fullscreen.clicked.connect(self.open_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)

        # 2. More options menu button (···)
        self.btn_more = QToolButton()
        self.btn_more.setToolTip("更多图表选项")
        self.btn_more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_more.setAutoRaise(True)
        self.btn_more.clicked.connect(self._show_more_menu)
        top_bar.addWidget(self.btn_more)

        main_layout.addLayout(top_bar)

        # Matplotlib Canvas with Interactive Hover
        self.canvas = ThesisChartCanvas(title=self.display_title, is_dark=self.is_dark, parent=self)
        main_layout.addWidget(self.canvas, stretch=1)
        self._apply_card_style()
        self._update_toolbutton_styles()

    def wheelEvent(self, event):
        """Forward mouse wheel events to parent QScrollArea."""
        p = self.parent()
        scroll_area = None
        while p:
            if isinstance(p, QScrollArea):
                scroll_area = p
                break
            p = p.parent()

        if scroll_area is not None:
            QApplication.sendEvent(scroll_area.viewport(), event)
        else:
            event.ignore()

    def enterEvent(self, event):
        super().enterEvent(event)
        if hasattr(self, "canvas") and self.canvas:
            self.canvas._set_curves_highlighted(True)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if hasattr(self, "canvas") and self.canvas:
            pos = self.mapFromGlobal(QCursor.pos())
            if not self.rect().contains(pos):
                self.canvas._set_curves_highlighted(False)
                self.canvas._on_mouse_leave(None)

    def _apply_card_style(self):
        if hasattr(self, "lbl_title"):
            fg = "#f4f4f5" if self.is_dark else "#0f172a"
            _MkThemeEngine.apply_style_sheet(self.lbl_title, f"color: {fg}; background: transparent;")


    def _update_toolbutton_styles(self):
        btn_bg = "#27272a" if self.is_dark else "#f8fafc"
        btn_hover = "#3f3f46" if self.is_dark else "#e2e8f0"
        border = "#3f3f46" if self.is_dark else "#f1f5f9"
        btn_qss = f"""
            QToolButton {{
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px;
                background: {btn_bg};
            }}
            QToolButton:hover {{
                background: {btn_hover};
                border-color: #cbd5e1;
            }}
        """
        _MkThemeEngine.apply_style_sheet(self.btn_fullscreen, btn_qss)
        _MkThemeEngine.apply_style_sheet(self.btn_more, btn_qss)
        icon_color = "#94a3b8" if self.is_dark else "#64748b"
        self.btn_fullscreen.setIcon(PhArrowsOutSimple.regular(size=14, color=icon_color))
        self.btn_more.setIcon(PhDotsThree.bold(size=16, color=icon_color))

    def _show_more_menu(self):
        menu = QMenu(self)
        _MkThemeEngine.apply_style_sheet(menu, """
            QMenu {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 12px;
                color: #1e293b;
            }
            QMenu::item:selected {
                background-color: #eff6ff;
                color: #2563eb;
            }
        """)
        icon_color = "#94a3b8" if self.is_dark else "#64748b"
        act_export = menu.addAction(PhCamera.regular(size=14, color=icon_color), "导出高分辨率图片 (PNG / 300 DPI)")
        act_full = menu.addAction(PhCornersOut.regular(size=14, color=icon_color), "全屏放大查看")
        act_copy = menu.addAction(PhCopy.regular(size=14, color=icon_color), "复制最新评估数据")

        action = menu.exec(QCursor.pos())
        if action == act_export:
            self._export_image()
        elif action == act_full:
            self.open_fullscreen()
        elif action == act_copy:
            self._copy_metrics()

    def _export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存学术图表", f"{self.display_title}_chart.png", "PNG Images (*.png)")
        if path:
            self.canvas.fig.savefig(path, dpi=300, facecolor=self.canvas.bg_color, bbox_inches="tight")

    def _copy_metrics(self):
        lines = [f"{self.display_title} 最新数据:"]
        for name, data in self.series_data.items():
            vals = data.get("values", [])
            if vals:
                lines.append(f"  {name}: {vals[-1]:.4f} (Epoch {data.get('epochs', [len(vals)])[-1]})")
        QApplication.clipboard().setText("\n".join(lines))

    def set_theme(self, is_dark: bool):
        if getattr(self, "_applied_dark_mode", None) == is_dark:
            return
        self._applied_dark_mode = is_dark
        self.is_dark = is_dark
        self._apply_card_style()
        self._update_toolbutton_styles()
        self.canvas.set_theme(is_dark)

    def is_in_viewport(self) -> bool:
        """Check if this chart card intersects the visible viewport of its parent scroll area."""
        if not self.isVisible():
            return False
        p = self.parentWidget()
        scroll_area = None
        while p is not None:
            if isinstance(p, QScrollArea):
                scroll_area = p
                break
            p = p.parentWidget()
        if scroll_area is None:
            return True
        vp = scroll_area.viewport()
        top_left = self.mapTo(vp, QPoint(0, 0))
        card_rect = QRect(top_left, self.size())
        # 120px buffer above and below for smooth scroll anticipation
        vp_rect = vp.rect().adjusted(0, -120, 0, 120)
        return vp_rect.intersects(card_rect)

    def update_series(self, series_dict: Dict[str, Dict]):
        """Update multi-model curves data and redraw (lazy rendering if hidden or offscreen)."""
        self.series_data = series_dict or {}
        if not self.is_in_viewport():
            self._pending_render = True
            return
        self._pending_render = False
        self.canvas.render_series(self.series_data)

    def open_fullscreen(self):
        dlg = FullscreenChartDialog(
            title=self.display_title,
            y_label=self.display_title,
            series_dict=self.series_data,
            is_dark=self.is_dark,
            parent=self.window()
        )
        dlg.exec()

    def set_loading(self, loading: bool = True):
        """设置图表卡片流光骨架屏加载态。"""
        if getattr(self, "_loading", False) == loading:
            return
        self._loading = bool(loading)
        if hasattr(self, "lbl_title"):
            self.lbl_title.setVisible(not self._loading)
        if hasattr(self, "btn_fullscreen"):
            self.btn_fullscreen.setVisible(not self._loading)
        if hasattr(self, "btn_more"):
            self.btn_more.setVisible(not self._loading)
        if hasattr(self, "canvas"):
            self.canvas.setVisible(not self._loading)

        if self._loading:
            if not hasattr(self, "_shimmer_timer") or self._shimmer_timer is None:
                from PySide6.QtCore import QTimer
                self._shimmer_timer = QTimer(self)
                self._shimmer_timer.setInterval(30)
                self._shimmer_timer.timeout.connect(self._on_chart_shimmer_step)
            if not self._shimmer_timer.isActive() and self.isVisible():
                self._shimmer_timer.start()
        else:
            if hasattr(self, "_shimmer_timer") and self._shimmer_timer and self._shimmer_timer.isActive():
                self._shimmer_timer.stop()
        self.update()

    def _on_chart_shimmer_step(self):
        if not self.isVisible() or not getattr(self, "_loading", False):
            return
        self._shimmer_progress = (getattr(self, "_shimmer_progress", 0.0) + 0.03) % 1.0
        self.update()

    def hideEvent(self, event):
        if hasattr(self, "_shimmer_timer") and self._shimmer_timer and self._shimmer_timer.isActive():
            self._shimmer_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_pending_render", False) and self.is_in_viewport():
            self._pending_render = False
            self.canvas.render_series(self.series_data)
        if getattr(self, "_loading", False):
            if hasattr(self, "_shimmer_timer") and self._shimmer_timer and not self._shimmer_timer.isActive():
                self._shimmer_timer.start()

    def paintEvent(self, event):
        super().paintEvent(event)
        if getattr(self, "_pending_render", False) and self.is_in_viewport():
            self._pending_render = False
            self.canvas.render_series(self.series_data)
        if getattr(self, "_loading", False):
            try:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                is_dark = self.is_dark
                base_color = QColor(36, 36, 42) if is_dark else QColor(241, 245, 249)
                highlight_color = QColor(54, 54, 64, 200) if is_dark else QColor(255, 255, 255, 230)
                rect = self.rect()
                pad_x = 14.0
                content_w = max(10.0, float(rect.width() - pad_x * 2))
                content_h = max(10.0, float(rect.height() - 24.0))

                blocks = [
                    QRectF(pad_x, 14.0, min(120.0, content_w * 0.45), 14.0),
                    QRectF(pad_x, 38.0, content_w, content_h - 26.0)
                ]
                painter.setPen(Qt.PenStyle.NoPen)
                shimmer_prog = getattr(self, "_shimmer_progress", 0.0)
                sweep_x = float(rect.left()) + float(rect.width() + 200) * shimmer_prog - 100.0
                gradient_width = max(120.0, float(rect.width()) * 0.45)

                for block in blocks:
                    rad = 6.0 if block.height() > 20.0 else 4.0
                    grad = QLinearGradient(sweep_x - gradient_width, 0, sweep_x + gradient_width, 0)
                    grad.setColorAt(0.0, base_color)
                    grad.setColorAt(0.5, highlight_color)
                    grad.setColorAt(1.0, base_color)
                    painter.setBrush(QBrush(grad))
                    painter.drawRoundedRect(block, rad, rad)
                painter.end()
            except Exception:
                pass
