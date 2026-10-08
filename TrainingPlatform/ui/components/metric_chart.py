# -*- coding: utf-8 -*-
"""
Real-time Training Metrics Chart Component for AI Vision Training Platform
Supports pyqtgraph when installed, with a native QPainter fallback.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt

PYQTGRAPH_AVAILABLE = False
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class NativeChartWidget(QWidget):
    """Fallback custom canvas for loss and mAP rendering if pyqtgraph is missing."""
    def __init__(self, title="Chart", parent=None):
        super().__init__(parent)
        self.title = title
        self.data_series = {}  # label -> list of float values
        self.colors = [QColor(64, 158, 255), QColor(103, 194, 58), QColor(230, 162, 60), QColor(245, 108, 108)]
        self.setMinimumHeight(200)

    def add_series(self, label, values):
        self.data_series[label] = values
        self.update()

    def clear(self):
        self.data_series.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 35))

        # Title
        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.drawText(15, 25, self.title)

        if not self.data_series or all(len(v) == 0 for v in self.data_series.values()):
            painter.setPen(QColor(120, 120, 120))
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待训练数据输出...")
            return

        w, h = self.width(), self.height()
        margin = 40

        # Draw grid lines
        painter.setPen(QPen(QColor(60, 60, 65), 1, Qt.DashLine))
        for i in range(1, 5):
            y = margin + (h - 2 * margin) * i / 4
            painter.drawLine(margin, int(y), w - margin, int(y))

        # Draw curves
        color_idx = 0
        for label, values in self.data_series.items():
            if not values:
                continue
            color = self.colors[color_idx % len(self.colors)]
            painter.setPen(QPen(color, 2))

            max_val = max(max(values), 1.0)
            min_val = min(values)
            val_range = max(max_val - min_val, 0.0001)

            num_points = len(values)
            pts = []
            for idx, val in enumerate(values):
                x = margin + (w - 2 * margin) * (idx / max(num_points - 1, 1))
                y = h - margin - (h - 2 * margin) * ((val - min_val) / val_range)
                pts.append((int(x), int(y)))

            for i in range(len(pts) - 1):
                painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

            color_idx += 1


class MetricChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.epochs = []
        self.box_losses = []
        self.cls_losses = []
        self.map50s = []

        if PYQTGRAPH_AVAILABLE:
            pg.setConfigOption('background', '#18181c')
            pg.setConfigOption('foreground', '#e0e0e0')

            # Loss plot
            self.loss_plot = pg.PlotWidget(title="<b>Loss 损失曲线</b>")
            self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
            self.loss_plot.addLegend()
            self.curve_box = self.loss_plot.plot(pen=pg.mkPen('#409eff', width=2), name="Box Loss")
            self.curve_cls = self.loss_plot.plot(pen=pg.mkPen('#e6a23c', width=2), name="Cls Loss")

            # mAP plot
            self.map_plot = pg.PlotWidget(title="<b>mAP 评估指标</b>")
            self.map_plot.showGrid(x=True, y=True, alpha=0.3)
            self.map_plot.addLegend()
            self.curve_map50 = self.map_plot.plot(pen=pg.mkPen('#67c23a', width=2), name="mAP@50")

            layout.addWidget(self.loss_plot)
            layout.addWidget(self.map_plot)
        else:
            self.loss_chart = NativeChartWidget(title="Loss 损失曲线 (Box & Cls Loss)")
            self.map_chart = NativeChartWidget(title="mAP 评估指标 (mAP@50)")
            layout.addWidget(self.loss_chart)
            layout.addWidget(self.map_chart)

    def add_epoch_metric(self, epoch, box_loss, cls_loss, map50):
        self.epochs.append(epoch)
        self.box_losses.append(box_loss)
        self.cls_losses.append(cls_loss)
        self.map50s.append(map50)

        if PYQTGRAPH_AVAILABLE:
            self.curve_box.setData(self.epochs, self.box_losses)
            self.curve_cls.setData(self.epochs, self.cls_losses)
            self.curve_map50.setData(self.epochs, self.map50s)
        else:
            self.loss_chart.add_series("Box Loss", self.box_losses)
            self.loss_chart.add_series("Cls Loss", self.cls_losses)
            self.map_chart.add_series("mAP50", self.map50s)

    def clear(self):
        self.epochs.clear()
        self.box_losses.clear()
        self.cls_losses.clear()
        self.map50s.clear()

        if PYQTGRAPH_AVAILABLE:
            self.curve_box.setData([], [])
            self.curve_cls.setData([], [])
            self.curve_map50.setData([], [])
        else:
            self.loss_chart.clear()
            self.map_chart.clear()

    def load_history(self, history_rows):
        """Batch load metrics from DB metrics_history table."""
        self.clear()
        for r in history_rows:
            self.add_epoch_metric(
                epoch=r.get("epoch", 0),
                box_loss=r.get("train_box_loss", 0.0),
                cls_loss=r.get("train_cls_loss", 0.0),
                map50=r.get("map50", 0.0)
            )
