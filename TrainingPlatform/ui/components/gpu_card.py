# -*- coding: utf-8 -*-
"""
GPU Status Card Component for AI Vision Training Platform
Using MonkeyQt MkCard, MkProgressBar
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer

from monkeyqt import MkCard, MkProgressBar, MkAlert, ThemeEngine, MkQWidget
from TrainingPlatform.core.gpu_monitor import GPUMonitor

class GPUCard(MkQWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, role="transparent")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.card = MkCard(title="GPU 算力与 CUDA 驱动状态")
        card_layout = self.card.content_layout

        self.lbl_gpu_name = QLabel("GPU 状态: 正在检测中...")
        _MkThemeEngine.apply_style_sheet(self.lbl_gpu_name, "font-size: 14px; font-weight: bold; color: #303133;")

        self.lbl_vram_info = QLabel("显存使用: 0.0 GB / 0.0 GB")
        _MkThemeEngine.apply_style_sheet(self.lbl_vram_info, "font-size: 12px; color: #606266;")

        self.progress_vram = MkProgressBar()
        self.progress_vram.percentage = 0

        self.lbl_cuda_info = QLabel("CUDA 状态: 检测中...")
        _MkThemeEngine.apply_style_sheet(self.lbl_cuda_info, "font-size: 12px; color: #909399;")

        card_layout.addWidget(self.lbl_gpu_name)
        card_layout.addWidget(self.lbl_vram_info)
        card_layout.addWidget(self.progress_vram)
        card_layout.addWidget(self.lbl_cuda_info)

        layout.addWidget(self.card)

        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)

        # Refresh GPU info periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_gpu_status)
        self.timer.start(3000)
        self.refresh_gpu_status()

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self.lbl_vram_info, "font-size: 12px; color: #cbd5e1;")
            _MkThemeEngine.apply_style_sheet(self.lbl_cuda_info, "font-size: 12px; color: #94a3b8;")
        else:
            _MkThemeEngine.apply_style_sheet(self.lbl_vram_info, "font-size: 12px; color: #475569;")
            _MkThemeEngine.apply_style_sheet(self.lbl_cuda_info, "font-size: 12px; color: #64748b;")

    def refresh_gpu_status(self):
        info = GPUMonitor.get_gpu_info()
        is_dark = ThemeEngine.is_dark()
        if not info["cuda_available"] or not info["devices"]:
            warn_color = "#fbbf24" if is_dark else "#d97706"
            self.lbl_gpu_name.setText("未检测到可用 NVIDIA GPU (CUDA 未开启)")
            _MkThemeEngine.apply_style_sheet(self.lbl_gpu_name, f"font-size: 14px; font-weight: bold; color: {warn_color};")
            self.lbl_vram_info.setText("建议使用 CPU 模式进行小规模测试")
            self.progress_vram.percentage = 0
            self.lbl_cuda_info.setText("CUDA 可用性: False")
        else:
            gpu = info["devices"][0]
            name = gpu["name"]
            used = gpu["used_memory_gb"]
            total = gpu["total_memory_gb"]
            percent = int((used / total) * 100) if total > 0 else 0

            success_color = "#4ade80" if is_dark else "#16a34a"
            self.lbl_gpu_name.setText(f"GPU: {name}")
            _MkThemeEngine.apply_style_sheet(self.lbl_gpu_name, f"font-size: 14px; font-weight: bold; color: {success_color};")
            self.lbl_vram_info.setText(f"显存: {used:.2f} GB / {total:.2f} GB ({percent}%)")
            self.progress_vram.percentage = percent
            self.lbl_cuda_info.setText(f"CUDA 版本: {info['cuda_version']} | 设备数: {info['device_count']}")
