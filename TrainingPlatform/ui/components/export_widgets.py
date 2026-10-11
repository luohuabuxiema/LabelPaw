# -*- coding: utf-8 -*-
"""
Export Widgets & Dialogs (模型导出组件库)
Benchmarked strictly against Ultralytics HUB Export Management (Images 1-5):
- 8 Core Export Formats: ONNX, TensorRT, OpenVINO, CoreML, RKNN, Hailo, NCNN, LiteRT
- Schema-driven Export Dialog with encapsulated atomic parameter fields
- Expandable Format Cards with export history records, downloads, folder browsing, and deletion
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QLineEdit, QComboBox, QDialog, QPushButton, QFileDialog,
    QSizePolicy, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QUrl
from PySide6.QtGui import QFont, QColor, QCursor, QDesktopServices, QPainter, QPen

from monkeyqt import MkButton, MkMessage, MkSwitch, ThemeEngine
from monkeyqt_icons import (
    PhCube, PhCpu, PhAppleLogo, PhLightning, PhDeviceMobile,
    PhDownloadSimple, PhTrash, PhFolder, PhCheckCircle, PhCaretDown,
    PhCaretRight, PhArrowSquareOut, PhX, PhSparkle, PhInfo, PhFloppyDisk
)
from TrainingPlatform.ui.components.custom_tooltip import (
    InfoIconButton, CustomTooltipPopover as InfoTooltipPopover, attach_tooltip
)

# ─────────────────────────────────────────────────────────────────────────────
# ── 1. 导出格式元数据规范配置 (Schema Specification)
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_SPECS: Dict[str, Dict[str, Any]] = {
    "onnx": {
        "name": "ONNX",
        "tag": "onnx",
        "badge": "跨平台通用",
        "badge_color": "#2563eb",
        "badge_bg": "#eff6ff",
        "desc": "工业界开放通用模型格式，支持云端服务器、OpenCV DNN、ONNX Runtime 与嵌入式部署",
        "doc_url": "https://docs.ultralytics.com/integrations/onnx/",
        "icon_color": "#0284c7",
        "fields": ["imgsz", "batch", "precision", "dynamic", "simplify", "opset", "post_processing"],
        "default_ext": ".onnx"
    },
    "engine": {
        "name": "TensorRT",
        "tag": "engine",
        "badge": "GPU 极致加速",
        "badge_color": "#16a34a",
        "badge_bg": "#f0fdf4",
        "desc": "NVIDIA GPU 显卡极致加速引擎，支持层融合、FP16/INT8 量化加速与毫秒级低延迟",
        "doc_url": "https://docs.ultralytics.com/integrations/tensorrt/",
        "icon_color": "#15803d",
        "fields": ["target_gpu", "imgsz", "batch", "precision", "dynamic", "simplify", "workspace", "post_processing"],
        "default_ext": ".engine"
    },
    "openvino": {
        "name": "OpenVINO",
        "tag": "openvino",
        "badge": "Intel 优化",
        "badge_color": "#0284c7",
        "badge_bg": "#f0f9ff",
        "desc": "英特尔官方深度优化引擎，针对 Intel CPU、核芯显卡 (iGPU) 及独立显卡深度加速",
        "doc_url": "https://docs.ultralytics.com/integrations/openvino/",
        "icon_color": "#0369a1",
        "fields": ["imgsz", "batch", "precision", "dynamic", "post_processing"],
        "default_ext": "_openvino_model"
    },
    "coreml": {
        "name": "CoreML",
        "tag": "coreml",
        "badge": "Apple 生态",
        "badge_color": "#4b5563",
        "badge_bg": "#f3f4f6",
        "desc": "苹果全生态专用（iOS、macOS、iPadOS），深度调用 Apple Neural Engine (ANE) 神经引擎",
        "doc_url": "https://docs.ultralytics.com/integrations/coreml/",
        "icon_color": "#1f2937",
        "fields": ["imgsz", "batch", "precision", "dynamic", "post_processing"],
        "default_ext": ".mlpackage"
    },
    "rknn": {
        "name": "RKNN (瑞芯微)",
        "tag": "rknn",
        "badge": "国产 NPU 板卡",
        "badge_color": "#ea580c",
        "badge_bg": "#fff7ed",
        "desc": "瑞芯微国产嵌入式主板专属（RK3588、RK3568、RV1126/1109 等），跑满硬件片上 NPU 算力",
        "doc_url": "https://docs.ultralytics.com/integrations/rknn/",
        "icon_color": "#c2410c",
        "fields": ["rknn_target", "imgsz", "batch", "post_processing"],
        "default_ext": "_rknn_model"
    },
    "hailo": {
        "name": "Hailo",
        "tag": "hailo",
        "badge": "树莓派 5 AI Kit",
        "badge_color": "#7c3aed",
        "badge_bg": "#f5f3ff",
        "desc": "Hailo-8 / Hailo-8L 边缘 AI 加速模块，树莓派 5 官方 AI 套件（Raspberry Pi AI Kit）首选",
        "doc_url": "https://docs.ultralytics.com/integrations/hailo/",
        "icon_color": "#6d28d9",
        "fields": ["hailo_arch", "imgsz", "batch", "precision", "post_processing"],
        "default_ext": "_hailo_model.onnx"
    },
    "ncnn": {
        "name": "NCNN",
        "tag": "ncnn",
        "badge": "纯 CPU / ARM",
        "badge_color": "#0891b2",
        "badge_bg": "#ecfeff",
        "desc": "腾讯开源轻量级移动端与纯 CPU 推理框架，针对 ARM NEON 汇编深度优化，无第三方依赖",
        "doc_url": "https://docs.ultralytics.com/integrations/ncnn/",
        "icon_color": "#0e7490",
        "fields": ["imgsz", "batch", "precision"],
        "default_ext": "_ncnn_model"
    },
    "litert": {
        "name": "LiteRT (TFLite)",
        "tag": "litert",
        "badge": "移动端 / Android",
        "badge_color": "#d97706",
        "badge_bg": "#fffbeb",
        "desc": "谷歌移动端及嵌入式轻量引擎（原 TensorFlow Lite / TFLite），适用于 Android 及树莓派",
        "doc_url": "https://docs.ultralytics.com/integrations/tflite/",
        "icon_color": "#b45309",
        "fields": ["imgsz", "batch", "precision", "post_processing"],
        "default_ext": ".tflite"
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ── 2. 参数原子控件 (Atomic Form Fields)
# ─────────────────────────────────────────────────────────────────────────────

def _make_field_header(title: str, tip: str = "") -> QWidget:
    """Creates a clean Chinese field title with integrated InfoIconButton popup tooltip."""
    w = QWidget()
    _MkThemeEngine.apply_style_sheet(w, "background: transparent; border: none;")
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 2)
    h.setSpacing(6)

    lbl = QLabel(title)
    lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
    is_dark = ThemeEngine.is_dark()
    title_color = "#f1f5f9" if is_dark else "#111827"
    _MkThemeEngine.apply_style_sheet(lbl, f"color: {title_color}; background: transparent; border: none;")
    h.addWidget(lbl)

    if tip:
        ic = InfoIconButton(tip)
        h.addWidget(ic)

    h.addStretch()
    return w


def _make_subtitle(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", 9))
    is_dark = ThemeEngine.is_dark()
    sub_color = "#9ca3af" if is_dark else "#6b7280"
    _MkThemeEngine.apply_style_sheet(lbl, f"color: {sub_color}; margin-bottom: 4px; background: transparent; border: none;")
    return lbl


class BaseFormField(QWidget):
    """Base class for encapsulated form fields."""
    def __init__(self, parent=None):
        super().__init__(parent)
        _MkThemeEngine.apply_style_sheet(self, "background: transparent; border: none;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 10)
        self.layout.setSpacing(3)

    def get_value(self) -> Any:
        raise NotImplementedError


class ImageSizeField(BaseFormField):
    """Image Size: e.g. 640"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("图像尺寸", "输入模型推理的图像像素分辨率 (如 640x640)，有效范围 32 - 8192"))
        self.layout.addWidget(_make_subtitle("输入模型的正方形分辨率尺寸 (32-8192)"))
        self.edit = QLineEdit("640")
        _MkThemeEngine.apply_style_sheet(self.edit, """
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: #111827;
            }
            QLineEdit:focus { border: 1.5px solid #2563eb; }
        """)
        self.layout.addWidget(self.edit)

    def get_value(self) -> int:
        try:
            val = int(self.edit.text().strip())
            return max(32, min(8192, val))
        except Exception:
            return 640


class BatchSizeField(BaseFormField):
    """Batch Size: e.g. 1"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("批次大小", "单次前向推理同时处理的图像数量，范围 1 - 64，通常部署设为 1"))
        self.layout.addWidget(_make_subtitle("单次推理批次大小 (1-32)"))
        self.edit = QLineEdit("1")
        _MkThemeEngine.apply_style_sheet(self.edit, """
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: #111827;
            }
            QLineEdit:focus { border: 1.5px solid #2563eb; }
        """)
        self.layout.addWidget(self.edit)

    def get_value(self) -> int:
        try:
            val = int(self.edit.text().strip())
            return max(1, min(64, val))
        except Exception:
            return 1


class PrecisionField(BaseFormField):
    """Precision: FP16 / FP32 / INT8"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("精度与量化", "模型权重与计算激活值的精度模式。FP16 兼顾速度与精度，INT8 提供极致轻量化"))
        self.layout.addWidget(_make_subtitle("导出模型的数值量化精度"))
        self.combo = QComboBox()
        self.combo.addItems(["FP16 半精度加速 (推荐)", "FP32 全精度", "INT8 8位整型极致量化"])
        _MkThemeEngine.apply_style_sheet(self.combo, """
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                color: #111827;
            }
            QComboBox::drop-down { border: none; width: 24px; }
        """)
        self.layout.addWidget(self.combo)

    def get_value(self) -> Dict[str, bool]:
        txt = self.combo.currentText()
        if "FP16" in txt:
            return {"half": True, "int8": False}
        elif "INT8" in txt:
            return {"half": False, "int8": True}
        else:
            return {"half": False, "int8": False}


class DynamicSwitchField(BaseFormField):
    """Dynamic input shapes toggle switch."""
    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        v_lbl = QVBoxLayout()
        v_lbl.setSpacing(2)
        v_lbl.addWidget(_make_field_header("动态输入尺寸", "允许模型在推理时接受动态可变的 batch 大小或图像宽高尺寸"))
        v_lbl.addWidget(_make_subtitle("支持动态可变输入分辨率推理"))
        row.addLayout(v_lbl)
        row.addStretch()

        self.switch = MkSwitch(checked=False)
        row.addWidget(self.switch, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.layout.addLayout(row)

    def get_value(self) -> bool:
        return self.switch._checked


class SimplifySwitchField(BaseFormField):
    """Simplify ONNX/TensorRT graph toggle switch."""
    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        v_lbl = QVBoxLayout()
        v_lbl.setSpacing(2)
        v_lbl.addWidget(_make_field_header("算子精简优化", "使用 onnxslim 等工具精简冗余计算节点与折叠常量"))
        v_lbl.addWidget(_make_subtitle("自动精简计算图以提高推理速度"))
        row.addLayout(v_lbl)
        row.addStretch()

        self.switch = MkSwitch(checked=True)
        row.addWidget(self.switch, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.layout.addLayout(row)

    def get_value(self) -> bool:
        return self.switch._checked


class OpsetField(BaseFormField):
    """ONNX Opset version."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("算子集版本", "ONNX 算子集规范版本 (9-23)，留空或 auto 为自动匹配最佳推荐版本"))
        self.layout.addWidget(_make_subtitle("ONNX 算子集版本 (留空或 auto 为自动推荐)"))
        self.edit = QLineEdit("auto")
        _MkThemeEngine.apply_style_sheet(self.edit, """
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: #111827;
            }
            QLineEdit:focus { border: 1.5px solid #2563eb; }
        """)
        self.layout.addWidget(self.edit)

    def get_value(self) -> Optional[int]:
        txt = self.edit.text().strip().lower()
        if txt in ("", "auto", "none"):
            return None
        try:
            return int(txt)
        except Exception:
            return None


class PostProcessingField(BaseFormField):
    """Post-processing dropdown benchmarked against Fig 3."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("后处理模式", "非极大值抑制 (NMS) 用于过滤重叠的多余检测框"))
        self.layout.addWidget(_make_subtitle("决定是否在模型计算图中内置非极大值抑制 (NMS)"))
        self.combo = QComboBox()
        self.combo.addItems([
            "原始未过滤预测输出 (默认)",
            "集成非极大值抑制 (NMS 过滤)",
            "End-to-End 无 NMS 架构"
        ])
        _MkThemeEngine.apply_style_sheet(self.combo, """
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                color: #111827;
            }
            QComboBox::drop-down { border: none; width: 24px; }
        """)
        self.layout.addWidget(self.combo)

    def get_value(self) -> Dict[str, Any]:
        txt = self.combo.currentText()
        if "NMS 过滤" in txt:
            return {"nms": True, "end2end": False}
        elif "End-to-End" in txt:
            return {"nms": False, "end2end": True}
        else:
            return {"nms": False, "end2end": False}


class TargetGpuField(BaseFormField):
    """Target GPU selection for TensorRT benchmarked against Fig 4."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("目标 GPU", "TensorRT 引擎需针对特定 GPU 微架构编译优化"))
        self.layout.addWidget(_make_subtitle("选择部署的目标显卡硬件架构"))
        self.combo = QComboBox()

        # Detect local GPUs
        gpu_detected = False
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = round(props.total_memory / (1024 ** 3))
                    self.combo.addItem(f"{name} ({vram_gb} GB) [本机可用]", userData=str(i))
                    gpu_detected = True
        except Exception:
            pass

        if not gpu_detected:
            self.combo.addItem("CUDA:0 (默认首选显卡)", userData="0")
            self.combo.addItem("RTX 4090 (24 GB) [交叉编译目标]", userData="0")

        _MkThemeEngine.apply_style_sheet(self.combo, """
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                color: #111827;
            }
            QComboBox::drop-down { border: none; width: 24px; }
        """)
        self.layout.addWidget(self.combo)

    def get_value(self) -> str:
        data = self.combo.currentData()
        return str(data) if data is not None else "0"


class WorkspaceField(BaseFormField):
    """TensorRT Workspace size in GB benchmarked against Fig 4."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("显存工作区大小", "TensorRT 构建优化网络时允许使用的最大临时显存 (1-32 GB)"))
        self.layout.addWidget(_make_subtitle("TensorRT 构建允许使用的显存大小 (1-32 GB，auto 为自动调整)"))
        self.edit = QLineEdit("auto")
        _MkThemeEngine.apply_style_sheet(self.edit, """
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: #111827;
            }
            QLineEdit:focus { border: 1.5px solid #2563eb; }
        """)
        self.layout.addWidget(self.edit)

    def get_value(self) -> Optional[int]:
        txt = self.edit.text().strip().lower()
        if txt in ("", "auto", "none"):
            return None
        try:
            return max(1, min(32, int(txt)))
        except Exception:
            return None


class RknnTargetField(BaseFormField):
    """Rockchip RKNN target SoC platform."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("目标芯片平台", "瑞芯微国产主板的核心主控芯片架构"))
        self.layout.addWidget(_make_subtitle("选择目标瑞芯微嵌入式板卡芯片型号"))
        self.combo = QComboBox()
        self.combo.addItems([
            "rk3588 (旗舰 6 TOPS NPU - 推荐)",
            "rk3568 (主流工控 1 TOPS)",
            "rk3566 (轻量工控 1 TOPS)",
            "rv1126 (低功耗视觉 2 TOPS)",
            "rv1109 (轻量智能相机 1.2 TOPS)",
            "rk3399pro (3 TOPS)"
        ])
        _MkThemeEngine.apply_style_sheet(self.combo, """
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                color: #111827;
            }
            QComboBox::drop-down { border: none; width: 24px; }
        """)
        self.layout.addWidget(self.combo)

    def get_value(self) -> str:
        txt = self.combo.currentText().split()[0].lower()
        return txt


class HailoArchField(BaseFormField):
    """Hailo target architecture."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout.addWidget(_make_field_header("芯片架构", "目标 Hailo 边缘 AI 加速硬件架构型号"))
        self.layout.addWidget(_make_subtitle("选择 Hailo 边缘算力芯片型号"))
        self.combo = QComboBox()
        self.combo.addItems([
            "Hailo-8L (树莓派 5 官方 AI Kit 13 TOPS - 推荐)",
            "Hailo-8 (高性能标准版 26 TOPS)"
        ])
        _MkThemeEngine.apply_style_sheet(self.combo, """
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                color: #111827;
            }
            QComboBox::drop-down { border: none; width: 24px; }
        """)
        self.layout.addWidget(self.combo)

    def get_value(self) -> str:
        txt = self.combo.currentText()
        return "hailo8l" if "8L" in txt else "hailo8"


# ─────────────────────────────────────────────────────────────────────────────
# ── 3. 配置弹窗 (BaseExportDialog) 严格对标图 2 / 图 4 / 图 5
# ─────────────────────────────────────────────────────────────────────────────

class BaseExportDialog(QDialog):
    """
    Export Settings Modal Dialog benchmarked against Ultralytics HUB (Fig 2, 4, 5).
    """
    export_submitted = Signal(str, dict)  # format_key, params_dict

    def __init__(self, format_key: str, parent=None):
        super().__init__(parent)
        self.format_key = format_key
        self.spec = EXPORT_SPECS.get(format_key, {})
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(480, 640)

        self._fields: Dict[str, BaseFormField] = {}
        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)

        # Card container with shadow
        card = QFrame()
        card.setObjectName("ExportModalCard")
        _MkThemeEngine.apply_style_sheet(card, """
            QFrame#ExportModalCard {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 35))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(0)

        # ── 1. Top Header Row ──────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 4)
        header_row.setSpacing(10)

        title_text = f"导出至 {self.spec.get('name', self.format_key.upper())}"
        lbl_title = QLabel(title_text)
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        _MkThemeEngine.apply_style_sheet(lbl_title, "color: #111827; background: transparent;")
        header_row.addWidget(lbl_title)
        header_row.addStretch()

        btn_close = QPushButton()
        btn_close.setIcon(PhX.regular(size=14, color="#9ca3af"))
        btn_close.setFixedSize(24, 24)
        _MkThemeEngine.apply_style_sheet(btn_close, """
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.06);
            }
        """)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.reject)
        header_row.addWidget(btn_close)
        card_layout.addLayout(header_row)

        # Subtitle with doc link
        sub_row = QHBoxLayout()
        sub_row.setContentsMargins(0, 0, 0, 14)
        lbl_sub = QLabel(f"配置 {self.spec.get('name', '')} 格式的导出参数。")
        lbl_sub.setFont(QFont("Segoe UI", 10))
        _MkThemeEngine.apply_style_sheet(lbl_sub, "color: #6b7280; background: transparent;")
        sub_row.addWidget(lbl_sub)

        link_url = self.spec.get("doc_url", "")
        if link_url:
            lbl_link = QLabel('<a href="' + link_url + '" style="color: #2563eb; text-decoration: none;">了解更多 ↗</a>')
            lbl_link.setFont(QFont("Segoe UI", 10))
            lbl_link.setOpenExternalLinks(True)
            _MkThemeEngine.apply_style_sheet(lbl_link, "background: transparent;")
            sub_row.addWidget(lbl_link)
        sub_row.addStretch()
        card_layout.addLayout(sub_row)

        # ── 2. Scrollable Form Area ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        _MkThemeEngine.apply_style_sheet(scroll, """
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #d1d5db;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #9ca3af; }
        """)

        form_widget = QWidget()
        _MkThemeEngine.apply_style_sheet(form_widget, "background: transparent;")
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 8, 0)
        form_layout.setSpacing(6)

        fields_to_add = self.spec.get("fields", ["imgsz", "batch", "precision"])

        # Dynamically instantiate required fields based on schema
        field_factory = {
            "target_gpu": TargetGpuField,
            "rknn_target": RknnTargetField,
            "hailo_arch": HailoArchField,
            "imgsz": ImageSizeField,
            "batch": BatchSizeField,
            "precision": PrecisionField,
            "dynamic": DynamicSwitchField,
            "simplify": SimplifySwitchField,
            "workspace": WorkspaceField,
            "opset": OpsetField,
            "post_processing": PostProcessingField,
        }

        for f_key in fields_to_add:
            cls = field_factory.get(f_key)
            if cls:
                field_obj = cls(parent=form_widget)
                self._fields[f_key] = field_obj
                form_layout.addWidget(field_obj)

        form_layout.addStretch()
        scroll.setWidget(form_widget)
        card_layout.addWidget(scroll, stretch=1)

        # Status text (for validation/error messages)
        self.lbl_status = QLabel("")
        self.lbl_status.setFont(QFont("Segoe UI", 9))
        _MkThemeEngine.apply_style_sheet(self.lbl_status, "color: #dc2626; margin-top: 4px; background: transparent;")
        self.lbl_status.hide()
        card_layout.addWidget(self.lbl_status)

        # ── 3. Bottom Action Bar ───────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 16, 0, 0)
        btn_bar.setSpacing(12)
        btn_bar.addStretch()

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        _MkThemeEngine.apply_style_sheet(self.btn_cancel, """
            QPushButton {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 7px 20px;
                color: #374151;
            }
            QPushButton:hover { background: #f9fafb; border-color: #9ca3af; }
        """)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_start = QPushButton("开始导出")
        self.btn_start.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        _MkThemeEngine.apply_style_sheet(self.btn_start, """
            QPushButton {
                background: #111827;
                border: none;
                border-radius: 8px;
                padding: 7px 22px;
                color: #ffffff;
            }
            QPushButton:hover { background: #1f2937; }
            QPushButton:disabled { background: #9ca3af; color: #f3f4f6; }
        """)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_bar.addWidget(self.btn_start)

        card_layout.addLayout(btn_bar)
        root_layout.addWidget(card)

    def closeEvent(self, event):
        InfoTooltipPopover.hide_popover()
        super().closeEvent(event)

    def reject(self):
        InfoTooltipPopover.hide_popover()
        super().reject()

    def _on_start_clicked(self):
        InfoTooltipPopover.hide_popover()
        params = {"format": self.format_key}
        for k, field in self._fields.items():
            val = field.get_value()
            if isinstance(val, dict):
                params.update(val)
            else:
                params[k] = val

        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在准备导出...")
        self.export_submitted.emit(self.format_key, params)
        self.accept()


class CircularSpinnerWidget(QWidget):
    """
    Smooth, modern rotating arc spinner matching Ultralytics HUB export indicator.
    """
    def __init__(self, color="#2563eb", size=18, line_width=2.2, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self._line_width = line_width
        self._angle = 0
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._timer = QTimer(self)
        self._timer.setInterval(25)  # ~40 fps
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def _on_tick(self):
        self._angle = (self._angle + 10) % 360
        self.update()

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pad = int(self._line_width / 2) + 1
        rect = self.rect().adjusted(pad, pad, -pad, -pad)

        # Faint track
        track_col = QColor(self._color.red(), self._color.green(), self._color.blue(), 25)
        track_pen = QPen(track_col, self._line_width)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawEllipse(rect)

        # Active rotating arc (270 degrees)
        arc_pen = QPen(self._color, self._line_width)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        start_angle = int(-self._angle * 16)
        span_angle = int(270 * 16)
        painter.drawArc(rect, start_angle, span_angle)


def format_export_summary(params: dict) -> str:
    """Format params dict into clean summary string e.g. imgsz=640, quantize=16, dynamic=false..."""
    parts = []
    if "imgsz" in params:
        parts.append(f"imgsz={params['imgsz']}")
    if "precision" in params:
        p_val = str(params["precision"])
        if "INT8" in p_val:
            parts.append("quantize=8")
        elif "FP16" in p_val:
            parts.append("quantize=16")
        else:
            parts.append("quantize=32")
    elif params.get("int8"):
        parts.append("quantize=8")
    elif params.get("half"):
        parts.append("quantize=16")

    if "dynamic" in params:
        parts.append(f"dynamic={str(params['dynamic']).lower()}")
    if "simplify" in params:
        parts.append(f"simplify={str(params['simplify']).lower()}")
    if "batch" in params:
        parts.append(f"batch={params['batch']}")
    if "opset" in params and params["opset"]:
        parts.append(f"opset={params['opset']}")
    if "workspace" in params and params["workspace"]:
        parts.append(f"workspace={params['workspace']}")
    if "target_gpu" in params and params["target_gpu"]:
        parts.append(f"gpu={params['target_gpu']}")
    if "rknn_target" in params and params["rknn_target"]:
        parts.append(f"target={params['rknn_target']}")
    if "hailo_arch" in params and params["hailo_arch"]:
        parts.append(f"arch={params['hailo_arch']}")
    if "file" in params:
        parts.append(f"file={params['file']}")

    if not parts:
        return "imgsz=640, batch=1, precision=FP16"
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ── 4. 历史记录单条目 (ExportHistoryItem) 严格对标图 1（无外框、支持动态加载）
# ─────────────────────────────────────────────────────────────────────────────

class ExportHistoryItem(QFrame):
    """
    Single exported model row strictly matching Fig 1 & Fig 2:
    - Absolutely NO redundant card borders / pill frames.
    - Pending (Exporting) state: dynamic rotating blue circular spinner, param summary, "· 刚刚", cancel red trash button.
    - Finished state: green circle checkmark, size / duration / timestamp, open folder, download, delete buttons.
    """
    download_clicked = Signal(dict)
    delete_clicked = Signal(dict)
    open_folder_clicked = Signal(dict)
    cancel_clicked = Signal(dict)

    def __init__(self, record: dict, is_pending: bool = False, parent=None):
        super().__init__(parent)
        self.record = record
        self.is_pending = is_pending
        self.is_dark = ThemeEngine.is_dark()
        self.setObjectName("ExportHistoryItem")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)

        # 1. Left Icon: dynamic spinner when pending, green checkmark when finished
        self.spinner = CircularSpinnerWidget(color="#2563eb", size=18, line_width=2.2, parent=self)
        layout.addWidget(self.spinner)

        self.lbl_check = QLabel()
        self.lbl_check.setPixmap(PhCheckCircle.fill(color="#16a34a", size=18).pixmap(18, 18))
        _MkThemeEngine.apply_style_sheet(self.lbl_check, "background: transparent; border: none;")
        self.lbl_check.setFixedSize(18, 18)
        layout.addWidget(self.lbl_check)

        # 2. Middle: Config summary (top) & meta (bottom)
        v_info = QVBoxLayout()
        v_info.setContentsMargins(0, 0, 0, 0)
        v_info.setSpacing(2)

        summary_text = record.get("summary")
        if not summary_text:
            summary_text = format_export_summary(record.get("params", {}))
        self.lbl_summary = QLabel(summary_text)
        self.lbl_summary.setFont(QFont("Consolas", 10, QFont.Weight.DemiBold))
        v_info.addWidget(self.lbl_summary)

        self.lbl_meta = QLabel()
        self.lbl_meta.setFont(QFont("Segoe UI", 9))
        v_info.addWidget(self.lbl_meta)

        layout.addLayout(v_info, stretch=1)

        # 3. Actions for Finished state
        self.btn_folder = QPushButton()
        self.btn_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_folder, "打开模型所在文件夹")
        self.btn_folder.clicked.connect(lambda: self.open_folder_clicked.emit(self.record))
        layout.addWidget(self.btn_folder)

        self.btn_down = QPushButton()
        self.btn_down.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_down, "另存为 / 下载模型文件")
        self.btn_down.clicked.connect(lambda: self.download_clicked.emit(self.record))
        layout.addWidget(self.btn_down)

        self.btn_del = QPushButton()
        self.btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_del, "删除此导出记录与文件")
        self.btn_del.clicked.connect(lambda: self.delete_clicked.emit(self.record))
        layout.addWidget(self.btn_del)

        # 4. Action for Pending state: Red Trash Can / Cancel button (Matching Fig 1)
        self.btn_cancel = QPushButton()
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_tooltip(self.btn_cancel, "取消导出任务")
        self.btn_cancel.clicked.connect(lambda: self.cancel_clicked.emit(self.record))
        layout.addWidget(self.btn_cancel)

        # Apply theme styling
        self.set_theme(self.is_dark)

        # Initial state setup
        if self.is_pending:
            self._apply_pending_ui()
        else:
            self._apply_finished_ui()

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        hover_bg = "#27272a" if is_dark else "#f9fafb"
        _MkThemeEngine.apply_style_sheet(self, f"""
            #ExportHistoryItem {{
                background: transparent;
                border: none;
                border-radius: 6px;
            }}
            #ExportHistoryItem:hover {{
                background: {hover_bg};
            }}
            #ExportHistoryItem QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        summary_color = "#f1f5f9" if is_dark else "#1f2937"
        meta_color = "#9ca3af" if is_dark else "#6b7280"
        _MkThemeEngine.apply_style_sheet(self.lbl_summary, f"color: {summary_color}; background: transparent; border: none;")
        _MkThemeEngine.apply_style_sheet(self.lbl_meta, f"color: {meta_color}; background: transparent; border: none;")

        btn_hover = "#3f3f46" if is_dark else "#e5e7eb"
        icon_color = "#d4d4d8" if is_dark else "#4b5563"
        self.btn_folder.setIcon(PhFolder.regular(color=icon_color, size=15))
        _MkThemeEngine.apply_style_sheet(self.btn_folder, f"""
            QPushButton {{ background: transparent; border: none; padding: 5px; border-radius: 4px; }}
            QPushButton:hover {{ background: {btn_hover}; }}
        """)
        self.btn_down.setIcon(PhDownloadSimple.regular(color=icon_color, size=15))
        _MkThemeEngine.apply_style_sheet(self.btn_down, f"""
            QPushButton {{ background: transparent; border: none; padding: 5px; border-radius: 4px; }}
            QPushButton:hover {{ background: {btn_hover}; }}
        """)
        del_hover = "#450a0a" if is_dark else "#fee2e2"
        del_color = "#f87171" if is_dark else "#9ca3af"
        self.btn_del.setIcon(PhTrash.regular(color=del_color, size=15))
        _MkThemeEngine.apply_style_sheet(self.btn_del, f"""
            QPushButton {{ background: transparent; border: none; padding: 5px; border-radius: 4px; }}
            QPushButton:hover {{ background: {del_hover}; }}
        """)
        cancel_border = "#7f1d1d" if is_dark else "#fee2e2"
        cancel_hover = "#450a0a" if is_dark else "#fef2f2"
        self.btn_cancel.setIcon(PhTrash.regular(color="#ef4444", size=16))
        _MkThemeEngine.apply_style_sheet(self.btn_cancel, f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {cancel_border};
                border-radius: 6px;
                padding: 4px 6px;
            }}
            QPushButton:hover {{
                background: {cancel_hover};
                border-color: #ef4444;
            }}
        """)

    def _apply_pending_ui(self):
        self.spinner.show()
        self.spinner.start()
        self.lbl_check.hide()

        created_at = self.record.get("created_at", "刚刚")
        self.lbl_meta.setText(f"· {created_at} 正在导出中...")

        self.btn_folder.hide()
        self.btn_down.hide()
        self.btn_del.hide()
        self.btn_cancel.show()

    def _apply_finished_ui(self):
        self.spinner.stop()
        self.spinner.hide()
        self.lbl_check.show()

        size_str = self.record.get("size", "0.0 MB")
        time_cost = self.record.get("duration", "---")
        date_str = self.record.get("created_at", "刚刚")
        self.lbl_meta.setText(f"{size_str} · {time_cost} · {date_str}")

        self.btn_folder.show()
        self.btn_down.show()
        self.btn_del.show()
        self.btn_cancel.hide()

    def mark_finished(self, final_record: dict):
        """Transition smoothly from pending to finished state."""
        self.record.update(final_record)
        self.is_pending = False
        summary_text = self.record.get("summary")
        if not summary_text:
            summary_text = format_export_summary(self.record.get("params", {}))
        self.lbl_summary.setText(summary_text)
        self._apply_finished_ui()


# ─────────────────────────────────────────────────────────────────────────────
# ── 5. 格式主卡片 (ExportFormatCard) 严格对标图 1
# ─────────────────────────────────────────────────────────────────────────────

class ExportFormatCard(QFrame):
    """
    Format card row in main Export tab matching Fig 1:
    - Expand arrow
    - Logo icon
    - Name + tag badge + subtitle
    - Right: Dynamic Starting spinner badge or count badge + '+ Export' button
    - Expandable records container (borderless, transparent)
    """
    export_requested = Signal(str)  # format_key
    record_download = Signal(dict)
    record_delete = Signal(dict)
    record_open_folder = Signal(dict)
    cancel_export_requested = Signal(str, dict)  # format_key, record

    def __init__(self, format_key: str, parent=None):
        super().__init__(parent)
        self.format_key = format_key
        self.spec = EXPORT_SPECS.get(format_key, {})
        self._is_expanded = False
        self._records: List[dict] = []
        self._pending_items: Dict[str, ExportHistoryItem] = {}
        self.is_dark = ThemeEngine.is_dark()

        self.setObjectName("FormatCard")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 12, 16, 12)
        self.main_layout.setSpacing(8)

        self._init_header()
        self._init_expand_container()
        self.set_theme(self.is_dark)
        # The containing experiment page owns theme propagation for these cards.

    def _init_header(self):
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(12)

        # Expand arrow
        self.btn_toggle = QPushButton()
        self.btn_toggle.setIcon(PhCaretRight.bold(color="#6b7280", size=14))
        self.btn_toggle.setFixedSize(22, 22)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        _MkThemeEngine.apply_style_sheet(self.btn_toggle, "QPushButton { background: transparent; border: none; }")
        self.btn_toggle.clicked.connect(self.toggle_expand)
        header_row.addWidget(self.btn_toggle)

        # Format Icon container
        icon_box = QLabel()
        icon_box.setFixedSize(32, 32)
        icon_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color = self.spec.get("icon_color", "#2563eb")
        if self.format_key in ("onnx",):
            icon_box.setPixmap(PhCube.duotone(color=color, size=24).pixmap(24, 24))
        elif self.format_key in ("engine", "openvino", "rknn", "hailo"):
            icon_box.setPixmap(PhCpu.duotone(color=color, size=24).pixmap(24, 24))
        elif self.format_key in ("coreml",):
            icon_box.setPixmap(PhAppleLogo.fill(color=color, size=24).pixmap(24, 24))
        else:
            icon_box.setPixmap(PhDeviceMobile.duotone(color=color, size=24).pixmap(24, 24))
        header_row.addWidget(icon_box)

        # Title + tag + link + desc
        v_title = QVBoxLayout()
        v_title.setSpacing(2)

        top_line = QHBoxLayout()
        top_line.setSpacing(6)

        self.lbl_name = QLabel(self.spec.get("name", self.format_key.upper()))
        self.lbl_name.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        top_line.addWidget(self.lbl_name)

        self.lbl_tag = QLabel(self.spec.get("tag", ""))
        self.lbl_tag.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        top_line.addWidget(self.lbl_tag)

        extra_badge = self.spec.get("badge", "")
        if extra_badge:
            self.lbl_extra = QLabel(extra_badge)
            self.lbl_extra.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            b_col = self.spec.get("badge_color", "#2563eb")
            b_bg = self.spec.get("badge_bg", "#eff6ff")
            _MkThemeEngine.apply_style_sheet(self.lbl_extra, f"background: {b_bg}; color: {b_col}; border-radius: 4px; padding: 1px 5px; border: none;")
            top_line.addWidget(self.lbl_extra)
        else:
            self.lbl_extra = None

        link_url = self.spec.get("doc_url", "")
        if link_url:
            link_lbl = QLabel('<a href="' + link_url + '" style="color: #9ca3af; text-decoration: none;">↗</a>')
            link_lbl.setFont(QFont("Segoe UI", 11))
            link_lbl.setOpenExternalLinks(True)
            _MkThemeEngine.apply_style_sheet(link_lbl, "background: transparent; border: none;")
            top_line.addWidget(link_lbl)

        top_line.addStretch()
        v_title.addLayout(top_line)

        self.lbl_desc = QLabel(self.spec.get("desc", ""))
        self.lbl_desc.setFont(QFont("Segoe UI", 9))
        v_title.addWidget(self.lbl_desc)

        header_row.addLayout(v_title, stretch=1)

        # Dynamic Starting Badge (Matching Fig 1)
        self.badge_starting = QFrame()
        _MkThemeEngine.apply_style_sheet(self.badge_starting, """
            QFrame {
                background: #eff6ff;
                border: 1px solid #dbeafe;
                border-radius: 12px;
            }
        """)
        badge_lay = QHBoxLayout(self.badge_starting)
        badge_lay.setContentsMargins(8, 3, 10, 3)
        badge_lay.setSpacing(6)

        self.spinner_badge = CircularSpinnerWidget(color="#2563eb", size=13, line_width=2.0, parent=self.badge_starting)
        badge_lay.addWidget(self.spinner_badge)

        self.lbl_starting_text = QLabel("正在导出")
        self.lbl_starting_text.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        _MkThemeEngine.apply_style_sheet(self.lbl_starting_text, "color: #2563eb; background: transparent; border: none;")
        badge_lay.addWidget(self.lbl_starting_text)

        self.badge_starting.hide()
        self.spinner_badge.stop()
        header_row.addWidget(self.badge_starting)

        # Right side: exported count badge + '+ 导出' button (using MonkeyQt MkButton)
        self.lbl_count = QLabel("0 已导出")
        self.lbl_count.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        self.lbl_count.hide()
        header_row.addWidget(self.lbl_count)

        self.btn_export = MkButton("+ 导出", size="small", btn_type="default")
        self.btn_export.clicked.connect(lambda: self.export_requested.emit(self.format_key))
        header_row.addWidget(self.btn_export)

        self.main_layout.addLayout(header_row)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        card_bg = "#18181b" if is_dark else "#ffffff"
        card_border = "#27272a" if is_dark else "#e5e7eb"
        _MkThemeEngine.apply_style_sheet(self, f"""
            QFrame#FormatCard {{
                background: {card_bg};
                border: 1px solid {card_border};
                border-radius: 12px;
            }}
        """)
        name_color = "#f1f5f9" if is_dark else "#111827"
        _MkThemeEngine.apply_style_sheet(self.lbl_name, f"color: {name_color}; background: transparent; border: none;")

        tag_bg = "#27272a" if is_dark else "#f3f4f6"
        tag_color = "#a1a1aa" if is_dark else "#4b5563"
        _MkThemeEngine.apply_style_sheet(self.lbl_tag, f"""
            background: {tag_bg}; color: {tag_color};
            border-radius: 4px; padding: 1px 6px; border: none;
        """)

        desc_color = "#a1a1aa" if is_dark else "#6b7280"
        _MkThemeEngine.apply_style_sheet(self.lbl_desc, f"color: {desc_color}; background: transparent; border: none;")

        count_bg = "#27272a" if is_dark else "#f3f4f6"
        count_color = "#d4d4d8" if is_dark else "#374151"
        _MkThemeEngine.apply_style_sheet(self.lbl_count, f"""
            background: {count_bg}; color: {count_color};
            border-radius: 12px; padding: 3px 10px; border: none;
        """)

        if self._is_expanded:
            caret_color = "#f1f5f9" if is_dark else "#111827"
            self.btn_toggle.setIcon(PhCaretDown.bold(color=caret_color, size=14))
        else:
            caret_color = "#a1a1aa" if is_dark else "#6b7280"
            self.btn_toggle.setIcon(PhCaretRight.bold(color=caret_color, size=14))

        # Propagate to all child items
        for i in range(self.records_layout.count()):
            item = self.records_layout.itemAt(i)
            if item and item.widget() and hasattr(item.widget(), "set_theme"):
                item.widget().set_theme(is_dark)

    def _init_expand_container(self):
        self.records_container = QWidget()
        _MkThemeEngine.apply_style_sheet(self.records_container, "background: transparent; border: none;")
        self.records_layout = QVBoxLayout(self.records_container)
        self.records_layout.setContentsMargins(34, 4, 0, 0)
        self.records_layout.setSpacing(4)
        self.records_container.hide()
        self.main_layout.addWidget(self.records_container)

    def toggle_expand(self):
        self.set_expanded(not self._is_expanded)

    def set_expanded(self, expanded: bool):
        self._is_expanded = expanded
        if expanded:
            caret_color = "#f1f5f9" if getattr(self, "is_dark", False) else "#111827"
            self.btn_toggle.setIcon(PhCaretDown.bold(color=caret_color, size=14))
            self.records_container.show()
        else:
            caret_color = "#a1a1aa" if getattr(self, "is_dark", False) else "#6b7280"
            self.btn_toggle.setIcon(PhCaretRight.bold(color=caret_color, size=14))
            self.records_container.hide()

    def set_exporting(self, is_exporting: bool):
        """Toggle top right starting badge and export button state."""
        if is_exporting:
            self.badge_starting.show()
            self.spinner_badge.start()
            self.lbl_count.hide()
            self.btn_export.setEnabled(False)
        else:
            self.badge_starting.hide()
            self.spinner_badge.stop()
            self.btn_export.setEnabled(True)
            if self._records:
                self.lbl_count.show()

    def add_pending_record(self, record_id: str, params: dict) -> ExportHistoryItem:
        """Insert pending item at top of list with active spinner (Fig 1 effect)."""
        pending_rec = {
            "id": record_id,
            "format": self.format_key,
            "summary": format_export_summary(params),
            "created_at": "刚刚",
            "params": params
        }
        item_w = ExportHistoryItem(pending_rec, is_pending=True, parent=self.records_container)
        item_w.set_theme(getattr(self, "is_dark", False))
        item_w.cancel_clicked.connect(lambda rec: self.cancel_export_requested.emit(self.format_key, rec))
        self.records_layout.insertWidget(0, item_w)
        self._pending_items[record_id] = item_w

        self.set_exporting(True)
        self.set_expanded(True)
        return item_w

    def update_record_finished(self, record_id: str, final_record: dict):
        """Mark pending item as finished, transition to green checkmark."""
        item_w = self._pending_items.pop(record_id, None)
        if item_w:
            item_w.mark_finished(final_record)
            item_w.set_theme(getattr(self, "is_dark", False))
            item_w.download_clicked.connect(self.record_download.emit)
            item_w.delete_clicked.connect(self.record_delete.emit)
            item_w.open_folder_clicked.connect(self.record_open_folder.emit)
            self._records.insert(0, final_record)
        else:
            # Fallback
            self._records.insert(0, final_record)
            self.set_records(self._records)

        self.set_exporting(False)
        count = len(self._records)
        self.lbl_count.setText(f"{count} 个已导出")
        self.lbl_count.show()

    def remove_pending_record(self, record_id: str):
        """Cancel or remove a pending record."""
        item_w = self._pending_items.pop(record_id, None)
        if item_w:
            self.records_layout.removeWidget(item_w)
            item_w.setParent(None)
            item_w.deleteLater()

        if not self._pending_items:
            self.set_exporting(False)
            if not self._records:
                self.set_expanded(False)

    def set_records(self, records: List[dict]):
        self._records = list(records)
        # Clear existing non-pending widgets
        while self.records_layout.count():
            item = self.records_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._pending_items.clear()

        count = len(records)
        if count > 0:
            self.lbl_count.setText(f"{count} 个已导出")
            self.lbl_count.show()
            for rec in records:
                item_w = ExportHistoryItem(rec, is_pending=False, parent=self.records_container)
                item_w.set_theme(getattr(self, "is_dark", False))
                item_w.download_clicked.connect(self.record_download.emit)
                item_w.delete_clicked.connect(self.record_delete.emit)
                item_w.open_folder_clicked.connect(self.record_open_folder.emit)
                self.records_layout.addWidget(item_w)
            self.set_expanded(True)
        else:
            self.lbl_count.hide()
            self.set_expanded(False)

