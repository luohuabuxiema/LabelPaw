# -*- coding: utf-8 -*-
"""
Experiment Detail View (训练任务详情页)
Strictly benchmarked against Ultralytics HUB Experiment Detail Pages (Images 3, 4, 5):
- Top Breadcrumb Navigation: Back to Project + Task Title + Architecture / Tags + Status Badge
- Top KPI Metric Cards: mAP50, mAP50-95, Precision, Recall + Dataset Badge
- Tab 0: Overview (Run Information, Training CLI Command, Hyperparameters Table, Performance Table)
- Tab 1: Train (All Thesis-style Metric & Loss Curves with Fullscreen + Live / History Terminal Console)
- Tab 2: Predict (Interactive Inference: Dropzone, Sample Picks, Conf/IoU Sliders, Live Box/Mask Overlay, Latency)
- Tab 3: Export (Model format conversion to ONNX, TensorRT, TorchScript, OpenVINO)
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
import sys
import re
import json
import time
from copy import deepcopy
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QScrollArea, QFrame, QToolButton,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
                               QFileDialog, QMessageBox, QSplitter, QSizePolicy,
                               QPushButton, QLineEdit, QGridLayout, QPlainTextEdit,
                               QStackedWidget, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize, QTimer, QRectF
from PySide6.QtGui import QFont, QColor, QPixmap, QImage, QTextCursor, QTextCharFormat

from monkeyqt import (MkCard, MkButton, MkInput, MkConsole, MkMessage, MkQWidget, SegmentedTabBar, ThemeEngine, MkProTable, MkPage, MkStackedWidget)
from monkeyqt_icons import (PhArrowLeft, PhPlay, PhDownloadSimple, PhFolder,
                            PhCopy, PhCheck, PhFileText, PhSliders, PhSparkle,
                            PhCpu, PhTerminalWindow, PhChartLine, PhMagnifyingGlass, PhLayout)

from TrainingPlatform.ui.components.thesis_chart import ThesisChartCard, ACADEMIC_COLORS
from TrainingPlatform.ui.components.terminal_formatter import (apply_terminal_console_style,
                                                               append_terminal_line_to_console)
from TrainingPlatform.ui.components.predict_widgets import (
    LoadingOverlayWidget, DropzoneWidget, ExampleThumbCard, FloatingToolbarWidget
)
from TrainingPlatform.ui.components.export_widgets import (
    BaseExportDialog, ExportFormatCard, EXPORT_SPECS
)
from TrainingPlatform.ui.components.custom_tooltip import attach_tooltip, CustomTooltipPopover


class InferenceWorkerThread(QThread):
    """Background thread running YOLO prediction without blocking the GUI."""
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, weights_path: str, image_path: str, conf: float, iou: float, imgsz: int, parent=None):
        super().__init__(parent)
        self.weights_path = weights_path
        self.image_path = image_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def run(self):
        try:
            from ultralytics import YOLO
            import cv2

            if not os.path.exists(self.weights_path):
                self.error_occurred.emit(f"模型权重文件不存在: {self.weights_path}")
                return

            model = YOLO(self.weights_path)
            t0 = time.time()
            results = model.predict(
                source=self.image_path,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False
            )
            total_time = (time.time() - t0) * 1000.0

            res = results[0]
            plotted_bgr = res.plot()  # numpy array BGR
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

            h, w, ch = plotted_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(plotted_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            # Parse detections
            detections = []
            names = res.names or {}
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0].item()) if hasattr(b, "cls") else 0
                    cls_name = names.get(cls_id, str(cls_id))
                    conf_val = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                    xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
                    detections.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "box": xyxy
                    })

            speed = getattr(res, "speed", {})
            out_data = {
                "qimage": q_img,
                "raw_rgb": plotted_rgb,
                "detections": detections,
                "speed": {
                    "preprocess": speed.get("preprocess", 0.0),
                    "inference": speed.get("inference", 0.0),
                    "postprocess": speed.get("postprocess", 0.0),
                    "total": total_time
                },
                "orig_shape": (w, h)
            }
            self.result_ready.emit(out_data)

        except Exception as e:
            self.error_occurred.emit(str(e))


class ExportWorkerThread(QThread):
    """
    Background worker that runs YOLO model export inside an isolated subprocess.
    Guarantees:
    - Zero GUI freezing, completely asynchronous.
    - Zero process crashes (even if onnx/onnxslim/tensorrt DLLs encounter 0xC0000005).
    - Clean cancellation via process.kill().
    """
    finished = Signal(bool, str, dict)  # success, file_path_or_err, meta_dict

    def __init__(self, weights_path: str, params: dict, parent=None):
        super().__init__(parent)
        self.weights_path = weights_path
        self.params = deepcopy(params)
        self._proc: Optional[subprocess.Popen] = None
        self._is_cancelled = False
        self._tmp_cfg_path: Optional[str] = None

    def cancel(self):
        self._is_cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self):
        import tempfile
        import subprocess

        # 1. Write params to temp file
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump(self.params, f, ensure_ascii=False)
                self._tmp_cfg_path = f.name
        except Exception as e:
            self.finished.emit(False, f"准备导出参数文件失败: {e}", {})
            return

        runner_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "core", "export_runner.py")
        )

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            runner_script,
            "--weights", self.weights_path,
            "--config", self._tmp_cfg_path
        ]

        # Hide console window on Windows
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                startupinfo=startupinfo
            )

            stdout_lines = []
            result_data = None

            while True:
                line = self._proc.stdout.readline()
                if not line and self._proc.poll() is not None:
                    break
                if line:
                    line_str = line.strip()
                    stdout_lines.append(line_str)
                    try:
                        print(f"[ExportWorker] {line_str}", flush=True)
                    except Exception:
                        pass
                    if line_str.startswith("__EXPORT_RESULT__:"):
                        try:
                            json_str = line_str[len("__EXPORT_RESULT__:"):]
                            result_data = json.loads(json_str)
                        except Exception:
                            pass

            rc = self._proc.poll()

            if self._is_cancelled:
                self.finished.emit(False, "用户取消了导出任务", {})
                return

            if result_data and result_data.get("success"):
                exported_path = result_data.get("path", "")
                self.finished.emit(True, exported_path, result_data)
            else:
                err_msg = ""
                if result_data and result_data.get("error"):
                    err_msg = result_data.get("error")
                elif rc is not None and rc != 0:
                    hex_code = f"0x{rc & 0xFFFFFFFF:08X}"
                    err_msg = f"底层导出引擎异常退出 (错误代码: {hex_code})\n" + "\n".join(stdout_lines[-5:])
                else:
                    err_msg = "导出未返回有效结果\n" + "\n".join(stdout_lines[-5:])

                self.finished.emit(False, err_msg, {})

        except Exception as e:
            self.finished.emit(False, f"启动隔离导出进程失败: {e}", {})
        finally:
            if self._tmp_cfg_path and os.path.exists(self._tmp_cfg_path):
                try:
                    os.remove(self._tmp_cfg_path)
                except Exception:
                    pass


class RunInformationCard(MkCard):
    """
    Modern Run Information Card benchmarked against Ultralytics HUB (Image 1):
    - Displays strictly the circled fields:
      1. Start Time (开始时间)
      2. Python Version (Python 版本)
      3. CPU (Cores badge + Processor brand)
      4. Ultralytics (Version + ↗ link)
      5. GPU (Device count badge + GPU Model name)
      6. Command (Terminal command box with copy button)
    - Status pill badge in top right (e.g. "Completed")
    """

    def __init__(self, is_dark: bool = False, parent=None):
        super().__init__(title="运行信息 (Run Information)", show_title=False, parent=parent)
        self.is_dark = is_dark
        self.cmd_text = ""
        self.cpu_count = 1
        self.gpu_count = 0
        self.cpu_name = "Generic Processor"
        self.gpu_name = "CPU Only"
        self.setObjectName("RunInformationCard")
        self.setMinimumHeight(320)
        self._init_ui()
        self.set_theme(is_dark)

    def _init_ui(self):
        main_layout = self.content_layout
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)

        # ── 1. Top Header (Title + Subtitle + Status Badge) ──
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        v_title = QVBoxLayout()
        v_title.setSpacing(3)
        self.lbl_title = QLabel("运行信息 (Run Information)")
        self.lbl_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        v_title.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("训练时间与运行环境配置详情")
        self.lbl_subtitle.setFont(QFont("Segoe UI", 10))
        v_title.addWidget(self.lbl_subtitle)
        header_layout.addLayout(v_title, stretch=1)

        # Status Pill Badge
        self.badge_status = QLabel("已完成")
        self.badge_status.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.badge_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.badge_status)

        main_layout.addLayout(header_layout)

        # ── 2. Environment & Hardware Metrics Grid (Circled Items in Image 1) ──
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(32)
        grid_layout.setVerticalSpacing(16)

        # Row 0: Start Time | Python Version
        # Col 0: Start Time
        v_start = QVBoxLayout()
        v_start.setSpacing(4)
        self.lbl_start_title = QLabel("开始时间")
        self.lbl_start_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.lbl_start_val = QLabel("-")
        self.lbl_start_val.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        v_start.addWidget(self.lbl_start_title)
        v_start.addWidget(self.lbl_start_val)
        grid_layout.addLayout(v_start, 0, 0)

        # Col 1: Python Version
        v_py = QVBoxLayout()
        v_py.setSpacing(4)
        self.lbl_py_title = QLabel("Python 版本")
        self.lbl_py_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.lbl_py_val = QLabel("-")
        self.lbl_py_val.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        v_py.addWidget(self.lbl_py_title)
        v_py.addWidget(self.lbl_py_val)
        grid_layout.addLayout(v_py, 0, 1)

        # Row 1: CPU | GPU
        # Col 0: CPU
        v_cpu = QVBoxLayout()
        v_cpu.setSpacing(4)
        self.lbl_cpu_title = QLabel("CPU")
        self.lbl_cpu_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.lbl_cpu_val = QLabel("-")
        self.lbl_cpu_val.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        v_cpu.addWidget(self.lbl_cpu_title)
        v_cpu.addWidget(self.lbl_cpu_val)
        grid_layout.addLayout(v_cpu, 1, 0)

        # Col 1: GPU
        v_gpu = QVBoxLayout()
        v_gpu.setSpacing(4)
        self.lbl_gpu_title = QLabel("GPU")
        self.lbl_gpu_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.lbl_gpu_val = QLabel("-")
        self.lbl_gpu_val.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        v_gpu.addWidget(self.lbl_gpu_title)
        v_gpu.addWidget(self.lbl_gpu_val)
        grid_layout.addLayout(v_gpu, 1, 1)

        main_layout.addLayout(grid_layout)

        # ── 3. CLI Command Terminal Box (Image 1) ──
        cmd_container = QVBoxLayout()
        cmd_container.setSpacing(8)

        cmd_header = QHBoxLayout()
        self.lbl_cmd_header = QLabel("训练命令 (Command)")
        self.lbl_cmd_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        attach_tooltip(self.lbl_cmd_header, "可直接在终端中复制并执行此命令复现模型训练")
        cmd_header.addWidget(self.lbl_cmd_header)
        cmd_header.addStretch()

        self.btn_copy = MkButton("复制", size="small", icon=PhCopy.regular(size=13))
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.clicked.connect(self._copy_cmd)
        cmd_header.addWidget(self.btn_copy)
        cmd_container.addLayout(cmd_header)

        # Command Code Box (Monospace + Horizontal Scroll)
        self.editor_cmd = QPlainTextEdit()
        self.editor_cmd.setReadOnly(True)
        self.editor_cmd.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor_cmd.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor_cmd.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.editor_cmd.setFixedHeight(50)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor_cmd.setFont(font)
        cmd_container.addWidget(self.editor_cmd)

        main_layout.addLayout(cmd_container)

    def load_task_info(self, task: dict, save_dir: str = ""):
        """Populate the circled fields from task metadata & environment."""
        if not task:
            return

        # 1. 开始时间 (中国标准时间格式: 2026年9月10日 20:45)
        raw_start = task.get("start_time") or task.get("created_at") or ""
        formatted_start = "-"
        if raw_start:
            try:
                clean_str = str(raw_start).replace("T", " ")[:19]
                if len(clean_str) >= 19:
                    dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                    if dt.second:
                        formatted_start = f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
                    else:
                        formatted_start = f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"
                elif len(clean_str) == 16:
                    dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M")
                    formatted_start = f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"
                else:
                    formatted_start = clean_str
            except Exception:
                formatted_start = str(raw_start)
        self.lbl_start_val.setText(formatted_start)
        self.lbl_start_val.setToolTip(str(raw_start))

        # 2. Status Badge
        status = str(task.get("status", "SUCCESS")).upper()
        self._set_status_badge(status)

        # 3. Python Version
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.lbl_py_val.setText(py_ver)

        # 5. Hardware Info (CPU & GPU)
        cpu_name = "Generic Processor"
        cpu_count = os.cpu_count() or 1
        try:
            if sys.platform == "win32":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            elif sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":", 1)[1].strip()
                            break
        except Exception:
            import platform
            cpu_name = platform.processor() or "CPU"

        gpu_name = "CPU Only"
        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
        except Exception:
            pass

        self.cpu_count = cpu_count
        self.gpu_count = gpu_count
        self.cpu_name = cpu_name
        self.gpu_name = gpu_name
        self._update_hardware_labels(cpu_name, gpu_name)

        # 6. Real Terminal Command (Reconstructed or from task config)
        self._load_command(task, save_dir)

    def _update_hardware_labels(self, cpu_name: str, gpu_name: str):
        lbl_fg = "#a6adc8" if self.is_dark else "#64748b"
        badge_fg = "#60a5fa" if self.is_dark else "#2563eb"

        self.lbl_cpu_title.setText(
            f'<span style="color: {lbl_fg}; font-weight: 600;">CPU</span> '
            f'<span style="color: {badge_fg}; font-size: 11px; font-weight: 600;">{self.cpu_count}</span>'
        )
        self.lbl_cpu_val.setText(cpu_name)

        self.lbl_gpu_title.setText(
            f'<span style="color: {lbl_fg}; font-weight: 600;">GPU</span> '
            f'<span style="color: {badge_fg}; font-size: 11px; font-weight: 600;">{self.gpu_count}</span>'
        )
        self.lbl_gpu_val.setText(gpu_name)

    def _load_command(self, task: dict, save_dir: str):
        # Load args.yaml if present
        args_data = {}
        if save_dir and os.path.isdir(save_dir):
            args_path = os.path.join(save_dir, "args.yaml")
            if os.path.isfile(args_path):
                try:
                    import yaml
                    with open(args_path, "r", encoding="utf-8") as f:
                        args_data = yaml.safe_load(f) or {}
                except Exception:
                    pass

        cfg = {}
        raw_cfg = task.get("config_json")
        if isinstance(raw_cfg, str):
            try:
                cfg = json.loads(raw_cfg)
            except Exception:
                pass
        elif isinstance(raw_cfg, dict):
            cfg = raw_cfg

        # 1. Check if explicit cli_command was stored
        cli_cmd = cfg.get("cli_command", "").strip()

        # 2. Reconstruct exact CLI command matching train_create_dialog
        if not cli_cmd:
            model = args_data.get("model") or task.get("model_name") or "yolo26n-seg.pt"
            data = args_data.get("data") or task.get("dataset_yaml") or ""
            task_type = args_data.get("task") or "segment"
            name = args_data.get("name") or task.get("task_name") or "exp"
            epochs = args_data.get("epochs") if "epochs" in args_data else cfg.get("epochs", 100)
            batch = args_data.get("batch") if "batch" in args_data else cfg.get("batch", -1)
            imgsz = args_data.get("imgsz") if "imgsz" in args_data else cfg.get("imgsz", 640)
            device = args_data.get("device") if "device" in args_data else cfg.get("device", "0")

            parts = [
                "yolo train",
                f'model="{model}"',
                f'data="{data}"',
                f'task="{task_type}"',
                f'name="{name}"',
                f'epochs={epochs}',
                f'batch={batch}',
                f'imgsz={imgsz}',
                f'device="{device}"'
            ]
            cli_cmd = " ".join(parts)

        self.cmd_text = cli_cmd
        self.editor_cmd.setPlainText(cli_cmd)

    def _set_status_badge(self, status: str):
        if status in ("SUCCESS", "COMPLETED"):
            text = "已完成"
            bg = "#064e3b" if self.is_dark else "#ecfdf5"
            fg = "#34d399" if self.is_dark else "#059669"
            bd = "#047857" if self.is_dark else "#a7f3d0"
        elif status == "RUNNING":
            text = "运行中"
            bg = "#1e3a8a" if self.is_dark else "#eff6ff"
            fg = "#60a5fa" if self.is_dark else "#2563eb"
            bd = "#1d4ed8" if self.is_dark else "#bfdbfe"
        elif status in ("FAILED", "ERROR"):
            text = "训练失败"
            bg = "#7f1d1d" if self.is_dark else "#fef2f2"
            fg = "#f87171" if self.is_dark else "#dc2626"
            bd = "#991b1b" if self.is_dark else "#fecaca"
        elif status == "WAITING":
            text = "等待中"
            bg = "#78350f" if self.is_dark else "#fffbeb"
            fg = "#fbbf24" if self.is_dark else "#d97706"
            bd = "#92400e" if self.is_dark else "#fde68a"
        else:
            text = f"{status}"
            bg = "#313244" if self.is_dark else "#f8fafc"
            fg = "#a6adc8" if self.is_dark else "#64748b"
            bd = "#45475a" if self.is_dark else "#e2e8f0"

        self.badge_status.setText(text)
        _MkThemeEngine.apply_style_sheet(self.badge_status, f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {bd};
                border-radius: 12px;
                padding: 4px 12px;
            }}
        """)

    def _copy_cmd(self):
        text = self.cmd_text.strip()
        if text:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)
            self.btn_copy.setText("已复制!")
            self.btn_copy.setIcon(PhCheck.bold(size=13, color="#16a34a"))
            QTimer.singleShot(1500, self._reset_copy_btn)
            MkMessage.success(self, "训练命令已复制到剪贴板！")

    def _reset_copy_btn(self):
        self.btn_copy.setText("复制")
        self.btn_copy.setIcon(PhCopy.regular(size=13))

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        text_primary = "#cdd6f4" if is_dark else "#0f172a"
        text_muted = "#a6adc8" if is_dark else "#64748b"

        _MkThemeEngine.apply_style_sheet(self.lbl_title, f"color: {text_primary}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_subtitle, f"color: {text_muted}; background: transparent;")

        _MkThemeEngine.apply_style_sheet(self.lbl_start_title, f"color: {text_muted}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_start_val, f"color: {text_primary}; background: transparent;")

        _MkThemeEngine.apply_style_sheet(self.lbl_py_title, f"color: {text_muted}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_py_val, f"color: {text_primary}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_cpu_val, f"color: {text_primary}; background: transparent;")
        _MkThemeEngine.apply_style_sheet(self.lbl_gpu_val, f"color: {text_primary}; background: transparent;")
        self._update_hardware_labels(self.cpu_name, self.gpu_name)

        _MkThemeEngine.apply_style_sheet(self.lbl_cmd_header, f"color: {text_muted}; background: transparent;")

    def _get_skeleton_blocks(self, rect, content_w, content_h, pad_x, pad_y):
        """生成与真实字段 1:1 精确对应的流光骨架屏布局块（100% 清空真实文字，杜绝大色块）"""
        blocks = []
        # 1. 顶部标题 + 副标题 + 右侧状态胶囊
        blocks.append(QRectF(float(pad_x), float(pad_y + 4), min(180.0, content_w * 0.35), 18.0))
        blocks.append(QRectF(float(pad_x), float(pad_y + 26), min(140.0, content_w * 0.25), 12.0))
        blocks.append(QRectF(float(rect.right()) - float(pad_x) - 68.0, float(pad_y + 4), 68.0, 22.0))

        # 2. 硬件与环境指标栅格 (2 行 x 2 列，对应开始时间/Python版本/CPU/GPU)
        col_w = (content_w - 32.0) / 2.0
        col0_x = float(pad_x)
        col1_x = float(pad_x) + col_w + 32.0

        # 行 0: 开始时间 | Python 版本
        y0 = float(pad_y) + 56.0
        blocks.append(QRectF(col0_x, y0, 60.0, 12.0))
        blocks.append(QRectF(col0_x, y0 + 18.0, min(140.0, col_w * 0.6), 16.0))
        blocks.append(QRectF(col1_x, y0, 65.0, 12.0))
        blocks.append(QRectF(col1_x, y0 + 18.0, min(90.0, col_w * 0.4), 16.0))

        # 行 1: CPU | GPU
        y1 = float(pad_y) + 104.0
        blocks.append(QRectF(col0_x, y1, 45.0, 12.0))
        blocks.append(QRectF(col0_x, y1 + 18.0, min(180.0, col_w * 0.75), 16.0))
        blocks.append(QRectF(col1_x, y1, 45.0, 12.0))
        blocks.append(QRectF(col1_x, y1 + 18.0, min(120.0, col_w * 0.55), 16.0))

        # 3. 训练命令终端框 (标题 + 终端框)
        y2 = float(pad_y) + 152.0
        blocks.append(QRectF(float(pad_x), y2, 110.0, 12.0))
        blocks.append(QRectF(float(pad_x), y2 + 18.0, content_w, 46.0))

        return blocks




class ExperimentDetailView(MkPage):
    """
    Dedicated view for inspecting a single training experiment.
    Matches Image 3 (Overview), Image 4 (Train & Charts), Image 5 (Predict & Export).
    """
    back_requested = Signal()

    def __init__(self, db_manager=None, task_manager=None, parent=None):
        super().__init__(parent=parent, layout="v", margins=(24, 16, 24, 16), spacing=14, role="transparent")
        self.db = db_manager
        self.task_mgr = task_manager
        self.current_task: Optional[dict] = None
        self.current_task_id: Optional[int] = None
        self.current_metrics_df: Optional[pd.DataFrame] = None
        self.last_predicted_image_rgb = None
        self.is_dark = ThemeEngine.is_dark()

        self._init_ui()
        if self.task_mgr and hasattr(self.task_mgr, "task_log"):
            self.task_mgr.task_log.connect(self._on_task_log)
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)
        self.apply_theme()

    def set_loading(self, loading: bool = True, recursive: bool = True):
        """统一管理整页全部卡片、表格、图表和按钮的骨架屏加载态，100% 清空文字与多余控件"""
        super().set_loading(loading, recursive=recursive)

        # 1. 明确级联更新所有卡片与组件
        for kpi in (getattr(self, "card_map50", None),
                    getattr(self, "card_map50_95", None),
                    getattr(self, "card_precision", None),
                    getattr(self, "card_recall", None)):
            if kpi and hasattr(kpi, "set_loading"):
                kpi.set_loading(loading)

        if hasattr(self, "card_run_info") and self.card_run_info:
            self.card_run_info.set_loading(loading)

        if hasattr(self, "card_training_config") and self.card_training_config:
            self.card_training_config.set_loading(loading)

        # 2. 加载期间彻底隐藏操作按钮，避免突兀文字
        if hasattr(self, "btn_export_hyper") and self.btn_export_hyper:
            self.btn_export_hyper.setVisible(not loading)
        if hasattr(self, "btn_open_folder") and self.btn_open_folder:
            self.btn_open_folder.setEnabled(not loading)

        for attr in ("chart_precision", "chart_recall", "chart_map50", "chart_map50_95",
                     "chart_box_loss", "chart_seg_loss", "chart_cls_loss", "chart_val_loss"):
            ch = getattr(self, attr, None)
            if ch and hasattr(ch, "set_loading"):
                ch.set_loading(loading)

    def _on_task_log(self, task_id: int, msg: str):
        if self.current_task_id == task_id:
            self.append_terminal_line(msg)

    def _init_ui(self):
        main_layout = self.inner_layout

        # ── 1. Top Navigation & Header ──
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        self.btn_back = MkButton(" 返回项目", btn_type="default")
        self.btn_back.setIcon(PhArrowLeft.regular(size=15))
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_back.clicked.connect(self.back_requested.emit)
        header_layout.addWidget(self.btn_back)

        self.lbl_task_title = QLabel("exp-1")
        self.lbl_task_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header_layout.addWidget(self.lbl_task_title)

        self.lbl_task_type_tag = QLabel("Segment")
        _MkThemeEngine.apply_style_sheet(self.lbl_task_type_tag, """
            background: #eff6ff; color: #2563eb; font-weight: 600;
            padding: 3px 8px; border-radius: 4px; font-size: 11px;
            border: 1px solid #dbeafe;
        """)
        header_layout.addWidget(self.lbl_task_type_tag)

        self.lbl_status_badge = QLabel("COMPLETED")
        _MkThemeEngine.apply_style_sheet(self.lbl_status_badge, """
            background: #dcfce7; color: #15803d; font-weight: 700;
            padding: 3px 8px; border-radius: 4px; font-size: 11px;
        """)
        header_layout.addWidget(self.lbl_status_badge)

        self.lbl_updated_time = QLabel("刚刚更新")
        _MkThemeEngine.apply_style_sheet(self.lbl_updated_time, "color: #94a3b8; font-size: 12px;")
        header_layout.addWidget(self.lbl_updated_time)

        header_layout.addStretch()

        self.btn_open_folder = MkButton("打开目录", btn_type="default")
        self.btn_open_folder.setIcon(PhFolder.regular(size=14))
        self.btn_open_folder.clicked.connect(self._open_task_folder)
        header_layout.addWidget(self.btn_open_folder)

        main_layout.addLayout(header_layout)

        # ── 2. Top KPI Cards Ribbon (Image 3: Standard casing mAP50, mAP50-95, Precision, Recall) ──
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(12)

        self.card_map50 = self._create_kpi_card("mAP50", "-", "#2563eb", "#eff6ff")
        self.card_map50_95 = self._create_kpi_card("mAP50-95", "-", "#7c3aed", "#f5f3ff")
        self.card_precision = self._create_kpi_card("Precision", "-", "#059669", "#ecfdf5")
        self.card_recall = self._create_kpi_card("Recall", "-", "#d97706", "#fffbeb")

        kpi_layout.addWidget(self.card_map50)
        kpi_layout.addWidget(self.card_map50_95)
        kpi_layout.addWidget(self.card_precision)
        kpi_layout.addWidget(self.card_recall)

        main_layout.addLayout(kpi_layout)

        # ── 3. Primary Segmented Navigation Tab Bar (Overview | Train | Predict | Export) ──
        nav_container = QHBoxLayout()
        nav_container.setContentsMargins(0, 2, 0, 4)

        self.primary_nav = SegmentedTabBar(size="large")
        self.primary_nav.add_tab("overview", "概览", PhLayout)
        self.primary_nav.add_tab("train", "训练与曲线", PhChartLine)
        self.primary_nav.add_tab("predict", "推理预测", PhSparkle)
        self.primary_nav.add_tab("export", "模型导出", PhDownloadSimple)

        nav_container.addWidget(self.primary_nav)
        nav_container.addStretch()
        main_layout.addLayout(nav_container)

        self.pages_stack = MkStackedWidget()
        _MkThemeEngine.apply_style_sheet(self.pages_stack, "QStackedWidget { background: transparent; border: none; }")
        self.pages_stack.currentChanged.connect(self._on_stack_page_changed)

        # Page 0: Overview (Image 3)
        self.tab_overview = MkQWidget(role="transparent")
        self._setup_overview_tab()
        self.pages_stack.addWidget(self.tab_overview)

        # Page 1: Train & Charts (Image 4)
        self.tab_train = MkQWidget(role="transparent")
        self._setup_train_tab()
        self.pages_stack.addWidget(self.tab_train)

        # Page 2: Predict (Image 5)
        self.tab_predict = MkQWidget(role="transparent")
        self._setup_predict_tab()
        self.pages_stack.addWidget(self.tab_predict)

        # Page 3: Export
        self.tab_export = MkQWidget(role="transparent")
        self._setup_export_tab()
        self.pages_stack.addWidget(self.tab_export)

        self.primary_nav.attach_to_stack(self.pages_stack)
        main_layout.addWidget(self.pages_stack, stretch=1)

    def _on_stack_page_changed(self, index: int):
        self.scroll_to_top()

    def _create_kpi_card(self, title: str, val: str, accent_color: str, bg_tint: str) -> MkCard:
        card = MkCard(title=title, show_title=False, parent=self)
        card.setObjectName("KpiCard")
        card._accent_color = accent_color
        c_layout = card.content_layout
        c_layout.setContentsMargins(18, 14, 18, 14)
        c_layout.setSpacing(4)

        lbl_t = QLabel(title)
        _MkThemeEngine.apply_style_sheet(lbl_t, "color: #64748b; font-size: 13px; font-weight: 600;")
        lbl_v = QLabel(val)
        lbl_v.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        _MkThemeEngine.apply_style_sheet(lbl_v, f"color: {accent_color}; background: transparent;")
        card._lbl_val = lbl_v
        card._lbl_title = lbl_t

        c_layout.addWidget(lbl_t)
        c_layout.addWidget(lbl_v)

        def _card_set_theme(is_dark: bool):
            fg_title = "#a6adc8" if is_dark else "#64748b"
            _MkThemeEngine.apply_style_sheet(lbl_t, f"color: {fg_title}; font-size: 13px; font-weight: 600; background: transparent;")
            _MkThemeEngine.apply_style_sheet(lbl_v, f"color: {card._accent_color}; background: transparent;")

        card.set_theme = _card_set_theme
        _card_set_theme(self.is_dark if hasattr(self, "is_dark") else False)
        return card

    # ─────────────────────────────────────────────────────────
    # ── Tab 0: Overview (概览)
    # ─────────────────────────────────────────────────────────
    # ── Hyperparameter export helper (shared with MkProTable action button) ──
    def _export_hyperparameters(self):
        raw = getattr(self, "_hyper_raw_config", {})
        current = getattr(self, "_hyper_current_config", {})
        if not current:
            MkMessage.warning(self, "暂无超参数数据可导出")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "导出训练超参数配置 (Training Configuration)",
            "training_configuration.json",
            "JSON Files (*.json);;YAML Files (*.yaml);;CSV Files (*.csv)"
        )
        if not filepath:
            return
        try:
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                import yaml
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(raw if raw else current, f, allow_unicode=True, default_flow_style=False)
            elif filepath.endswith(".csv"):
                import csv
                with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Parameter", "Value"])
                    for k, v in current.items():
                        writer.writerow([k, v])
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(raw if raw else current, f, indent=2, ensure_ascii=False)
            MkMessage.success(self, f"训练配置已成功导出至:\n{os.path.basename(filepath)}")
        except Exception as e:
            MkMessage.error(self, f"导出配置失败: {e}")

    def _setup_overview_tab(self):
        layout = QVBoxLayout(self.tab_overview)
        layout.setContentsMargins(0, 8, 0, 16)
        layout.setSpacing(16)

        # 1. Run Information Card
        self.card_run_info = RunInformationCard(is_dark=False)
        layout.addWidget(self.card_run_info)

        # 2. Hyperparameters — MkProTable (开箱即用搜索、排序、分页、主题联动)
        hyper_header = QHBoxLayout()
        hyper_header.setContentsMargins(0, 0, 0, 0)
        hyper_header.addStretch()
        self.btn_export_hyper = MkButton("导出数据", size="small", icon=PhDownloadSimple.regular(size=14))
        self.btn_export_hyper.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_hyper.clicked.connect(self._export_hyperparameters)
        hyper_header.addWidget(self.btn_export_hyper)
        layout.addLayout(hyper_header)

        # MkProTable with built-in search and sort
        self._hyper_raw_config: Dict[str, Any] = {}
        self._hyper_current_config: Dict[str, Any] = {}
        self.card_training_config = MkProTable(
            columns=[
                {"key": "param",  "label": "参数名称", "sortable": True},
                {"key": "value",  "label": "设定值",   "sortable": True},
                {"key": "desc",   "label": "说明",     "sortable": False},
            ],
            data=[],
            title="训练超参数配置 (Training Configuration)",
            description="模型训练过程中使用的全部超参数设定",
            searchable=True,
            search_placeholder="搜索超参数...",
            selectable=False,
            page_size=20,
            page_size_options=[10, 20, 50, 100],
            row_height=40,
            border_radius=12,
            auto_height=True,
        )
        layout.addWidget(self.card_training_config)

    def _copy_cmd(self):
        if hasattr(self, "card_run_info"):
            self.card_run_info._copy_cmd()

    # ─────────────────────────────────────────────────────────
    # ── Tab 1: Train & Charts (训练与曲线 - Image 4)
    # ─────────────────────────────────────────────────────────
    def _setup_train_tab(self):
        layout = QVBoxLayout(self.tab_train)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)

        # Sub-Navigation: Charts (指标图表) & Console (终端控制台)
        sub_nav_row = QHBoxLayout()
        sub_nav_row.setContentsMargins(0, 0, 0, 4)

        self.train_sub_nav = SegmentedTabBar(size="default")
        self.train_sub_nav.add_tab("charts", "指标图表 (Charts)", PhChartLine)
        self.train_sub_nav.add_tab("console", "终端控制台 (Console)", PhTerminalWindow)
        sub_nav_row.addWidget(self.train_sub_nav)
        sub_nav_row.addStretch()
        layout.addLayout(sub_nav_row)

        self.train_sub_stack = MkStackedWidget()
        _MkThemeEngine.apply_style_sheet(self.train_sub_stack, "QStackedWidget { background: transparent; border: none; }")
        self.train_sub_stack.currentChanged.connect(lambda idx: (self.train_sub_stack.updateGeometry(), self.updateGeometry()))

        # Sub-page 0: Charts (Metrics & Loss)
        tab_charts = QWidget()
        cc_layout = QVBoxLayout(tab_charts)
        cc_layout.setContentsMargins(0, 8, 0, 16)
        cc_layout.setSpacing(16)

        # Section 1: Metrics
        lbl_sec1 = QLabel("评估指标曲线 (Metrics)")
        lbl_sec1.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        cc_layout.addWidget(lbl_sec1)

        row1 = QHBoxLayout()
        row1.setSpacing(12)
        self.chart_precision = ThesisChartCard("metrics/precision(B)", "precision(B)", is_dark=False)
        self.chart_recall = ThesisChartCard("metrics/recall(B)", "recall(B)", is_dark=False)
        self.chart_map50 = ThesisChartCard("metrics/mAP50(B)", "mAP50(B)", is_dark=False)
        self.chart_map50_95 = ThesisChartCard("metrics/mAP50-95(B)", "mAP50-95(B)", is_dark=False)
        row1.addWidget(self.chart_precision)
        row1.addWidget(self.chart_recall)
        row1.addWidget(self.chart_map50)
        row1.addWidget(self.chart_map50_95)
        cc_layout.addLayout(row1)

        # Section 2: Loss
        lbl_sec2 = QLabel("损失函数收敛曲线 (Losses)")
        lbl_sec2.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        cc_layout.addWidget(lbl_sec2)

        row2 = QHBoxLayout()
        row2.setSpacing(12)
        self.chart_box_loss = ThesisChartCard("train/box_loss", "train/box_loss", is_dark=False)
        self.chart_seg_loss = ThesisChartCard("train/seg_loss", "train/seg_loss", is_dark=False)
        self.chart_cls_loss = ThesisChartCard("train/cls_loss", "train/cls_loss", is_dark=False)
        self.chart_val_loss = ThesisChartCard("val/box_loss", "val/box_loss", is_dark=False)
        row2.addWidget(self.chart_box_loss)
        row2.addWidget(self.chart_seg_loss)
        row2.addWidget(self.chart_cls_loss)
        row2.addWidget(self.chart_val_loss)
        cc_layout.addLayout(row2)

        cc_layout.addStretch()
        self.train_sub_stack.addWidget(tab_charts)

        # Sub-page 1: Console
        tab_console = QWidget()
        tc_layout = QVBoxLayout(tab_console)
        tc_layout.setContentsMargins(0, 4, 0, 0)
        self.train_console = MkConsole()
        self.train_console.setMinimumHeight(420)
        self.train_console.set_max_lines(5000)

        tc_layout.addWidget(self.train_console)
        self.train_sub_stack.addWidget(tab_console)

        self.train_sub_nav.attach_to_stack(self.train_sub_stack)
        layout.addWidget(self.train_sub_stack, stretch=1)

    # ─────────────────────────────────────────────────────────
    # ── Tab 2: Predict (推理预测 - Image 5)
    # ─────────────────────────────────────────────────────────
    def _setup_predict_tab(self):
        # ── Assets directory (bus.jpg / zidane.jpg) ──────────────────
        _tp_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets")
        )
        self._built_in_bus = os.path.join(_tp_root, "bus.jpg")
        self._built_in_zidane = os.path.join(_tp_root, "zidane.jpg")
        self.current_test_image_path = ""
        self.last_predicted_image_rgb = None
        self._selected_example_card = None  # track which card is highlighted

        layout = QHBoxLayout(self.tab_predict)
        layout.setContentsMargins(0, 8, 0, 16)
        layout.setSpacing(16)

        # ── Left Column ──────────────────────────────────────────────
        left_col = QWidget()
        lc_layout = QVBoxLayout(left_col)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(10)

        # Header row
        header_row = QHBoxLayout()
        self.lbl_predict_in = QLabel("输入图片 (Input Image)")
        self.lbl_predict_in.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        _MkThemeEngine.apply_style_sheet(self.lbl_predict_in, "color: #0f172a;")
        header_row.addWidget(self.lbl_predict_in)
        header_row.addStretch()
        lc_layout.addLayout(header_row)

        self.lbl_predict_sub = QLabel("上传图片或选择内置示例，选中后自动开始推理")
        _MkThemeEngine.apply_style_sheet(self.lbl_predict_sub, "color: #64748b; font-size: 11px;")
        lc_layout.addWidget(self.lbl_predict_sub)

        # ── Image preview area: QStackedWidget ───────────────────────
        # Page 0: DropzoneWidget  |  Page 1: result image label
        self._preview_stack = QStackedWidget()
        self._preview_stack.setMinimumHeight(240)

        # Page 0 – Dropzone
        self._dropzone = DropzoneWidget()
        self._dropzone.imageSelected.connect(self._on_image_selected_and_predict)
        self._preview_stack.addWidget(self._dropzone)  # index 0

        # Page 1 – Result view container (image label + floating toolbar)
        self._result_container = QFrame()
        self._result_container.setObjectName("ResultContainer")
        _MkThemeEngine.apply_style_sheet(self._result_container, """
            QFrame#ResultContainer {
                background: #0f172a;
                border-radius: 10px;
            }
        """)
        result_v = QVBoxLayout(self._result_container)
        result_v.setContentsMargins(0, 0, 0, 0)
        self.lbl_preview_canvas = QLabel()
        self.lbl_preview_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_canvas.setScaledContents(False)
        result_v.addWidget(self.lbl_preview_canvas, stretch=1)
        self._preview_stack.addWidget(self._result_container)  # index 1

        # Floating toolbar (top-right corner, child of result_container)
        self._floating_toolbar = FloatingToolbarWidget(self._result_container)
        self._floating_toolbar.downloadClicked.connect(self._save_prediction_image)
        self._floating_toolbar.replaceClicked.connect(self._choose_test_image)
        self._floating_toolbar.clearClicked.connect(self._clear_test_image)
        self._floating_toolbar.adjustSize()
        self._floating_toolbar.raise_()
        self._floating_toolbar.hide()  # only show when on page 1

        # Position toolbar after container is shown
        def _place_toolbar():
            tb = self._floating_toolbar
            tb.adjustSize()
            margin = 10
            tb.move(self._result_container.width() - tb.width() - margin, margin)
            tb.raise_()
            tb.show()
        self._place_toolbar = _place_toolbar

        lc_layout.addWidget(self._preview_stack, stretch=1)

        # Loading overlay (child of preview stack, covers whole stack)
        self.loading_overlay = LoadingOverlayWidget(parent=self._preview_stack)

        # ── Example cards row ────────────────────────────────────────
        self.lbl_example = QLabel("或尝试示例图片")
        _MkThemeEngine.apply_style_sheet(self.lbl_example, "color: #475569; font-size: 11px; font-weight: 600; margin-top: 4px;")
        lc_layout.addWidget(self.lbl_example)

        self._example_cards = []
        examples_row = QHBoxLayout()
        examples_row.setSpacing(10)

        # bus.jpg card
        self._card_bus = ExampleThumbCard(
            image_path=self._built_in_bus,
            title="bus.jpg"
        )
        self._card_bus.clicked.connect(self._on_example_clicked)
        examples_row.addWidget(self._card_bus)
        self._example_cards.append(self._card_bus)

        # zidane.jpg card
        self._card_zidane = ExampleThumbCard(
            image_path=self._built_in_zidane,
            title="zidane.jpg"
        )
        self._card_zidane.clicked.connect(self._on_example_clicked)
        examples_row.addWidget(self._card_zidane)
        self._example_cards.append(self._card_zidane)

        # Browse / upload card
        self._card_browse = ExampleThumbCard(is_browse_card=True)
        self._card_browse.clicked.connect(self._on_image_selected_and_predict)
        examples_row.addWidget(self._card_browse)
        self._example_cards.append(self._card_browse)

        examples_row.addStretch()

        # Quick-picks container (dataset val images, populated later)
        self.quick_picks_layout = QHBoxLayout()
        self.quick_picks_layout.setSpacing(8)
        self.lbl_samples_hint = QLabel("快速选择数据集样例:")
        _MkThemeEngine.apply_style_sheet(self.lbl_samples_hint, "color: #64748b; font-size: 11px;")
        self.quick_picks_layout.addWidget(self.lbl_samples_hint)
        self.quick_picks_container = QWidget()
        self.quick_picks_container.setLayout(self.quick_picks_layout)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(6)
        examples_col.addLayout(examples_row)
        examples_col.addWidget(self.quick_picks_container)
        lc_layout.addLayout(examples_col)

        # ── Sliders / param controls row ─────────────────────────────
        self._sliders_card = QFrame()
        self._sliders_card.setObjectName("SlidersCard")
        _MkThemeEngine.apply_style_sheet(self._sliders_card, """
            QFrame#SlidersCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 4px;
            }
        """)
        sl_layout = QHBoxLayout(self._sliders_card)
        sl_layout.setSpacing(20)
        sl_layout.setContentsMargins(12, 8, 12, 8)

        # Confidence slider
        v_conf = QVBoxLayout()
        v_conf.setSpacing(3)
        self.lbl_conf_val = QLabel("置信度 (Confidence)")
        _MkThemeEngine.apply_style_sheet(self.lbl_conf_val, "font-size: 11px; font-weight: 600; color: #475569;")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(25)
        self._lbl_conf_num = QLabel("0.25")
        _MkThemeEngine.apply_style_sheet(self._lbl_conf_num, "font-size: 11px; color: #2563eb; font-weight: 700;")
        self._lbl_conf_num.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.slider_conf.valueChanged.connect(
            lambda v: self._lbl_conf_num.setText(f"{v / 100.0:.2f}")
        )
        self.slider_conf.sliderReleased.connect(self._on_param_changed)
        v_conf.addWidget(self.lbl_conf_val)
        v_conf.addWidget(self.slider_conf)
        v_conf.addWidget(self._lbl_conf_num)
        sl_layout.addLayout(v_conf)

        # IoU slider
        v_iou = QVBoxLayout()
        v_iou.setSpacing(3)
        self.lbl_iou_val = QLabel("交并比 (IoU)")
        _MkThemeEngine.apply_style_sheet(self.lbl_iou_val, "font-size: 11px; font-weight: 600; color: #475569;")
        self.slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_iou.setRange(10, 95)
        self.slider_iou.setValue(70)
        self._lbl_iou_num = QLabel("0.70")
        _MkThemeEngine.apply_style_sheet(self._lbl_iou_num, "font-size: 11px; color: #2563eb; font-weight: 700;")
        self._lbl_iou_num.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.slider_iou.valueChanged.connect(
            lambda v: self._lbl_iou_num.setText(f"{v / 100.0:.2f}")
        )
        self.slider_iou.sliderReleased.connect(self._on_param_changed)
        v_iou.addWidget(self.lbl_iou_val)
        v_iou.addWidget(self.slider_iou)
        v_iou.addWidget(self._lbl_iou_num)
        sl_layout.addLayout(v_iou)

        # Image size combo
        v_sz = QVBoxLayout()
        v_sz.setSpacing(3)
        self.lbl_sz = QLabel("推理尺寸 (imgsz)")
        _MkThemeEngine.apply_style_sheet(self.lbl_sz, "font-size: 11px; font-weight: 600; color: #475569;")
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["640", "320", "1280"])
        self.combo_imgsz.currentIndexChanged.connect(self._on_param_changed)
        v_sz.addWidget(self.lbl_sz)
        v_sz.addWidget(self.combo_imgsz)
        sl_layout.addLayout(v_sz)

        # Run button
        self.btn_run_predict = MkButton("运行推理预测", btn_type="primary")
        self.btn_run_predict.setIcon(PhSparkle.regular(size=15))
        self.btn_run_predict.clicked.connect(self._run_prediction)
        sl_layout.addWidget(self.btn_run_predict)

        lc_layout.addWidget(self._sliders_card)
        layout.addWidget(left_col, stretch=6)

        # ── Right Column ─────────────────────────────────────────────
        right_col = QWidget()
        _MkThemeEngine.apply_style_sheet(right_col, "background: transparent;")
        rc_layout = QVBoxLayout(right_col)
        rc_layout.setContentsMargins(10, 0, 8, 0)
        rc_layout.setSpacing(0)

        # ─── 单一结果卡片（严格对标图二红框，纯单一卡片结构）──────────
        self._result_card = QFrame()
        self._result_card.setObjectName("ResultCard")
        _MkThemeEngine.apply_style_sheet(self._result_card, """
            QFrame#ResultCard {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self._result_card)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 16))
        self._result_card.setGraphicsEffect(shadow)

        result_card_layout = QVBoxLayout(self._result_card)
        result_card_layout.setContentsMargins(22, 20, 22, 20)
        result_card_layout.setSpacing(0)

        # 顶部：标题 + 总耗时徽章
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self._lbl_result_title = QLabel("推理结果")
        self._lbl_result_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        _MkThemeEngine.apply_style_sheet(self._lbl_result_title, "color: #111827; background: transparent;")
        top_row.addWidget(self._lbl_result_title)
        top_row.addStretch()
        self._lbl_total_ms = QLabel("-- ms")
        self._lbl_total_ms.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        _MkThemeEngine.apply_style_sheet(self._lbl_total_ms, """
            color: #374151;
            background: #f3f4f6;
            border-radius: 8px;
            padding: 3px 10px;
        """)
        top_row.addWidget(self._lbl_total_ms)
        result_card_layout.addLayout(top_row)

        # 副标题：检测数量 + 图像尺寸（紧随标题下方，图二规范）
        self._lbl_det_subtitle = QLabel("尚未运行推理")
        self._lbl_det_subtitle.setFont(QFont("Segoe UI", 10))
        _MkThemeEngine.apply_style_sheet(self._lbl_det_subtitle, "color: #6b7280; margin-top: 4px; margin-bottom: 14px; background: transparent;")
        result_card_layout.addWidget(self._lbl_det_subtitle)

        # 4格速度统计框（对标图二：数值在上、标签在下，轻量浅灰底框，无任何横线切割）
        self._speed_grid = QHBoxLayout()
        self._speed_grid.setContentsMargins(0, 0, 0, 0)
        self._speed_grid.setSpacing(6)
        speed_labels = [("前处理", "preprocess"), ("推理", "inference"), ("后处理", "postprocess"), ("端到端", "total")]
        self._speed_boxes = {}
        for label_cn, key in speed_labels:
            box = QFrame()
            _MkThemeEngine.apply_style_sheet(box, "QFrame { background: #f3f4f6; border-radius: 8px; border: none; }")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(6, 8, 6, 8)
            box_layout.setSpacing(2)
            # 时间数值在上（严格对标图二）
            lbl_val = QLabel("--")
            lbl_val.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            _MkThemeEngine.apply_style_sheet(lbl_val, "color: #111827; background: transparent;")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            # 指标标签在下（严格对标图二）
            lbl_name = QLabel(label_cn)
            lbl_name.setFont(QFont("Segoe UI", 9))
            _MkThemeEngine.apply_style_sheet(lbl_name, "color: #6b7280; background: transparent;")
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            box_layout.addWidget(lbl_val)
            box_layout.addWidget(lbl_name)
            self._speed_boxes[key] = lbl_val
            self._speed_grid.addWidget(box)
        result_card_layout.addLayout(self._speed_grid)

        # 识别目标小标题（图二规范：小标题直接紧接，无横向割裂分隔线）
        self._lbl_det_section = QLabel("识别目标")
        self._lbl_det_section.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        _MkThemeEngine.apply_style_sheet(self._lbl_det_section, "color: #374151; margin-top: 18px; margin-bottom: 8px; background: transparent;")
        result_card_layout.addWidget(self._lbl_det_section)

        # 检测列表容器（动态填充）
        self._det_list_container = QFrame()
        _MkThemeEngine.apply_style_sheet(self._det_list_container, "background: transparent; border: none;")
        self._det_list_layout = QVBoxLayout(self._det_list_container)
        self._det_list_layout.setContentsMargins(0, 0, 0, 0)
        self._det_list_layout.setSpacing(2)
        self._lbl_no_det = QLabel("暂无检测结果")
        _MkThemeEngine.apply_style_sheet(self._lbl_no_det, "color: #9ca3af; font-size: 11px; padding: 4px 0; background: transparent;")
        self._det_list_layout.addWidget(self._lbl_no_det)
        result_card_layout.addWidget(self._det_list_container)

        # 底部框架与版本信息（严格对标图二底部的 Ultralytics · PyTorch 提示，轻量非阻塞加载）
        u_mod = sys.modules.get("ultralytics")
        u_ver = getattr(u_mod, "__version__", "8.x") if u_mod else "8.x"
        t_mod = sys.modules.get("torch")
        if t_mod:
            dev_str = "+cuda" if getattr(t_mod, "cuda", None) and t_mod.cuda.is_available() else "+cpu"
            t_ver = f"PyTorch {t_mod.__version__}{dev_str}"
        else:
            t_ver = "PyTorch"
        self.lbl_predict_footer = QLabel(f"Ultralytics {u_ver} · {t_ver}")
        self.lbl_predict_footer.setFont(QFont("Segoe UI", 9))
        _MkThemeEngine.apply_style_sheet(self.lbl_predict_footer, "color: #9ca3af; margin-top: 20px; background: transparent;")
        result_card_layout.addWidget(self.lbl_predict_footer)

        rc_layout.addWidget(self._result_card)
        rc_layout.addStretch()
        layout.addWidget(right_col, stretch=4)




    def _choose_test_image(self):
        """Open file dialog to browse local images; triggers auto-prediction."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待预测图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self._on_image_selected_and_predict(path)

    def _on_example_clicked(self, path: str):
        """Called when a built-in example thumbnail is clicked."""
        # Highlight the selected card
        for card in self._example_cards:
            card.set_selected(card.image_path == path)
        self._on_image_selected_and_predict(path)

    def _on_image_selected_and_predict(self, path: str):
        """Load image into preview (page 1) and immediately trigger prediction."""
        if not path or not os.path.isfile(path):
            return
        self.current_test_image_path = path

        # Show result page with the raw input image first
        pix = QPixmap(path)
        if not pix.isNull():
            self.lbl_preview_canvas.setPixmap(
                pix.scaled(
                    self._preview_stack.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        self._preview_stack.setCurrentIndex(1)  # switch to result page
        # Place and show floating toolbar
        self._place_toolbar()

        # Auto-trigger prediction
        self._run_prediction()

    def _clear_test_image(self):
        """Reset to dropzone (page 0) and clear state."""
        self.current_test_image_path = ""
        self.last_predicted_image_rgb = None
        self.lbl_preview_canvas.clear()
        self._floating_toolbar.hide()
        # Deselect all example cards
        for card in self._example_cards:
            card.set_selected(False)
        self._preview_stack.setCurrentIndex(0)  # back to dropzone

        # 重置结果卡片数据（恢复未推理纯净状态）
        if hasattr(self, "_lbl_det_subtitle"):
            self._lbl_det_subtitle.setText("尚未运行推理")
        if hasattr(self, "_lbl_total_ms"):
            self._lbl_total_ms.setText("-- ms")
        if hasattr(self, "_speed_boxes"):
            for lbl in self._speed_boxes.values():
                lbl.setText("--")
        if hasattr(self, "_det_list_layout"):
            while self._det_list_layout.count():
                item = self._det_list_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
            lbl_no_det = QLabel("暂无检测结果")
            _MkThemeEngine.apply_style_sheet(lbl_no_det, "color: #9ca3af; font-size: 11px; padding: 4px 0; background: transparent;")
            self._det_list_layout.addWidget(lbl_no_det)

    def _on_param_changed(self):
        """Re-run prediction when a slider is released (if image already loaded)."""
        if self.current_test_image_path and os.path.isfile(self.current_test_image_path):
            self._run_prediction()

    def _run_prediction(self):
        if not self.current_test_image_path or not os.path.isfile(self.current_test_image_path):
            MkMessage.warning(self, "请先上传或选择一张待预测图片！")
            return

        weights_path = self._get_best_weights_path()
        if not weights_path or not os.path.isfile(weights_path):
            MkMessage.error(self, "未找到最佳模型权重 best.pt，请确认训练是否产生权重！")
            return

        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0
        imgsz = int(self.combo_imgsz.currentText())

        self.btn_run_predict.setEnabled(False)
        self.btn_run_predict.setText("正在推理计算...")

        # Show animated loading overlay
        self.loading_overlay.setGeometry(self._preview_stack.rect())
        self.loading_overlay.start()

        self.worker = InferenceWorkerThread(
            weights_path=weights_path,
            image_path=self.current_test_image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            parent=self
        )
        self.worker.result_ready.connect(self._on_prediction_done)
        self.worker.error_occurred.connect(self._on_prediction_error)
        self.worker.start()

    def _on_prediction_done(self, data: dict):
        # Stop loading overlay
        self.loading_overlay.stop()
        self.btn_run_predict.setEnabled(True)
        self.btn_run_predict.setText("运行推理预测")

        qimg = data.get("qimage")
        self.last_predicted_image_rgb = data.get("raw_rgb")
        if qimg and not qimg.isNull():
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self._preview_stack.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_preview_canvas.setPixmap(scaled)
        # Ensure we're on result page with toolbar visible
        self._preview_stack.setCurrentIndex(1)
        self._place_toolbar()

        # ── Populate speed stat boxes ──────────────────────────────────
        speed = data.get("speed", {})
        total_ms = speed.get("total", 0.0)
        for key, lbl in self._speed_boxes.items():
            val = speed.get(key, 0.0)
            lbl.setText(f"{val:.1f} ms")
        self._lbl_total_ms.setText(f"{total_ms:.0f} ms")

        # ── Populate detections ───────────────────────────────────────
        dets = data.get("detections", [])
        orig_shape = data.get("orig_shape", (0, 0))
        w, h = orig_shape if orig_shape else (0, 0)
        n = len(dets)
        subtitle = f"{n} 个目标检出"
        if w and h:
            subtitle += f" · {w}×{h}px"
        self._lbl_det_subtitle.setText(subtitle)

        # Clear old detection rows
        while self._det_list_layout.count():
            item = self._det_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        # Dot colors palette
        _dot_colors = [
            "#7c3aed", "#1d4ed8", "#0891b2", "#059669",
            "#d97706", "#dc2626", "#db2777", "#4f46e5",
        ]

        if not dets:
            no_det = QLabel("暂无检测结果")
            _MkThemeEngine.apply_style_sheet(no_det, "color: #9ca3af; font-size: 11px; padding: 4px 0; background: transparent;")
            self._det_list_layout.addWidget(no_det)
        else:
            for i, d in enumerate(dets):
                row = QHBoxLayout()
                row.setContentsMargins(0, 3, 0, 3)
                row.setSpacing(8)

                # Colored dot
                dot_color = _dot_colors[i % len(_dot_colors)]
                dot = QLabel("●")
                dot.setFont(QFont("Segoe UI", 9))
                _MkThemeEngine.apply_style_sheet(dot, f"color: {dot_color}; background: transparent;")
                dot.setFixedWidth(14)
                row.addWidget(dot)

                # Class name
                lbl_cls = QLabel(d["class"])
                lbl_cls.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
                _MkThemeEngine.apply_style_sheet(lbl_cls, "color: #111827; background: transparent;")
                row.addWidget(lbl_cls)
                row.addStretch()

                # Confidence
                lbl_conf = QLabel(f"{d['confidence'] * 100:.1f}%")
                lbl_conf.setFont(QFont("Segoe UI", 11))
                _MkThemeEngine.apply_style_sheet(lbl_conf, "color: #6b7280; background: transparent;")
                row.addWidget(lbl_conf)

                row_widget = QWidget()
                _MkThemeEngine.apply_style_sheet(row_widget, "background: transparent;")
                row_widget.setLayout(row)
                self._det_list_layout.addWidget(row_widget)

        MkMessage.success(self, f"推理完成！检测到 {n} 个目标。")

    def _on_prediction_error(self, err: str):
        self.loading_overlay.stop()
        self.btn_run_predict.setEnabled(True)
        self.btn_run_predict.setText("运行推理预测")
        MkMessage.error(self, f"推理失败: {err}")

    def _save_prediction_image(self):
        if self.last_predicted_image_rgb is None:
            MkMessage.warning(self, "暂无推理结果图片可保存！")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存预测结果", "prediction_result.png", "Images (*.png *.jpg)"
        )
        if path:
            import cv2
            bgr = cv2.cvtColor(self.last_predicted_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            MkMessage.success(self, f"图片已保存至: {path}")

    # ─────────────────────────────────────────────────────────
    # ── Tab 3: Export (模型导出)
    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    # ── Tab 3: Export (模型导出管理 - 对标 Ultralytics HUB 图 1-5)
    # ─────────────────────────────────────────────────────────
    def _setup_export_tab(self):
        c_layout = QVBoxLayout(self.tab_export)
        c_layout.setContentsMargins(0, 8, 0, 16)
        c_layout.setSpacing(14)

        # 1. Header (Fig 1)
        top_box = QVBoxLayout()
        top_box.setSpacing(4)
        self.lbl_export_h1 = QLabel("导出格式 (Export Formats)")
        self.lbl_export_h1.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        _MkThemeEngine.apply_style_sheet(self.lbl_export_h1, "color: #111827; background: transparent;")
        top_box.addWidget(self.lbl_export_h1)

        self.lbl_export_desc = QLabel(
            "将 PyTorch 模型转换为不同工业端部署格式。您可以为每种格式导出具有不同设置（如不同图像尺寸、量化精度）的多个版本。"
        )
        self.lbl_export_desc.setFont(QFont("Segoe UI", 10))
        _MkThemeEngine.apply_style_sheet(self.lbl_export_desc, "color: #64748b; background: transparent;")
        top_box.addWidget(self.lbl_export_desc)
        c_layout.addLayout(top_box)

        # 2. 8 种核心格式卡片列表
        self._format_cards = {}
        formats_order = ["onnx", "engine", "openvino", "coreml", "rknn", "hailo", "ncnn", "litert"]

        for fmt_key in formats_order:
            card = ExportFormatCard(fmt_key, parent=self.tab_export)
            card.export_requested.connect(self._on_format_export_requested)
            card.record_download.connect(self._on_record_download)
            card.record_delete.connect(self._on_record_delete)
            card.record_open_folder.connect(self._on_record_open_folder)
            card.cancel_export_requested.connect(self._on_cancel_export)
            self._format_cards[fmt_key] = card
            c_layout.addWidget(card)

        c_layout.addStretch()

        self._load_export_history()

    def _get_export_history_file(self) -> Optional[str]:
        if not self.current_task:
            return None
        save_dir = self.current_task.get("save_dir", "")
        if save_dir and os.path.isdir(save_dir):
            return os.path.join(save_dir, "export_history.json")
        return None

    def _load_export_history(self):
        history = {}
        hist_file = self._get_export_history_file()
        if hist_file and os.path.isfile(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = {}

        self._discover_local_exported_files(history)

        for fmt_key, card in self._format_cards.items():
            records = history.get(fmt_key, [])
            card.set_records(records)

    def _discover_local_exported_files(self, history: dict):
        if not self.current_task:
            return
        save_dir = self.current_task.get("save_dir", "")
        if not save_dir or not os.path.isdir(save_dir):
            return

        weights_dir = os.path.join(save_dir, "weights")
        dirs_to_check = [save_dir]
        if os.path.isdir(weights_dir):
            dirs_to_check.append(weights_dir)

        ext_to_fmt = {
            ".onnx": "onnx",
            ".engine": "engine",
            ".mlpackage": "coreml",
            ".tflite": "litert",
        }

        updated = False
        for d in dirs_to_check:
            try:
                for item in os.listdir(d):
                    p = os.path.join(d, item)
                    ext = os.path.splitext(item)[1].lower()
                    fmt = ext_to_fmt.get(ext)
                    if not fmt:
                        if item.endswith("_openvino_model"):
                            fmt = "openvino"
                        elif item.endswith("_rknn_model"):
                            fmt = "rknn"
                        elif item.endswith("_ncnn_model"):
                            fmt = "ncnn"
                        elif "hailo" in item.lower() and ext == ".onnx":
                            fmt = "hailo"

                    if fmt and fmt in self._format_cards:
                        existing_paths = [r.get("path") for r in history.get(fmt, [])]
                        if p not in existing_paths:
                            size_mb = os.path.getsize(p) / (1024 * 1024) if os.path.isfile(p) else 0.0
                            mtime = os.path.getmtime(p)
                            record = {
                                "id": f"{fmt}_{int(mtime)}",
                                "format": fmt,
                                "path": p,
                                "summary": f"imgsz=640, file={item}",
                                "size": f"{size_mb:.1f} MB" if size_mb > 0 else "已生成",
                                "duration": "--",
                                "created_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                                "params": {"format": fmt}
                            }
                            if fmt not in history:
                                history[fmt] = []
                            history[fmt].append(record)
                            updated = True
            except Exception:
                pass

        if updated:
            self._save_export_history(history)

    def _save_export_history(self, history: dict):
        hist_file = self._get_export_history_file()
        if hist_file:
            try:
                with open(hist_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    def _on_format_export_requested(self, fmt_key: str):
        weights_path = self._get_best_weights_path()
        if not weights_path or not os.path.isfile(weights_path):
            MkMessage.error(self, "未找到可导出的 best.pt 模型权重文件，请先完成训练！")
            return

        dlg = BaseExportDialog(fmt_key, parent=self)
        dlg.export_submitted.connect(self._start_export)
        dlg.exec()

    def _start_export(self, fmt_key: str, params: dict):
        weights_path = self._get_best_weights_path()
        if not weights_path or not os.path.isfile(weights_path):
            MkMessage.error(self, "未找到模型权重文件！")
            return

        spec = EXPORT_SPECS.get(fmt_key, {})
        fmt_name = spec.get("name", fmt_key.upper())
        MkMessage.info(self, f"已启动 {fmt_name} 模型导出任务，请稍候...")

        export_id = f"exp_{fmt_key}_{int(time.time()*1000)}"
        card = self._format_cards.get(fmt_key)
        if card:
            card.add_pending_record(export_id, params)

        self.export_worker = ExportWorkerThread(
            weights_path=weights_path,
            params=params,
            parent=self
        )
        self.export_worker.finished.connect(
            lambda success, path_or_err, meta, fk=fmt_key, eid=export_id: self._on_export_finished(fk, eid, success, path_or_err, meta)
        )
        self.export_worker.start()

    def _on_cancel_export(self, fmt_key: str, record: dict):
        eid = record.get("id", "")
        if hasattr(self, "export_worker") and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait(1500)

        card = self._format_cards.get(fmt_key)
        if card:
            card.remove_pending_record(eid)
        MkMessage.info(self, "已取消模型导出任务")

    def _on_export_finished(self, fmt_key: str, export_id: str, success: bool, path_or_err: str, meta: dict):
        card = self._format_cards.get(fmt_key)
        spec = EXPORT_SPECS.get(fmt_key, {})
        fmt_name = spec.get("name", fmt_key.upper())

        if success:
            meta["id"] = export_id
            hist_file = self._get_export_history_file()
            history = {}
            if hist_file and os.path.isfile(hist_file):
                try:
                    with open(hist_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = {}

            if fmt_key not in history:
                history[fmt_key] = []
            history[fmt_key].insert(0, meta)
            self._save_export_history(history)

            if card:
                card.update_record_finished(export_id, meta)

            MkMessage.success(self, f"{fmt_name} 模型导出成功！已保存至: {path_or_err}")
        else:
            if card:
                card.remove_pending_record(export_id)
            MkMessage.error(self, f"{fmt_name} 模型导出失败: {path_or_err}")

    def _on_record_download(self, record: dict):
        src_path = record.get("path", "")
        if not src_path or not os.path.exists(src_path):
            MkMessage.warning(self, "该导出的模型文件不存在或已被移动！")
            return

        base_name = os.path.basename(src_path)
        dest_path, _ = QFileDialog.getSaveFileName(self, "另存为 / 下载模型文件", base_name, "All Files (*.*)")
        if dest_path:
            import shutil
            try:
                if os.path.isfile(src_path):
                    shutil.copyfile(src_path, dest_path)
                elif os.path.isdir(src_path):
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
                MkMessage.success(self, f"模型已成功保存至: {dest_path}")
            except Exception as e:
                MkMessage.error(self, f"保存失败: {e}")

    def _on_record_open_folder(self, record: dict):
        path = record.get("path", "")
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if os.path.isdir(folder):
            try:
                os.startfile(folder)
            except Exception:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        else:
            MkMessage.warning(self, "目标文件夹不存在！")

    def _on_record_delete(self, record: dict):
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除此导出记录及对应文件吗？\n{record.get('path', '')}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            fmt_key = record.get("format", "")
            path = record.get("path", "")
            rec_id = record.get("id", "")

            try:
                if path and os.path.isfile(path):
                    os.remove(path)
                elif path and os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
            except Exception:
                pass

            hist_file = self._get_export_history_file()
            if hist_file and os.path.isfile(hist_file):
                try:
                    with open(hist_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                    if fmt_key in history:
                        history[fmt_key] = [r for r in history[fmt_key] if r.get("id") != rec_id and r.get("path") != path]
                        self._save_export_history(history)
                        if fmt_key in self._format_cards:
                            self._format_cards[fmt_key].set_records(history[fmt_key])
                except Exception:
                    pass

            MkMessage.success(self, "已删除该导出记录")

    # ─────────────────────────────────────────────────────────
    # ── Load Task Logic ──
    # ─────────────────────────────────────────────────────────
    def load_task(self, task_id: int):
        """Populate experiment view with given task details, metrics, and curves."""
        self.current_task_id = task_id
        if self.db:
            self.current_task = self.db.get_task_by_id(task_id)

        if not self.current_task:
            return

        task = self.current_task
        task_name = task.get("task_name", f"exp-{task_id}")
        model_name = task.get("model_name", "yolo26n-seg.pt")
        status = task.get("status", "COMPLETED")
        save_dir = task.get("save_dir", "")

        # 1. Update Header immediately (0ms instant response)
        self.lbl_task_title.setText(task_name)
        self.lbl_task_type_tag.setText(model_name)
        st_color = "#15803d" if status in ("SUCCESS", "COMPLETED") else "#b91c1c"
        st_bg = "#dcfce7" if status in ("SUCCESS", "COMPLETED") else "#fee2e2"
        self.lbl_status_badge.setText(status)
        _MkThemeEngine.apply_style_sheet(self.lbl_status_badge, f"background: {st_bg}; color: {st_color}; font-weight: bold; padding: 3px 8px; border-radius: 4px; font-size: 11px;")

        # 2. Enter Skeleton Shimmer Loading state on all cards
        self.set_loading(True)

        # 3. Background thread: read results.csv and disk files without blocking GUI
        def _fetch():
            import time
            time.sleep(0.12)  # 平滑微过渡（120ms），保证人眼感知到优雅科技流光过渡，杜绝瞬间白屏闪烁
            metrics_df = None
            if save_dir and os.path.isdir(save_dir):
                csv_path = os.path.join(save_dir, "results.csv")
                if os.path.isfile(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df.columns = [c.strip() for c in df.columns]
                        metrics_df = df
                    except Exception:
                        pass
            return metrics_df

        # 4. Main GUI thread callback: render charts, tables and fade out skeleton
        def _on_loaded(metrics_df):
            self.current_metrics_df = metrics_df
            self._update_kpi_cards(task)
            self._update_overview_tables(task)
            self._update_all_charts()
            self.apply_theme()
            self._load_task_console(save_dir)
            self._load_dataset_sample_picks(task)
            self._load_export_history()
            self.set_loading(False)

        self.run_async(_fetch, on_success=_on_loaded, auto_loading=False)

    def apply_theme(self, *args):
        if self.isVisible():
            self.set_theme(ThemeEngine.is_dark())

    def showEvent(self, event):
        super().showEvent(event)
        self.set_theme(ThemeEngine.is_dark())

    def _update_kpi_cards(self, task: dict):
        map50_val = 0.0
        map50_95_val = 0.0
        prec_val = 0.0
        rec_val = 0.0

        save_dir = task.get("save_dir", "") if task else ""
        final_m = {}
        if save_dir:
            try:
                from TrainingPlatform.core.metrics_parser import parse_final_experiment_metrics
                final_m = parse_final_experiment_metrics(save_dir)
            except Exception:
                pass

        if final_m.get("map50", 0.0) > 0.0:
            map50_val = final_m["map50"]
            map50_95_val = final_m["map50_95"]
            prec_val = final_m["precision"]
            rec_val = final_m["recall"]
        elif self.current_metrics_df is not None and not self.current_metrics_df.empty:
            df = self.current_metrics_df
            for col in df.columns:
                if "metrics/mAP50(B)" in col or "mAP50(B)" in col:
                    map50_val = float(df[col].iloc[-1])
                elif "metrics/mAP50-95(B)" in col or "mAP50-95(B)" in col:
                    map50_95_val = float(df[col].iloc[-1])
                elif "metrics/precision(B)" in col or "precision(B)" in col:
                    prec_val = float(df[col].iloc[-1])
                elif "metrics/recall(B)" in col or "recall(B)" in col:
                    rec_val = float(df[col].iloc[-1])
        else:
            map50_val = task.get("best_map50", 0.0) or 0.0
            map50_95_val = task.get("best_map50_95", 0.0) or 0.0

        self.card_map50._lbl_val.setText(f"{map50_val * 100:.1f}%" if map50_val > 0 else "-")
        self.card_map50_95._lbl_val.setText(f"{map50_95_val * 100:.1f}%" if map50_95_val > 0 else "-")
        self.card_precision._lbl_val.setText(f"{prec_val * 100:.1f}%" if prec_val > 0 else "-")
        self.card_recall._lbl_val.setText(f"{rec_val * 100:.1f}%" if rec_val > 0 else "-")

    def _update_overview_tables(self, task: dict):
        save_dir = task.get("save_dir", "")

        # 1. Run Information Card (Image 1: Modern card with circled fields and real CLI command)
        if hasattr(self, "card_run_info"):
            self.card_run_info.load_task_info(task, save_dir)

        # 2. Hyperparameters — feed into MkProTable
        if hasattr(self, "card_training_config"):
            self._load_hyperparameters_to_table(save_dir, task)

    # ── Hyperparameter dictionary-to-table descriptions ──
    _PARAM_TIPS = {
        "task": "计算机视觉任务类型 (detect/segment/pose/classify)",
        "mode": "执行模式 (train 训练 / val 验证 / predict 预测)",
        "model": "基础模型架构权重路径或名称",
        "data": "训练数据集 YAML 配置文件路径",
        "epochs": "训练总轮次 (Epochs)",
        "batch": "训练批次大小 (Batch Size，-1 为自动显存优化)",
        "imgsz": "输入图像尺寸分辨率 (宽 x 高)",
        "device": "计算设备编号 (如 0 表示 GPU 0，cpu 表示使用 CPU)",
        "workers": "数据加载 DataLoader 工作进程数",
        "optimizer": "优化器算法 (auto/SGD/Adam/AdamW 等)",
        "lr0": "初始学习率 (Initial Learning Rate)",
        "lrf": "最终学习率衰减比例 (Final Learning Rate)",
        "momentum": "SGD 动量或 Adam beta1 参数",
        "weight_decay": "权重衰减系数 (L2 正则化防止过拟合)",
        "warmup_epochs": "预热学习率轮数 (Warmup Epochs)",
        "box": "边界框定位损失权重 (Box loss gain)",
        "cls": "类别分类损失权重 (Classification loss gain)",
        "dfl": "分布焦点损失权重 (Distribution Focal Loss)",
        "close_mosaic": "训练结束前最后 N 个 Epoch 禁用 Mosaic 增强以稳定收敛",
        "amp": "自动混合精度训练 (Automatic Mixed Precision, 加速并节省显存)",
        "fraction": "用于训练的数据集比例分数 (1.0 表示使用全量数据)",
        "patience": "早停轮数阈值 (无精度提升时提前终止训练)",
        "save": "是否保存检查点权重文件 (best.pt, last.pt)",
        "save_period": "权重周期性保存间隔 (-1 表示仅保留最优与最终权重)",
        "augment": "是否启用数据增强策略",
        "auto_augment": "自动数据增强策略 (randaugment/autoaugment)",
        "hsv_h": "HSV 色相微调增强比例",
        "hsv_s": "HSV 饱和度增强比例",
        "hsv_v": "HSV 明度增强比例",
        "degrees": "图像随机旋转角度范围",
        "translate": "图像水平与垂直随机平移比例",
        "scale": "图像随机缩放比例范围",
        "shear": "图像剪切变换角度",
        "perspective": "随机透视变换比例",
        "flipud": "图像上下随机翻转概率",
        "fliplr": "图像左右随机翻转概率",
        "mosaic": "Mosaic 4 图马赛克拼接增强概率",
        "mixup": "MixUp 图像混合增强概率",
        "copy_paste": "Copy-Paste 实例分割遮罩复制粘贴增强概率",
        "overlap_mask": "训练时分割遮罩是否允许相互重叠",
        "mask_ratio": "分割遮罩降采样下采样率 (通常为 4)",
    }

    def _load_hyperparameters_to_table(self, save_dir: str, task: dict):
        """Read YOLO args.yaml or fallback configs and feed into MkProTable."""
        config_dict = {}
        if save_dir and os.path.isdir(save_dir):
            for fname in ("args.yaml", "hyp.yaml", "train_config.json", "config.json"):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        if fname.endswith((".yaml", ".yml")):
                            import yaml
                            with open(fpath, "r", encoding="utf-8") as f:
                                config_dict = yaml.safe_load(f)
                        elif fname.endswith(".json"):
                            with open(fpath, "r", encoding="utf-8") as f:
                                config_dict = json.load(f)
                        if config_dict:
                            break
                    except Exception:
                        pass

        if not config_dict and task:
            for k in ("model_name", "epochs", "batch", "imgsz", "device", "optimizer", "lr0", "lrf"):
                if k in task:
                    config_dict[k] = task[k]

        self._hyper_raw_config = config_dict or {}

        # Build MkProTable row dicts
        rows = []
        current_flat = {}
        for k, v in (config_dict or {}).items():
            if v is None:
                val_str = "None"
            elif isinstance(v, bool):
                val_str = "True" if v else "False"
            else:
                val_str = str(v)
            current_flat[str(k)] = val_str
            tip = self._PARAM_TIPS.get(str(k), "")
            rows.append({"param": str(k), "value": val_str, "desc": tip})

        self._hyper_current_config = current_flat
        self.card_training_config.set_data(rows)



    def _update_all_charts(self):
        """Feed current metrics dataframe into thesis chart cards."""
        task_name = self.current_task.get("task_name", "exp") if self.current_task else "exp"

        def get_series(col_pattern: str):
            if self.current_metrics_df is None or self.current_metrics_df.empty:
                return {}
            df = self.current_metrics_df
            matched_col = None
            for c in df.columns:
                if col_pattern in c:
                    matched_col = c
                    break
            if not matched_col:
                return {}
            epochs = df["epoch"].tolist() if "epoch" in df.columns else list(range(1, len(df) + 1))
            values = df[matched_col].tolist()
            return {
                task_name: {
                    "epochs": epochs,
                    "values": values,
                    "color": ACADEMIC_COLORS[0]
                }
            }

        self.chart_precision.update_series(get_series("metrics/precision(B)"))
        self.chart_recall.update_series(get_series("metrics/recall(B)"))
        self.chart_map50.update_series(get_series("metrics/mAP50(B)"))
        self.chart_map50_95.update_series(get_series("metrics/mAP50-95(B)"))

        self.chart_box_loss.update_series(get_series("train/box_loss"))
        self.chart_seg_loss.update_series(get_series("train/seg_loss"))
        self.chart_cls_loss.update_series(get_series("train/cls_loss"))
        self.chart_val_loss.update_series(get_series("val/box_loss"))

    def _apply_console_style(self):
        pass

    def append_terminal_line(self, text: str):
        if hasattr(self, "train_console") and self.train_console:
            is_dark = self.is_dark if hasattr(self, "is_dark") else ThemeEngine.is_dark()
            append_terminal_line_to_console(self.train_console, text, is_dark=is_dark)

    def _load_task_console(self, save_dir: str):
        self.train_console.clear()
        if hasattr(self.train_console, "_counts"):
            self.train_console._counts = {"info": 0, "success": 0, "warning": 0, "error": 0}
        if hasattr(self.train_console, "_update_badges"):
            self.train_console._update_badges()
        self._apply_console_style()
        if save_dir and os.path.isdir(save_dir):
            log_path = os.path.join(save_dir, "train.log")
            if os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        for l in lines[-1500:]:
                            self.append_terminal_line(l.rstrip("\r\n"))
                except Exception:
                    pass

    def _load_dataset_sample_picks(self, task: dict):
        """Find sample images from dataset val/images directory."""
        # Clear previous pick buttons
        while self.quick_picks_layout.count() > 1:
            item = self.quick_picks_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        dataset_yaml = task.get("dataset_yaml", "")
        if not dataset_yaml or not os.path.isfile(dataset_yaml):
            return

        try:
            import yaml
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f)
            val_dir = ydata.get("val", "")
            if val_dir and not os.path.isabs(val_dir):
                val_dir = os.path.normpath(os.path.join(os.path.dirname(dataset_yaml), val_dir))

            if os.path.isdir(val_dir):
                imgs = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                for p in imgs[:4]:
                    btn = QToolButton()
                    btn.setText(os.path.basename(p)[:14])
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    _MkThemeEngine.apply_style_sheet(btn, """
                        QToolButton {
                            background: #e2e8f0;
                            border: 1px solid #cbd5e1;
                            border-radius: 4px;
                            padding: 3px 8px;
                            font-size: 11px;
                        }
                        QToolButton:hover {
                            background: #2563eb;
                            color: #ffffff;
                        }
                    """)
                    btn.clicked.connect(lambda _, path=p: self._set_current_image(path))
                    self.quick_picks_layout.addWidget(btn)
        except Exception:
            pass

    def _get_best_weights_path(self) -> Optional[str]:
        """Find model weights with multiple fallback options."""
        workspace_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        candidates = []

        if self.current_task:
            save_dir = self.current_task.get("save_dir", "")
            if save_dir:
                candidates.extend([
                    os.path.join(save_dir, "weights", "best.pt"),
                    os.path.join(save_dir, "best.pt"),
                    os.path.join(save_dir, "weights", "last.pt"),
                    os.path.join(save_dir, "last.pt"),
                ])

        # Workspace-level fallbacks
        candidates.extend([
            os.path.join(workspace_root, "runs", "segment", "exp-1", "weights", "best.pt"),
            os.path.join(workspace_root, "runs", "segment", "exp-1", "weights", "last.pt"),
            os.path.join(workspace_root, "yolo26n-seg.pt"),
            os.path.join(workspace_root, "weights", "yolo26n.pt"),
        ])

        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _open_task_folder(self):
        if not self.current_task:
            return
        save_dir = self.current_task.get("save_dir", "")
        if save_dir and os.path.isdir(save_dir):
            import subprocess
            if os.name == "nt":
                os.startfile(save_dir)
            else:
                subprocess.Popen(["xdg-open", save_dir])
        else:
            MkMessage.warning(self, "该任务的存放目录尚不存在！")

    def set_theme(self, is_dark: bool):
        version = (ThemeEngine.theme_version(), is_dark)
        if getattr(self, "_applied_theme_version", None) == version:
            return
        self._applied_theme_version = version
        self.is_dark = is_dark

        # Top Header Elements
        if hasattr(self, "lbl_task_title"):
            _MkThemeEngine.apply_style_sheet(self.lbl_task_title, "color: #f1f5f9; background: transparent;" if is_dark else "color: #0f172a; background: transparent;")
        if hasattr(self, "lbl_task_type_tag"):
            _MkThemeEngine.apply_style_sheet(self.lbl_task_type_tag, """
                background: #1e3a5f; color: #60a5fa; font-weight: 600;
                padding: 3px 8px; border-radius: 4px; font-size: 11px;
                border: 1px solid #2563eb;
            """ if is_dark else """
                background: #eff6ff; color: #2563eb; font-weight: 600;
                padding: 3px 8px; border-radius: 4px; font-size: 11px;
                border: 1px solid #dbeafe;
            """)
        if hasattr(self, "lbl_status_badge"):
            _MkThemeEngine.apply_style_sheet(self.lbl_status_badge, """
                background: #064e3b; color: #34d399; font-weight: 700;
                padding: 3px 8px; border-radius: 4px; font-size: 11px;
            """ if is_dark else """
                background: #dcfce7; color: #15803d; font-weight: 700;
                padding: 3px 8px; border-radius: 4px; font-size: 11px;
            """)
        if hasattr(self, "lbl_updated_time"):
            _MkThemeEngine.apply_style_sheet(self.lbl_updated_time, "color: #94a3b8; font-size: 12px; background: transparent;")

        # Segmented Tabs
        if hasattr(self, "primary_nav") and self.primary_nav and hasattr(self.primary_nav, "set_theme"):
            self.primary_nav.set_theme(is_dark)
        if hasattr(self, "train_sub_nav") and self.train_sub_nav and hasattr(self.train_sub_nav, "set_theme"):
            self.train_sub_nav.set_theme(is_dark)

        # Update KPI cards
        for card in (getattr(self, "card_map50", None), getattr(self, "card_map50_95", None),
                     getattr(self, "card_precision", None), getattr(self, "card_recall", None)):
            if card and hasattr(card, "set_theme"):
                card.set_theme(is_dark)

        # Update run info card
        if hasattr(self, "card_run_info") and hasattr(self.card_run_info, "set_theme"):
            self.card_run_info.set_theme(is_dark)

        # MkProTable auto-handles theme via ThemeEngine signal — no manual set_theme needed

        # Update thesis chart cards
        for chart in (getattr(self, "chart_precision", None), getattr(self, "chart_recall", None),
                      getattr(self, "chart_map50", None), getattr(self, "chart_map50_95", None),
                      getattr(self, "chart_box_loss", None), getattr(self, "chart_seg_loss", None),
                      getattr(self, "chart_cls_loss", None), getattr(self, "chart_val_loss", None)):
            if chart and hasattr(chart, "set_theme"):
                chart.set_theme(is_dark)

        # Update Export tab headers & format cards
        if hasattr(self, "lbl_export_h1"):
            _MkThemeEngine.apply_style_sheet(self.lbl_export_h1, "color: #f1f5f9; background: transparent;" if is_dark else "color: #111827; background: transparent;")
        if hasattr(self, "lbl_export_desc"):
            _MkThemeEngine.apply_style_sheet(self.lbl_export_desc, "color: #a1a1aa; background: transparent;" if is_dark else "color: #64748b; background: transparent;")
        if hasattr(self, "_format_cards"):
            for card in self._format_cards.values():
                if hasattr(card, "set_theme"):
                    card.set_theme(is_dark)

        # Update Predict tab
        if hasattr(self, "lbl_predict_in"):
            _MkThemeEngine.apply_style_sheet(self.lbl_predict_in, "color: #f1f5f9; background: transparent;" if is_dark else "color: #0f172a; background: transparent;")
        if hasattr(self, "lbl_predict_sub"):
            _MkThemeEngine.apply_style_sheet(self.lbl_predict_sub, "color: #94a3b8; font-size: 11px; background: transparent;" if is_dark else "color: #64748b; font-size: 11px; background: transparent;")
        if hasattr(self, "lbl_example"):
            _MkThemeEngine.apply_style_sheet(self.lbl_example, "color: #94a3b8; font-size: 11px; font-weight: 600; margin-top: 4px; background: transparent;" if is_dark else "color: #475569; font-size: 11px; font-weight: 600; margin-top: 4px; background: transparent;")
        if hasattr(self, "lbl_samples_hint"):
            _MkThemeEngine.apply_style_sheet(self.lbl_samples_hint, "color: #94a3b8; font-size: 11px; background: transparent;" if is_dark else "color: #64748b; font-size: 11px; background: transparent;")
        if hasattr(self, "lbl_conf_val"):
            _MkThemeEngine.apply_style_sheet(self.lbl_conf_val, "font-size: 11px; font-weight: 600; color: #a1a1aa; background: transparent;" if is_dark else "font-size: 11px; font-weight: 600; color: #475569; background: transparent;")
        if hasattr(self, "lbl_iou_val"):
            _MkThemeEngine.apply_style_sheet(self.lbl_iou_val, "font-size: 11px; font-weight: 600; color: #a1a1aa; background: transparent;" if is_dark else "font-size: 11px; font-weight: 600; color: #475569; background: transparent;")
        if hasattr(self, "lbl_sz"):
            _MkThemeEngine.apply_style_sheet(self.lbl_sz, "font-size: 11px; font-weight: 600; color: #a1a1aa; background: transparent;" if is_dark else "font-size: 11px; font-weight: 600; color: #475569; background: transparent;")
        if hasattr(self, "combo_imgsz"):
            _MkThemeEngine.apply_style_sheet(self.combo_imgsz, f"""
                QComboBox {{
                    background: {'#1e293b' if is_dark else '#ffffff'};
                    color: {'#f1f5f9' if is_dark else '#111827'};
                    border: 1px solid {'#334155' if is_dark else '#cbd5e1'};
                    border-radius: 6px;
                    padding: 4px 8px;
                }}
            """)
        if hasattr(self, "_sliders_card"):
            _MkThemeEngine.apply_style_sheet(self._sliders_card, f"""
                QFrame#SlidersCard {{
                    background: {'#18181b' if is_dark else '#f8fafc'};
                    border: 1px solid {'#27272a' if is_dark else '#e2e8f0'};
                    border-radius: 10px;
                    padding: 4px;
                }}
            """)
        if hasattr(self, "_result_card"):
            _MkThemeEngine.apply_style_sheet(self._result_card, f"""
                QFrame#ResultCard {{
                    background: {'#18181b' if is_dark else '#ffffff'};
                    border: 1px solid {'#27272a' if is_dark else '#e5e7eb'};
                    border-radius: 16px;
                }}
            """)
        if hasattr(self, "_lbl_result_title"):
            _MkThemeEngine.apply_style_sheet(self._lbl_result_title, f"color: {'#f1f5f9' if is_dark else '#111827'}; background: transparent;")
        if hasattr(self, "_lbl_total_ms"):
            _MkThemeEngine.apply_style_sheet(self._lbl_total_ms, f"""
                color: {'#d4d4d8' if is_dark else '#374151'};
                background: {'#27272a' if is_dark else '#f3f4f6'};
                border-radius: 8px;
                padding: 3px 10px;
            """)
        if hasattr(self, "_lbl_det_subtitle"):
            _MkThemeEngine.apply_style_sheet(self._lbl_det_subtitle, f"color: {'#9ca3af' if is_dark else '#6b7280'}; margin-top: 4px; margin-bottom: 14px; background: transparent;")
        if hasattr(self, "_lbl_det_section"):
            _MkThemeEngine.apply_style_sheet(self._lbl_det_section, f"color: {'#d4d4d8' if is_dark else '#374151'}; margin-top: 18px; margin-bottom: 8px; background: transparent;")
        if hasattr(self, "_speed_boxes"):
            for key, val_lbl in self._speed_boxes.items():
                box = val_lbl.parent()
                if box and isinstance(box, QFrame):
                    _MkThemeEngine.apply_style_sheet(box, f"QFrame {{ background: {'#27272a' if is_dark else '#f3f4f6'}; border-radius: 8px; border: none; }}")
                _MkThemeEngine.apply_style_sheet(val_lbl, f"color: {'#f1f5f9' if is_dark else '#111827'}; background: transparent;")

        if hasattr(self, "_dropzone") and hasattr(self._dropzone, "set_theme"):
            self._dropzone.set_theme(is_dark)
        if hasattr(self, "loading_overlay") and hasattr(self.loading_overlay, "set_theme"):
            self.loading_overlay.set_theme(is_dark)
        if hasattr(self, "_floating_toolbar") and hasattr(self._floating_toolbar, "set_theme"):
            self._floating_toolbar.set_theme(is_dark)
        if hasattr(self, "_example_cards"):
            for card in self._example_cards:
                if hasattr(card, "set_theme"):
                    card.set_theme(is_dark)

