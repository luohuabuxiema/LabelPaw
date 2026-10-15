# -*- coding: utf-8 -*-
"""
New Training Task Configuration Dialog for AI Vision Platform
Styled strictly benchmarked against Ultralytics "Train New Model" Dialog:
- Frameless floating modal card with rounded corners (14px) & soft drop shadow (28px blur)
- Custom Top Header: "创建训练任务" + task badge + subtitle + top-right close "x" button
- Header draggable to move dialog
- Base Model & Dataset 2-column side-by-side layout (Pure Chinese UI)
- 4 Core Hyperparameters (训练轮数 ⓘ, 批次大小 ⓘ, 图像尺寸 ⓘ, 任务名称 ⓘ) in a clean horizontal row
- Comprehensive Collapsible Advanced Settings in a Unified Continuous View (No tabs):
  1. 优化器与学习率 (Optimizer, lr0, lrf, momentum, weight_decay, warmup, cos_lr)
  2. 训练策略与加速控制 (patience, close_mosaic, save_period, seed, amp, deterministic, pretrained, compile, rect, single_cls, etc.)
  3. 损失权重与检测头 (box, cls, dfl, cls_pw, max_det)
  4. 图像数据增强超参数 (mosaic, mixup, cutmix, bgr, hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, fliplr, flipud)
- One-click "恢复默认参数" (Reset to Defaults)
- Local Training Device & Hardware summary card (zero emoji, vector icons only)
- Sleek 6px floating scrollbar with zero arrow buttons (no native Windows scrollbar)
- Bottom action buttons (取消 / 开始训练)
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
import os
import yaml
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                               QWidget, QFileDialog, QScrollArea, QFrame,
                               QPushButton, QGraphicsDropShadowEffect, QComboBox, QApplication,
                               QPlainTextEdit, QToolButton, QStackedWidget)
from PySide6.QtGui import (QColor, QPainter, QPainterPath, QPolygonF, QBrush, QPen, QCursor,
                           QFont, QSyntaxHighlighter, QTextCharFormat, QLinearGradient, QPixmap)
from PySide6.QtCore import (Qt, Signal, QPoint, QPointF, QRect, QRectF, QPropertyAnimation, 
                            QVariantAnimation, QEasingCurve, QEvent, QTimer)
from monkeyqt import (MkInput, MkComboBox, MkSwitch, MkButton, MkMessage, apply_monkeyqt_theme, ThemeEngine)
from monkeyqt_icons import (PhPlay, PhFolderOpen, PhX, PhCube, 
                             PhDatabase, PhArrowUpRight, PhInfo, PhCpu, PhArrowCounterClockwise,
                             PhGearSix, PhCaretDown, PhCaretUp, PhCopy, PhCheck, PhDesktop, PhSliders)
from TrainingPlatform.ui.components.env_selector import SegmentedPillToggle, EnvironmentSelectorWidget, RoundedComboBox
from TrainingPlatform.core.env_manager import format_gpu_model_name

# Exact Default Training Specifications provided by User
DEFAULT_TRAIN_SETTINGS = {
    # Core
    "epochs": 100,
    "batch": "auto",
    "imgsz": 640,
    # Strategy & Regularization
    "pretrained": True,
    "patience": 100,
    "seed": 0,
    "deterministic": True,
    "amp": True,
    "cos_lr": False,
    "compile": False,
    "close_mosaic": 10,
    "save_period": -1,
    "nbs": 64,
    "cls_remap": True,
    "channels_last": None,
    "profile": False,
    "fraction": 1.0,
    "freeze": None,
    "single_cls": False,
    "rect": False,
    "multi_scale": 0.0,
    "resume": False,
    # Optimizer & LR
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    # Loss & Detection
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "dfl": 1.5,
    "max_det": 300,
    # Image Augmentations
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "bgr": 0.0,
    "cutmix": 0.0,
}

def get_default_yaml_template(task_type: str) -> str:
    """Return standard template according to task type (detect/segment/classify/obb vs pose),
    reading directly from TrainingPlatform/cfg/datasets/data.yaml and data-pose.yaml."""
    task = (task_type or "detect").lower()
    is_pose = task in ("pose", "keypoint", "kpt", "keypoints")
    yaml_name = "data-pose.yaml" if is_pose else "data.yaml"

    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.normpath(os.path.join(cur_dir, "..", "..", "..", "cfg", "datasets")),
            os.path.normpath(os.path.join(cur_dir, "..", "..", "cfg", "datasets")),
            os.path.abspath("TrainingPlatform/cfg/datasets"),
            os.path.abspath("cfg/datasets"),
        ]
        for cdir in candidate_dirs:
            target_path = os.path.join(cdir, yaml_name)
            if os.path.isfile(target_path):
                with open(target_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        return content
    except Exception as e:
        print(f"[WARN] Failed to read dataset yaml template ({yaml_name}): {e}")

    # Fallback if files cannot be read
    if is_pose:
        return """# Train and Val data path
train: E:/11-AI/标注工具/1_github/关键点数据集测试5/images/train
val: E:/11-AI/标注工具/1_github/关键点数据集测试5/images/val
test: # test images (optional)

# Number of classes
nc: 1

# Class names
names:
  0: robot

# Keypoints
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
"""
    else:
        return """# Train and Val data path
train: E:/11-AI/标注工具/1_github/fenge/images/train
val: E:/11-AI/标注工具/1_github/fenge/images/val
test: # test images (optional)

# Number of classes
nc: 2

# Class names
names:
  0: dog
  1: cat
"""

def format_filtered_yaml(yaml_path_or_content: str, task_type: str) -> str:
    """Parse uploaded YAML, extract ONLY the mandatory template keys, and format cleanly."""
    if os.path.exists(yaml_path_or_content):
        yaml_dir = os.path.dirname(os.path.abspath(yaml_path_or_content))
        with open(yaml_path_or_content, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    else:
        yaml_dir = ''
        data = yaml.safe_load(yaml_path_or_content) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML 内容格式无效，顶级结构须为键值对字典")

    root_path = data.get('path', '')
    def resolve_path(p):
        if not p:
            return ''
        p_str = str(p).strip()
        if not os.path.isabs(p_str) and yaml_dir:
            if root_path:
                full = os.path.normpath(os.path.join(yaml_dir, root_path, p_str))
            else:
                full = os.path.normpath(os.path.join(yaml_dir, p_str))
            return full.replace('\\', '/')
        return p_str.replace('\\', '/')

    train = resolve_path(data.get('train', ''))
    val = resolve_path(data.get('val', ''))
    test_raw = data.get('test', None)
    test = resolve_path(test_raw) if test_raw else '# test images (optional)'

    raw_names = data.get('names', {})
    names_dict = {}
    if isinstance(raw_names, list):
        for idx, name in enumerate(raw_names):
            names_dict[idx] = str(name)
    elif isinstance(raw_names, dict):
        for k, v in raw_names.items():
            try:
                names_dict[int(k)] = str(v)
            except ValueError:
                names_dict[k] = str(v)
    else:
        names_dict = {0: 'class0'}

    nc = data.get('nc', len(names_dict))

    lines = [
        '# Train and Val data path',
        f'train: {train}',
        f'val: {val}',
        f'test: {test}',
        '',
        '# Number of classes',
        f'nc: {nc}',
        '',
        '# Class names',
        'names:'
    ]
    for k in sorted(names_dict.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit()) else str(x)):
        lines.append(f'  {k}: {names_dict[k]}')

    task = (task_type or 'detect').lower()
    if task == 'pose':
        kpt_shape = data.get('kpt_shape', [17, 3])
        flip_idx = data.get('flip_idx', [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
        lines.extend([
            '',
            '# Keypoints',
            f'kpt_shape: {kpt_shape}',
            f'flip_idx: {flip_idx}'
        ])

    return '\n'.join(lines)


class YamlHighlighter(QSyntaxHighlighter):
    """Sleek syntax highlighter for YAML configuration cards benchmarked against modern IDEs."""
    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark

    def set_dark(self, is_dark: bool):
        self.is_dark = is_dark
        self.rehighlight()

    def highlightBlock(self, text: str):
        import re
        # Comments (# ...)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor('#71717a' if self.is_dark else '#64748b'))
        comment_format.setFontItalic(True)
        for m in re.finditer(r'#.*', text):
            self.setFormat(m.start(), m.end() - m.start(), comment_format)

        # Keys (e.g. 'train:', 'val:', 'nc:', '0:')
        key_format = QTextCharFormat()
        key_format.setForeground(QColor('#f4f4f5' if self.is_dark else '#09090b'))
        key_format.setFontWeight(QFont.Weight.Bold)
        for m in re.finditer(r'^\s*([a-zA-Z0-9_\-]+:)', text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), key_format)
        for m in re.finditer(r'^\s*(\d+:)', text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), key_format)

        # Numbers and Booleans
        val_format = QTextCharFormat()
        val_format.setForeground(QColor('#60a5fa' if self.is_dark else '#2563eb'))
        for m in re.finditer(r'\b(true|false|null|\d+(\.\d+)?)\b', text):
            c_idx = text.find('#')
            if c_idx != -1 and m.start() > c_idx:
                continue
            self.setFormat(m.start(), m.end() - m.start(), val_format)


class YamlCodeEditor(QFrame):
    """
    Code-block styled YAML editor matching modern cloud AI platforms (Ultralytics HUB, W&B):
    - Monospaced typography with syntax highlighting
    - Integrated floating copy button in the top-right corner
    - Drag & drop support for .yaml / .yml files
    - Sleek 5px scrollbars with zero OS arrows
    """
    def __init__(self, parent=None, task_type='detect'):
        super().__init__(parent)
        self.task_type = task_type
        self.setAcceptDrops(True)
        self._is_syncing = False
        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self.sync_to_local_files)
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("YamlCodeEditorCard")
        self.setFixedHeight(156)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)

        self.editor = QPlainTextEdit(self)
        self.editor.setFrameShape(QFrame.Shape.NoFrame)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(' ') * 2)
        layout.addWidget(self.editor)

        self.highlighter = YamlHighlighter(self.editor.document(), is_dark=False)

        # Floating Copy Button in top right
        self.btn_copy = QToolButton(self)
        self.btn_copy.setFixedSize(26, 26)
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setToolTip("复制 YAML 配置")
        self.btn_copy.setIcon(PhCopy.regular(size=13, color="#64748b"))
        self.btn_copy.clicked.connect(self._copy_content)

        # Real-time synchronization when user modifies text in editor
        self.editor.textChanged.connect(self._on_editor_text_changed)

        # Populate initial template and sync to disk
        self.reset_to_default_template()

    def _on_editor_text_changed(self):
        if not self._is_syncing:
            self._sync_timer.start(200)

    def sync_to_local_files(self):
        """Sync current YAML content in real time to actual TrainingPlatform/cfg/datasets file."""
        content = self.editor.toPlainText()
        if not content.strip():
            return

        self._is_syncing = True
        try:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            tp_dir = os.path.normpath(os.path.join(cur_dir, "..", "..", ".."))

            is_pose = (self.task_type or "detect").lower() in ("pose", "keypoint", "kpt", "keypoints")
            cfg_filename = "data-pose.yaml" if is_pose else "data.yaml"
            cfg_target = os.path.join(tp_dir, "cfg", "datasets", cfg_filename)

            # Update actual YAML file
            try:
                os.makedirs(os.path.dirname(cfg_target), exist_ok=True)
                with open(cfg_target, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"[WARN] Failed to write template {cfg_target}: {e}")
        finally:
            self._is_syncing = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_copy.move(self.width() - self.btn_copy.width() - 8, 8)

    def _copy_content(self):
        self.sync_to_local_files()
        text = self.editor.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.btn_copy.setIcon(PhCheck.bold(size=13, color="#16a34a"))
            self.btn_copy.setToolTip("已复制到剪贴板!")
            QTimer.singleShot(1500, self._reset_copy_icon)

    def _reset_copy_icon(self):
        color = "#a1a1aa" if getattr(self, "_is_dark", False) else "#64748b"
        self.btn_copy.setIcon(PhCopy.regular(size=13, color=color))
        self.btn_copy.setToolTip("复制 YAML 配置")

    def set_yaml_text(self, text: str):
        self._is_syncing = True
        try:
            self.editor.setPlainText(text)
        finally:
            self._is_syncing = False
        self.sync_to_local_files()

    def get_yaml_text(self) -> str:
        return self.editor.toPlainText()

    def reset_to_default_template(self):
        template = get_default_yaml_template(self.task_type)
        self.set_yaml_text(template)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(('.yaml', '.yml')) for u in urls):
                event.acceptProposedAction()
                return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            fpath = url.toLocalFile()
            if fpath.lower().endswith(('.yaml', '.yml')) and os.path.exists(fpath):
                formatted = format_filtered_yaml(fpath, self.task_type)
                self.set_yaml_text(formatted)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def apply_theme(self, is_dark: bool):
        self._is_dark = is_dark
        self.highlighter.set_dark(is_dark)
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#YamlCodeEditorCard {
                    background-color: #141416;
                    border: 1px solid #27272a;
                    border-radius: 8px;
                }
                QPlainTextEdit {
                    background: transparent;
                    color: #f4f4f5;
                    border: none;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 5px;
                    margin: 0px;
                    border-radius: 2px;
                }
                QScrollBar::handle:vertical {
                    background: #3f3f46;
                    min-height: 20px;
                    border-radius: 2px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #52525b;
                }
                QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    height: 5px;
                    margin: 0px;
                    border-radius: 2px;
                }
                QScrollBar::handle:horizontal {
                    background: #3f3f46;
                    min-width: 20px;
                    border-radius: 2px;
                }
                QScrollBar::handle:horizontal:hover {
                    background: #52525b;
                }
                QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                    background: none;
                    border: none;
                    width: 0px;
                    height: 0px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_copy, """
                QToolButton {
                    background-color: #27272a;
                    border: 1px solid #3f3f46;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #3f3f46;
                    border-color: #52525b;
                }
            """)
            self.btn_copy.setIcon(PhCopy.regular(size=13, color="#a1a1aa"))
        else:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#YamlCodeEditorCard {
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                }
                QPlainTextEdit {
                    background: transparent;
                    color: #09090b;
                    border: none;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 5px;
                    margin: 0px;
                    border-radius: 2px;
                }
                QScrollBar::handle:vertical {
                    background: #cbd5e1;
                    min-height: 20px;
                    border-radius: 2px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #94a3b8;
                }
                QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    height: 5px;
                    margin: 0px;
                    border-radius: 2px;
                }
                QScrollBar::handle:horizontal {
                    background: #cbd5e1;
                    min-width: 20px;
                    border-radius: 2px;
                }
                QScrollBar::handle:horizontal:hover {
                    background: #94a3b8;
                }
                QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                    background: none;
                    border: none;
                    width: 0px;
                    height: 0px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_copy, """
                QToolButton {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #f1f5f9;
                    border-color: #cbd5e1;
                }
            """)
            self.btn_copy.setIcon(PhCopy.regular(size=13, color="#64748b"))


class CliHighlighter(QSyntaxHighlighter):
    """
    Sleek syntax highlighter for YOLO CLI training commands benchmarked against Ultralytics HUB terminal blocks:
    - Keywords (yolo train) in primary bold
    - Argument keys (model, data, task, epochs, batch, imgsz, etc.) in slate gray
    - Quoted strings ("...") in vibrant accent blue (#2563eb / #60a5fa)
    - Numbers and booleans in clean modern accent
    - Line continuation backslashes in muted gray
    """
    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark

    def set_dark(self, is_dark: bool):
        self.is_dark = is_dark
        self.rehighlight()

    def highlightBlock(self, text: str):
        import re

        # 1. Command keywords (e.g. 'yolo train', 'yolo')
        cmd_format = QTextCharFormat()
        cmd_format.setForeground(QColor('#0f172a' if not self.is_dark else '#f8fafc'))
        cmd_format.setFontWeight(QFont.Weight.Bold)
        for m in re.finditer(r'\b(yolo\s+train|yolo)\b', text):
            self.setFormat(m.start(), m.end() - m.start(), cmd_format)

        # 2. Argument keys before '=' (e.g. 'model', 'data', 'task', 'epochs', 'batch', 'imgsz', 'fliplr')
        key_format = QTextCharFormat()
        key_format.setForeground(QColor('#334155' if not self.is_dark else '#cbd5e1'))
        key_format.setFontWeight(QFont.Weight.Medium)
        for m in re.finditer(r'^\s*([a-zA-Z0-9_\-]+)(?==)', text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), key_format)

        # 3. Quoted strings (e.g. "detect", "dataset.yaml", "yolo11n.pt")
        str_format = QTextCharFormat()
        str_format.setForeground(QColor('#2563eb' if not self.is_dark else '#60a5fa'))
        for m in re.finditer(r'\"[^\"]*\"', text):
            self.setFormat(m.start(), m.end() - m.start(), str_format)

        # 4. Numbers (e.g. 100, -1, 640, 0.6)
        num_format = QTextCharFormat()
        num_format.setForeground(QColor('#0284c7' if not self.is_dark else '#38bdf8'))
        for m in re.finditer(r'(?<==)-?\d+(\.\d+)?', text):
            self.setFormat(m.start(), m.end() - m.start(), num_format)

        # 5. Booleans (True / False / true / false)
        bool_format = QTextCharFormat()
        bool_format.setForeground(QColor('#7c3aed' if not self.is_dark else '#a78bfa'))
        for m in re.finditer(r'(?<==)(True|False|true|false)\b', text):
            self.setFormat(m.start(), m.end() - m.start(), bool_format)

        # 6. Line continuation symbols ('\', '^', '`')
        slash_format = QTextCharFormat()
        slash_format.setForeground(QColor('#94a3b8' if not self.is_dark else '#64748b'))
        for m in re.finditer(r'[\\^`]$', text):
            self.setFormat(m.start(), m.end() - m.start(), slash_format)



class CliTextEdit(QPlainTextEdit):
    """Custom PlainTextEdit that strips trailing whitespace on copy."""
    def copy(self):
        cursor = self.textCursor()
        text = cursor.selectedText()
        if text:
            text = text.replace('\u2029', '\n')
            clean_lines = [line.rstrip() for line in text.splitlines()]
            clean_text = "\n".join(clean_lines).strip()
            while clean_text.endswith(" ") or clean_text.endswith("\t"):
                clean_text = clean_text[:-1]
            QApplication.clipboard().setText(clean_text)
        else:
            full = self.toPlainText().strip()
            clean_lines = [line.rstrip() for line in full.splitlines()]
            clean_text = "\n".join(clean_lines).strip()
            while clean_text.endswith(" ") or clean_text.endswith("\t"):
                clean_text = clean_text[:-1]
            QApplication.clipboard().setText(clean_text)


class CliCommandViewer(QFrame):
    """
    Modern VSCode/Terminal-like CLI Preview Block.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CliCommandViewerCard")
        self._is_dark = False
        self._on_copy_hook = None
        self.setFixedHeight(162)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 36, 6)
        layout.setSpacing(0)

        self.editor = CliTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.editor.setFrameShape(QFrame.Shape.NoFrame)
        self.editor.document().setDocumentMargin(1)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(' ') * 2)
        layout.addWidget(self.editor)

        self.highlighter = CliHighlighter(self.editor.document(), is_dark=False)

        # Floating Copy Button in top right
        self.btn_copy = QToolButton(self)
        self.btn_copy.setFixedSize(26, 26)
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setToolTip("复制完整 CLI 命令 (Copy command)")
        self.btn_copy.setIcon(PhCopy.regular(size=13, color="#64748b"))
        self.btn_copy.clicked.connect(self._copy_content)

    def set_command_text(self, text: str):
        clean_text = text.strip()
        while clean_text.endswith(" ") or clean_text.endswith("\t"):
            clean_text = clean_text[:-1]
        if self.editor.toPlainText() != clean_text:
            tc = self.editor.textCursor()
            pos = tc.position()
            self.editor.setPlainText(clean_text)
            if pos < len(clean_text):
                tc.setPosition(pos)
                self.editor.setTextCursor(tc)

    def get_command_text(self) -> str:
        return self.editor.toPlainText()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_copy.move(self.width() - self.btn_copy.width() - 8, 8)

    def _copy_content(self):
        if hasattr(self, "_on_copy_hook") and callable(self._on_copy_hook):
            try:
                self._on_copy_hook()
            except Exception:
                pass
        text = self.editor.toPlainText().strip()
        if text:
            clean_lines = [line.rstrip() for line in text.splitlines()]
            clean_text = "\n".join(clean_lines).strip()
            while clean_text.endswith(" ") or clean_text.endswith("\t"):
                clean_text = clean_text[:-1]
            QApplication.clipboard().setText(clean_text)
            self.btn_copy.setIcon(PhCheck.bold(size=13, color="#16a34a"))
            self.btn_copy.setToolTip("已复制到剪贴板!")
            QTimer.singleShot(1500, self._reset_copy_icon)

    def _reset_copy_icon(self):
        color = "#a1a1aa" if self._is_dark else "#64748b"
        self.btn_copy.setIcon(PhCopy.regular(size=13, color=color))
        self.btn_copy.setToolTip("复制完整 CLI 命令 (Copy command)")

    def apply_theme(self, is_dark: bool):
        self._is_dark = is_dark
        if hasattr(self, "highlighter") and self.highlighter:
            self.highlighter.set_dark(is_dark)
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#CliCommandViewerCard {
                    background-color: #0c0c0e;
                    border: 1px solid #27272a;
                    border-radius: 8px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.editor, """
                QPlainTextEdit {
                    background-color: transparent;
                    color: #e2e8f0;
                    border: none;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    width: 5px;
                    height: 5px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                    background: #3f3f46;
                    border-radius: 2px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                    background: #52525b;
                }
                QScrollBar::add-line, QScrollBar::sub-line {
                    border: none;
                    background: none;
                    width: 0px;
                    height: 0px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_copy, """
                QToolButton {
                    border: 1px solid #3f3f46;
                    background-color: #1e1e20;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #27272a;
                    border-color: #52525b;
                }
            """)
            self.btn_copy.setIcon(PhCopy.regular(size=13, color="#a1a1aa"))
        else:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#CliCommandViewerCard {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.editor, """
                QPlainTextEdit {
                    background-color: transparent;
                    color: #0f172a;
                    border: none;
                    selection-background-color: #bfdbfe;
                    selection-color: #0f172a;
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    border: none;
                    background: transparent;
                    width: 5px;
                    height: 5px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                    background: #cbd5e1;
                    border-radius: 2px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                    background: #94a3b8;
                }
                QScrollBar::add-line, QScrollBar::sub-line {
                    border: none;
                    background: none;
                    width: 0px;
                    height: 0px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_copy, """
                QToolButton {
                    border: 1px solid #e2e8f0;
                    background-color: #f8fafc;
                    border-radius: 4px;
                }
                QToolButton:hover {
                    background-color: #f1f5f9;
                    border-color: #cbd5e1;
                }
            """)
            self.btn_copy.setIcon(PhCopy.regular(size=13, color="#64748b"))


def get_series_icon(series_name: str, size: int = 16) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    grad = QLinearGradient(0, 0, size, size)
    s_lower = series_name.lower()
    if '26' in s_lower:
        grad.setColorAt(0.0, QColor('#38bdf8'))
        grad.setColorAt(1.0, QColor('#1d4ed8'))
    elif '11' in s_lower:
        grad.setColorAt(0.0, QColor('#4ade80'))
        grad.setColorAt(1.0, QColor('#06b6d4'))
    elif 'v8' in s_lower:
        grad.setColorAt(0.0, QColor('#10b981'))
        grad.setColorAt(1.0, QColor('#0f766e'))
    else:
        grad.setColorAt(0.0, QColor('#a855f7'))
        grad.setColorAt(1.0, QColor('#6366f1'))
        
    painter.setBrush(grad)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return pix


MODEL_CATALOG = {
    "detect": {
        "tag": "Detect",
        "default": "yolo26n.pt",
        "series": {
            "YOLO26": [
                {"id": "yolo26n.pt", "name": "yolo26n", "meta": "40.1% · 5 MB"},
                {"id": "yolo26s.pt", "name": "yolo26s", "meta": "47.8% · 19 MB"},
                {"id": "yolo26m.pt", "name": "yolo26m", "meta": "52.5% · 42 MB"},
                {"id": "yolo26l.pt", "name": "yolo26l", "meta": "54.3% · 51 MB"},
                {"id": "yolo26x.pt", "name": "yolo26x", "meta": "56.8% · 113 MB"},
            ],
            "YOLO11": [
                {"id": "yolo11n.pt", "name": "yolo11n", "meta": "39.4% · 5 MB"},
                {"id": "yolo11s.pt", "name": "yolo11s", "meta": "46.9% · 18 MB"},
                {"id": "yolo11m.pt", "name": "yolo11m", "meta": "51.4% · 39 MB"},
                {"id": "yolo11l.pt", "name": "yolo11l", "meta": "53.3% · 49 MB"},
                {"id": "yolo11x.pt", "name": "yolo11x", "meta": "54.7% · 109 MB"},
            ],
            "YOLOv8": [
                {"id": "yolov8n.pt", "name": "yolov8n", "meta": "37.3% · 6 MB"},
                {"id": "yolov8s.pt", "name": "yolov8s", "meta": "44.9% · 22 MB"},
                {"id": "yolov8m.pt", "name": "yolov8m", "meta": "50.2% · 50 MB"},
                {"id": "yolov8l.pt", "name": "yolov8l", "meta": "52.9% · 84 MB"},
                {"id": "yolov8x.pt", "name": "yolov8x", "meta": "53.9% · 131 MB"},
            ],
        }
    },
    "segment": {
        "tag": "Segment",
        "default": "yolo26n-seg.pt",
        "series": {
            "YOLO26": [
                {"id": "yolo26n-seg.pt", "name": "yolo26n-seg", "meta": "32.5% · 6 MB"},
                {"id": "yolo26s-seg.pt", "name": "yolo26s-seg", "meta": "38.8% · 23 MB"},
                {"id": "yolo26m-seg.pt", "name": "yolo26m-seg", "meta": "43.1% · 52 MB"},
                {"id": "yolo26l-seg.pt", "name": "yolo26l-seg", "meta": "45.0% · 63 MB"},
                {"id": "yolo26x-seg.pt", "name": "yolo26x-seg", "meta": "47.1% · 139 MB"},
            ],
            "YOLO11": [
                {"id": "yolo11n-seg.pt", "name": "yolo11n-seg", "meta": "31.8% · 6 MB"},
                {"id": "yolo11s-seg.pt", "name": "yolo11s-seg", "meta": "38.0% · 22 MB"},
                {"id": "yolo11m-seg.pt", "name": "yolo11m-seg", "meta": "42.5% · 48 MB"},
                {"id": "yolo11l-seg.pt", "name": "yolo11l-seg", "meta": "44.2% · 60 MB"},
                {"id": "yolo11x-seg.pt", "name": "yolo11x-seg", "meta": "46.4% · 134 MB"},
            ],
            "YOLOv8": [
                {"id": "yolov8n-seg.pt", "name": "yolov8n-seg", "meta": "30.5% · 7 MB"},
                {"id": "yolov8s-seg.pt", "name": "yolov8s-seg", "meta": "36.8% · 26 MB"},
                {"id": "yolov8m-seg.pt", "name": "yolov8m-seg", "meta": "40.8% · 57 MB"},
                {"id": "yolov8l-seg.pt", "name": "yolov8l-seg", "meta": "42.6% · 96 MB"},
                {"id": "yolov8x-seg.pt", "name": "yolov8x-seg", "meta": "43.4% · 143 MB"},
            ],
        }
    },
    "pose": {
        "tag": "Pose",
        "default": "yolo26n-pose.pt",
        "series": {
            "YOLO26": [
                {"id": "yolo26n-pose.pt", "name": "yolo26n-pose", "meta": "51.5% · 6 MB"},
                {"id": "yolo26s-pose.pt", "name": "yolo26s-pose", "meta": "60.5% · 21 MB"},
                {"id": "yolo26m-pose.pt", "name": "yolo26m-pose", "meta": "65.2% · 46 MB"},
                {"id": "yolo26l-pose.pt", "name": "yolo26l-pose", "meta": "68.1% · 56 MB"},
                {"id": "yolo26x-pose.pt", "name": "yolo26x-pose", "meta": "70.2% · 124 MB"},
            ],
            "YOLO11": [
                {"id": "yolo11n-pose.pt", "name": "yolo11n-pose", "meta": "50.8% · 6 MB"},
                {"id": "yolo11s-pose.pt", "name": "yolo11s-pose", "meta": "59.8% · 20 MB"},
                {"id": "yolo11m-pose.pt", "name": "yolo11m-pose", "meta": "64.5% · 44 MB"},
                {"id": "yolo11l-pose.pt", "name": "yolo11l-pose", "meta": "67.2% · 54 MB"},
                {"id": "yolo11x-pose.pt", "name": "yolo11x-pose", "meta": "69.4% · 120 MB"},
            ],
            "YOLOv8": [
                {"id": "yolov8n-pose.pt", "name": "yolov8n-pose", "meta": "50.4% · 7 MB"},
                {"id": "yolov8s-pose.pt", "name": "yolov8s-pose", "meta": "60.0% · 24 MB"},
                {"id": "yolov8m-pose.pt", "name": "yolov8m-pose", "meta": "65.0% · 52 MB"},
                {"id": "yolov8l-pose.pt", "name": "yolov8l-pose", "meta": "67.6% · 88 MB"},
                {"id": "yolov8x-pose.pt", "name": "yolov8x-pose", "meta": "69.2% · 135 MB"},
            ],
        }
    },
    "obb": {
        "tag": "OBB",
        "default": "yolo26n-obb.pt",
        "series": {
            "YOLO26": [
                {"id": "yolo26n-obb.pt", "name": "yolo26n-obb", "meta": "79.5% · 6 MB"},
                {"id": "yolo26s-obb.pt", "name": "yolo26s-obb", "meta": "81.2% · 20 MB"},
                {"id": "yolo26m-obb.pt", "name": "yolo26m-obb", "meta": "82.3% · 44 MB"},
                {"id": "yolo26l-obb.pt", "name": "yolo26l-obb", "meta": "82.8% · 54 MB"},
                {"id": "yolo26x-obb.pt", "name": "yolo26x-obb", "meta": "83.4% · 120 MB"},
            ],
            "YOLO11": [
                {"id": "yolo11n-obb.pt", "name": "yolo11n-obb", "meta": "78.4% · 6 MB"},
                {"id": "yolo11s-obb.pt", "name": "yolo11s-obb", "meta": "80.5% · 19 MB"},
                {"id": "yolo11m-obb.pt", "name": "yolo11m-obb", "meta": "81.7% · 42 MB"},
                {"id": "yolo11l-obb.pt", "name": "yolo11l-obb", "meta": "82.2% · 52 MB"},
                {"id": "yolo11x-obb.pt", "name": "yolo11x-obb", "meta": "82.8% · 116 MB"},
            ],
            "YOLOv8": [
                {"id": "yolov8n-obb.pt", "name": "yolov8n-obb", "meta": "77.0% · 7 MB"},
                {"id": "yolov8s-obb.pt", "name": "yolov8s-obb", "meta": "79.5% · 23 MB"},
                {"id": "yolov8m-obb.pt", "name": "yolov8m-obb", "meta": "80.8% · 51 MB"},
                {"id": "yolov8l-obb.pt", "name": "yolov8l-obb", "meta": "81.3% · 86 MB"},
                {"id": "yolov8x-obb.pt", "name": "yolov8x-obb", "meta": "81.6% · 134 MB"},
            ],
        }
    },
    "classify": {
        "tag": "Classify",
        "default": "yolo26n-cls.pt",
        "series": {
            "YOLO26": [
                {"id": "yolo26n-cls.pt", "name": "yolo26n-cls", "meta": "78.2% · 4 MB"},
                {"id": "yolo26s-cls.pt", "name": "yolo26s-cls", "meta": "80.5% · 14 MB"},
                {"id": "yolo26m-cls.pt", "name": "yolo26m-cls", "meta": "82.1% · 30 MB"},
                {"id": "yolo26l-cls.pt", "name": "yolo26l-cls", "meta": "83.4% · 39 MB"},
                {"id": "yolo26x-cls.pt", "name": "yolo26x-cls", "meta": "84.2% · 86 MB"},
            ],
            "YOLO11": [
                {"id": "yolo11n-cls.pt", "name": "yolo11n-cls", "meta": "77.2% · 4 MB"},
                {"id": "yolo11s-cls.pt", "name": "yolo11s-cls", "meta": "79.8% · 13 MB"},
                {"id": "yolo11m-cls.pt", "name": "yolo11m-cls", "meta": "81.5% · 29 MB"},
                {"id": "yolo11l-cls.pt", "name": "yolo11l-cls", "meta": "82.8% · 38 MB"},
                {"id": "yolo11x-cls.pt", "name": "yolo11x-cls", "meta": "83.5% · 84 MB"},
            ],
            "YOLOv8": [
                {"id": "yolov8n-cls.pt", "name": "yolov8n-cls", "meta": "69.0% · 5 MB"},
                {"id": "yolov8s-cls.pt", "name": "yolov8s-cls", "meta": "73.8% · 16 MB"},
                {"id": "yolov8m-cls.pt", "name": "yolov8m-cls", "meta": "76.8% · 36 MB"},
                {"id": "yolov8l-cls.pt", "name": "yolov8l-cls", "meta": "78.3% · 62 MB"},
                {"id": "yolov8x-cls.pt", "name": "yolov8x-cls", "meta": "79.0% · 98 MB"},
            ],
        }
    }
}


class ModelRowItem(QFrame):
    clicked = Signal(str)

    def __init__(self, model_info: dict, is_selected: bool, is_dark: bool, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.is_selected = is_selected
        self.is_dark = is_dark
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(34)
        self._init_ui()

    def _init_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 12, 0)
        lay.setSpacing(8)

        self.lbl_check = QLabel()
        self.lbl_check.setFixedSize(14, 14)
        if self.is_selected:
            color = "#38bdf8" if self.is_dark else "#2563eb"
            self.lbl_check.setPixmap(PhCheck.pixmap(size=13, color=color, weight="bold", dpr=2.0))
        lay.addWidget(self.lbl_check)

        self.lbl_name = QLabel(self.model_info["name"])
        font = QFont("Consolas")
        font.setPointSize(10)
        if self.is_selected:
            font.setBold(True)
        self.lbl_name.setFont(font)
        lay.addWidget(self.lbl_name)

        lay.addStretch()

        self.lbl_meta = QLabel(self.model_info["meta"])
        _MkThemeEngine.apply_style_sheet(self.lbl_meta, "font-size: 11px; color: #94a3b8;" if self.is_dark else "font-size: 11px; color: #64748b;")
        lay.addWidget(self.lbl_meta)

        self._apply_style()

    def _apply_style(self):
        if self.is_dark:
            bg = "#27272a" if self.is_selected else "transparent"
            text_color = "#f4f4f5"
            hover_bg = "#3f3f46"
        else:
            bg = "#eff6ff" if self.is_selected else "transparent"
            text_color = "#0f172a"
            hover_bg = "#f1f5f9"

        _MkThemeEngine.apply_style_sheet(self.lbl_name, f"color: {text_color}; background: transparent; border: none;")
        _MkThemeEngine.apply_style_sheet(self, f"""
            QFrame {{
                background-color: {bg};
                border-radius: 6px;
                border: none;
            }}
            QFrame:hover {{
                background-color: {hover_bg};
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.model_info["id"])


class ModelPickerPopup(QWidget):
    modelSelected = Signal(str)

    def __init__(self, task_type: str, current_model: str, is_dark: bool = False, parent=None):
        super().__init__(None, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("ModelPickerPopupContainer")
        self.task_type = (task_type or "detect").lower()
        self.current_model = current_model
        self.is_dark = is_dark
        self.margin = 10
        self.active_series = self._detect_series(current_model)
        self._init_ui()

    def _detect_series(self, model_id: str) -> str:
        m = (model_id or "").lower()
        if "yolo26" in m:
            return "YOLO26"
        elif "yolo11" in m:
            return "YOLO11"
        elif "yolov8" in m:
            return "YOLOv8"
        return "YOLO26"

    def _init_ui(self):
        # Outer container layout for drop shadow buffer
        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(self.margin, self.margin, self.margin, self.margin)
        root_lay.setSpacing(0)

        # Main Card with Crisp Border & Soft Drop Shadow (Referencing Fig 3)
        self.card = QFrame(self)
        self.card.setObjectName("ModelPickerCard")

        self.shadow = QGraphicsDropShadowEffect(self.card)
        self.shadow.setBlurRadius(20)
        self.shadow.setColor(QColor(0, 0, 0, 40) if not self.is_dark else QColor(0, 0, 0, 95))
        self.shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(self.shadow)

        # 0px margin inside card to perfectly align child column corners with outer border
        card_lay = QHBoxLayout(self.card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        # ── Left Column: Series Categories ──
        self.col_left = QFrame(self.card)
        self.col_left.setFixedWidth(120)
        self.left_lay = QVBoxLayout(self.col_left)
        self.left_lay.setContentsMargins(8, 10, 8, 10)
        self.left_lay.setSpacing(4)
        self.left_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_series = QLabel("系列 (Series)")
        _MkThemeEngine.apply_style_sheet(lbl_series, "font-size: 11px; font-weight: 600; color: #94a3b8; padding-left: 4px;" if self.is_dark else "font-size: 11px; font-weight: 600; color: #64748b; padding-left: 4px;")
        self.left_lay.addWidget(lbl_series)

        self.series_btns = {}
        for s_name in ["YOLO26", "YOLO11", "YOLOv8"]:
            btn = QPushButton()
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(34)
            b_lay = QHBoxLayout(btn)
            b_lay.setContentsMargins(8, 0, 8, 0)
            b_lay.setSpacing(6)

            ic = QLabel()
            ic.setPixmap(get_series_icon(s_name, 16))
            _MkThemeEngine.apply_style_sheet(ic, "background: transparent; border: none;")
            b_lay.addWidget(ic)

            txt = QLabel(s_name)
            _MkThemeEngine.apply_style_sheet(txt, "font-size: 12px; font-weight: 600; background: transparent; border: none;")
            b_lay.addWidget(txt)
            b_lay.addStretch()

            count = QLabel("5")
            _MkThemeEngine.apply_style_sheet(count, "font-size: 11px; color: #94a3b8; background: transparent; border: none;")
            b_lay.addWidget(count)

            btn.clicked.connect(lambda checked=False, s=s_name: self._select_series(s))
            self.series_btns[s_name] = (btn, txt, count)
            self.left_lay.addWidget(btn)

        self.left_lay.addStretch()
        card_lay.addWidget(self.col_left)

        # ── Right Column: Models List ──
        self.col_right = QFrame(self.card)
        right_lay = QVBoxLayout(self.col_right)
        right_lay.setContentsMargins(10, 10, 10, 10)
        right_lay.setSpacing(2)

        # Models Container
        self.models_container = QWidget()
        self.models_lay = QVBoxLayout(self.models_container)
        self.models_lay.setContentsMargins(0, 0, 0, 0)
        self.models_lay.setSpacing(2)
        right_lay.addWidget(self.models_container)
        right_lay.addStretch()

        card_lay.addWidget(self.col_right, stretch=1)
        root_lay.addWidget(self.card)

        self.apply_theme(self.is_dark)
        self._refresh_series_buttons()
        self._refresh_models_list()

    def _select_series(self, series_name: str):
        self.active_series = series_name
        self._refresh_series_buttons()
        self._refresh_models_list()

    def _refresh_series_buttons(self):
        for s_name, (btn, txt, count) in self.series_btns.items():
            is_active = (s_name == self.active_series)
            if self.is_dark:
                if is_active:
                    _MkThemeEngine.apply_style_sheet(btn, "QPushButton { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; }")
                    _MkThemeEngine.apply_style_sheet(txt, "font-size: 12px; font-weight: 700; color: #ffffff; background: transparent; border: none;")
                else:
                    _MkThemeEngine.apply_style_sheet(btn, "QPushButton { background-color: transparent; border: none; border-radius: 6px; } QPushButton:hover { background-color: #1f1f23; }")
                    _MkThemeEngine.apply_style_sheet(txt, "font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent; border: none;")
            else:
                if is_active:
                    _MkThemeEngine.apply_style_sheet(btn, "QPushButton { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; }")
                    _MkThemeEngine.apply_style_sheet(txt, "font-size: 12px; font-weight: 700; color: #0f172a; background: transparent; border: none;")
                else:
                    _MkThemeEngine.apply_style_sheet(btn, "QPushButton { background-color: transparent; border: none; border-radius: 6px; } QPushButton:hover { background-color: #f1f5f9; }")
                    _MkThemeEngine.apply_style_sheet(txt, "font-size: 12px; font-weight: 500; color: #64748b; background: transparent; border: none;")

    def _refresh_models_list(self):
        while self.models_lay.count():
            item = self.models_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        task_data = MODEL_CATALOG.get(self.task_type, MODEL_CATALOG["detect"])
        models = task_data["series"].get(self.active_series, [])
        for m in models:
            is_sel = (m["id"] == self.current_model or m["name"] == self.current_model)
            row = ModelRowItem(m, is_selected=is_sel, is_dark=self.is_dark, parent=self.models_container)
            row.setFixedHeight(32)
            row.clicked.connect(self._on_model_clicked)
            self.models_lay.addWidget(row)

    def _on_model_clicked(self, model_id: str):
        self.current_model = model_id
        self.modelSelected.emit(model_id)
        self.close()

    def show_at(self, widget: QWidget):
        card_w = max(widget.width(), 320)
        total_w = card_w + self.margin * 2

        # Dynamically calculate required height based on actual models count
        models_count = self.models_lay.count()
        content_h = 20 + models_count * 32 + max(0, models_count - 1) * 2
        card_h = max(184, content_h)
        total_h = card_h + self.margin * 2
        
        self.setFixedSize(total_w, total_h)
        self.card.setFixedSize(card_w, card_h)

        # Align inner card precisely with widget: window X offset by -margin, Y offset by height + 4 - margin
        pos = widget.mapToGlobal(QPoint(-self.margin, widget.height() + 4 - self.margin))
        
        # Keep inside screen boundary
        screen = widget.screen().availableGeometry() if widget.screen() else QApplication.primaryScreen().availableGeometry()
        x = pos.x()
        y = pos.y()
        if x + total_w > screen.right():
            x = screen.right() - total_w - 8
        if x < screen.left():
            x = screen.left() + 8
        if y + total_h > screen.bottom():
            y = widget.mapToGlobal(QPoint(-self.margin, -total_h + self.margin - 4)).y()
            
        self.move(QPoint(x, y))
        self.show()

    def apply_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if hasattr(self, "shadow") and self.shadow:
            self.shadow.setColor(QColor(0, 0, 0, 40) if not is_dark else QColor(0, 0, 0, 95))

        if is_dark:
            _MkThemeEngine.apply_style_sheet(self, "QWidget#ModelPickerPopupContainer { background: transparent; border: none; }")
            _MkThemeEngine.apply_style_sheet(self.card, """
                QFrame#ModelPickerCard {
                    background-color: #141416;
                    border: 1px solid #3f3f46;
                    border-radius: 12px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.col_left, """
                background-color: #111113;
                border: none;
                border-right: 1px solid #27272a;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
            """)
            _MkThemeEngine.apply_style_sheet(self.col_right, """
                background-color: transparent;
                border: none;
            """)
        else:
            _MkThemeEngine.apply_style_sheet(self, "QWidget#ModelPickerPopupContainer { background: transparent; border: none; }")
            _MkThemeEngine.apply_style_sheet(self.card, """
                QFrame#ModelPickerCard {
                    background-color: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 12px;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.col_left, """
                background-color: #f8fafc;
                border: none;
                border-right: 1px solid #e2e8f0;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
            """)
            _MkThemeEngine.apply_style_sheet(self.col_right, """
                background-color: transparent;
                border: none;
            """)
        self._refresh_series_buttons()
        self._refresh_models_list()


class ModelPickerWidget(QFrame):
    currentTextChanged = Signal(str)
    currentIndexChanged = Signal(int)

    def __init__(self, task_type: str = "detect", parent=None):
        super().__init__(parent)
        self.task_type = (task_type or "detect").lower()
        task_data = MODEL_CATALOG.get(self.task_type, MODEL_CATALOG["detect"])
        self._current_model = task_data["default"]
        self._is_dark = False
        self._popup = None
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("ModelPickerCard")
        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        self.lbl_icon = QLabel()
        _MkThemeEngine.apply_style_sheet(self.lbl_icon, "background: transparent; border: none;")
        lay.addWidget(self.lbl_icon)

        self.lbl_text = QLabel()
        font = QFont("Consolas, Segoe UI, sans-serif")
        font.setPointSize(10)
        font.setBold(True)
        self.lbl_text.setFont(font)
        _MkThemeEngine.apply_style_sheet(self.lbl_text, "background: transparent; border: none;")
        lay.addWidget(self.lbl_text)

        lay.addStretch()

        self.lbl_caret = QLabel()
        _MkThemeEngine.apply_style_sheet(self.lbl_caret, "background: transparent; border: none;")
        lay.addWidget(self.lbl_caret)

        self._update_display()
        self.apply_theme(False)

    def _update_display(self):
        self.lbl_text.setText(self._current_model)
        m = self._current_model.lower()
        if "yolo26" in m:
            s_name = "YOLO26"
        elif "yolo11" in m:
            s_name = "YOLO11"
        elif "yolov8" in m:
            s_name = "YOLOv8"
        else:
            s_name = "YOLO26"
        self.lbl_icon.setPixmap(get_series_icon(s_name, 18))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_popup()

    def _open_popup(self):
        if not self._popup:
            self._popup = ModelPickerPopup(self.task_type, self._current_model, is_dark=self._is_dark, parent=self)
            self._popup.modelSelected.connect(self.setCurrentModel)
        self._popup.current_model = self._current_model
        self._popup.active_series = self._popup._detect_series(self._current_model)
        self._popup.apply_theme(self._is_dark)
        self._popup.show_at(self)

    def currentText(self) -> str:
        return self._current_model

    def currentData(self) -> str:
        return self._current_model

    def setCurrentModel(self, model_id: str):
        if self._current_model != model_id:
            self._current_model = model_id
            self._update_display()
            self.currentTextChanged.emit(model_id)
            self.currentIndexChanged.emit(0)

    def setCurrentText(self, model_id: str):
        self.setCurrentModel(model_id)

    def reset_to_default(self):
        task_data = MODEL_CATALOG.get(self.task_type, MODEL_CATALOG["detect"])
        self.setCurrentModel(task_data["default"])

    def clear(self):
        pass

    def apply_theme(self, is_dark: bool):
        self._is_dark = is_dark
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#ModelPickerCard {
                    background-color: #141416;
                    border: 1px solid #27272a;
                    border-radius: 8px;
                }
                QFrame#ModelPickerCard:hover {
                    border-color: #3f3f46;
                    background-color: #1a1a1e;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.lbl_text, "color: #f4f4f5; background: transparent; border: none;")
            self.lbl_caret.setPixmap(PhCaretDown.pixmap(size=14, color="#a1a1aa", weight="bold", dpr=2.0))
        else:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#ModelPickerCard {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                }
                QFrame#ModelPickerCard:hover {
                    border-color: #cbd5e1;
                    background-color: #f8fafc;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.lbl_text, "color: #0f172a; background: transparent; border: none;")
            self.lbl_caret.setPixmap(PhCaretDown.pixmap(size=14, color="#64748b", weight="bold", dpr=2.0))

        if self._popup:
            self._popup.apply_theme(is_dark)


class InfoTooltipPopover(QWidget):
    """
    Speech-bubble tooltip popover with pointer arrow.
    Benchmarked strictly against modern cloud platforms (Ultralytics HUB, etc.).
    - Light Mode: Pitch-black bubble (#0f0f11), white text, pointing down to icon
    - Dark Mode: Crisp-white bubble (#ffffff), black text, pointing down to icon
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InfoTooltipPopover()
        return cls._instance

    @classmethod
    def show_popover(cls, target_widget: QWidget, text: str, is_dark: bool):
        if not text:
            return
        inst = cls.get_instance()
        inst.set_content(text, is_dark)
        inst.adjustSize()

        pop_w = inst.width()
        pop_h = inst.height()

        target_center_x = target_widget.mapToGlobal(QPoint(target_widget.width() // 2, 0)).x()
        target_top_y = target_widget.mapToGlobal(QPoint(0, 0)).y()
        target_bot_y = target_top_y + target_widget.height()

        screen = target_widget.screen() or QApplication.primaryScreen()
        screen_avail = screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)

        arrow_position = 'bottom'
        pos_y = target_top_y - pop_h - 2

        if pos_y < screen_avail.top() + 10:
            pos_y = target_bot_y + 2
            arrow_position = 'top'

        pos_x = target_center_x - pop_w // 2
        pos_x = max(screen_avail.left() + 10, min(screen_avail.right() - pop_w - 10, pos_x))

        inst.set_arrow_position(arrow_position)
        inst.move(pos_x, pos_y)
        inst.set_arrow_global_x(target_center_x)
        inst.show()
        inst.raise_()

    @classmethod
    def hide_popover(cls):
        if cls._instance and cls._instance.isVisible():
            cls._instance.hide()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._arrow_x = 0.5
        self._arrow_position = 'bottom'
        self._is_dark_theme = False
        self._padding = 10

        self.box_layout = QVBoxLayout(self)
        self.box_layout.setContentsMargins(self._padding + 14, self._padding + 9, self._padding + 14, 5 + 11)

        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.box_layout.addWidget(self.label)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(16)
        self.shadow.setOffset(0, 4)
        self.setGraphicsEffect(self.shadow)

    def set_arrow_position(self, pos: str):
        self._arrow_position = pos
        p = self._padding
        if pos == 'bottom':
            self.box_layout.setContentsMargins(p + 14, p + 9, p + 14, 5 + 11)
        else:
            self.box_layout.setContentsMargins(p + 14, 5 + 11, p + 14, p + 9)

    def set_content(self, text: str, is_dark_theme: bool = False):
        self._is_dark_theme = is_dark_theme
        self.label.setText(text)

        font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif'
        if is_dark_theme:
            _MkThemeEngine.apply_style_sheet(self.label, f"""
                QLabel {{
                    color: #09090b;
                    font-family: {font_family};
                    font-size: 13px;
                    line-height: 1.45;
                    font-weight: 400;
                    background: transparent;
                }}
            """)
            self.shadow.setColor(QColor(0, 0, 0, 90))
        else:
            _MkThemeEngine.apply_style_sheet(self.label, f"""
                QLabel {{
                    color: #ffffff;
                    font-family: {font_family};
                    font-size: 13px;
                    line-height: 1.45;
                    font-weight: 400;
                    background: transparent;
                }}
            """)
            self.shadow.setColor(QColor(0, 0, 0, 70))

        # Dynamic width matching the reference card aspect ratio (360-380px for standard explanations)
        fm = self.label.fontMetrics()
        single_line_w = fm.horizontalAdvance(text)
        if single_line_w < 300:
            target_w = max(180, single_line_w + 10)
        elif single_line_w < 550:
            target_w = 320
        else:
            target_w = 380
        self.label.setFixedWidth(target_w)

        self.adjustSize()

    def set_arrow_global_x(self, target_global_x: int):
        local_x = target_global_x - self.x()
        p = self._padding
        min_x = p + 16
        max_x = self.width() - p - 16
        self._arrow_x = max(min_x, min(max_x, local_x))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        p = self._padding
        arrow_h = 5.0
        arrow_w = 10.0

        if self._arrow_position == 'bottom':
            body_rect = QRectF(p, p, self.width() - 2 * p, self.height() - p - arrow_h - 1)
            ax = getattr(self, '_arrow_x', self.width() / 2.0)
            arrow = QPolygonF([
                QPointF(ax - arrow_w / 2.0, body_rect.bottom()),
                QPointF(ax + arrow_w / 2.0, body_rect.bottom()),
                QPointF(ax, self.height() - 1)
            ])
        else:
            body_rect = QRectF(p, arrow_h + 1, self.width() - 2 * p, self.height() - p - arrow_h - 1)
            ax = getattr(self, '_arrow_x', self.width() / 2.0)
            arrow = QPolygonF([
                QPointF(ax - arrow_w / 2.0, body_rect.top()),
                QPointF(ax + arrow_w / 2.0, body_rect.top()),
                QPointF(ax, 1)
            ])

        path = QPainterPath()
        path.addRoundedRect(body_rect, 9.0, 9.0)

        arrow_path = QPainterPath()
        arrow_path.addPolygon(arrow)

        bubble_path = path.united(arrow_path).simplified()

        if self._is_dark_theme:
            painter.fillPath(bubble_path, QBrush(QColor("#ffffff")))
            painter.strokePath(bubble_path, QPen(QColor(0, 0, 0, 20), 1))
        else:
            painter.fillPath(bubble_path, QBrush(QColor("#0f0f11")))
            painter.strokePath(bubble_path, QPen(QColor(255, 255, 255, 25), 1))


class InfoIconButton(QLabel):
    """
    Hoverable vector info icon with circular badge hover effect and speech-bubble popover.
    Directly benchmarked against modern AI interfaces (Ultralytics HUB, etc.).
    """
    def __init__(self, hint_text: str = "", parent=None):
        super().__init__(parent)
        self.hint_text = hint_text
        self.setFixedSize(20, 20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._is_hovered = False
        self._is_dark = False
        self._update_appearance()

    def set_theme(self, is_dark: bool):
        self._is_dark = is_dark
        self._update_appearance()

    def set_hint(self, text: str):
        self.hint_text = text

    def _update_appearance(self):
        if self._is_dark:
            color = "#f4f4f5" if self._is_hovered else "#71717a"
            bg = "background-color: rgba(255, 255, 255, 0.12); border-radius: 10px;" if self._is_hovered else "background: transparent; border: none;"
        else:
            color = "#09090b" if self._is_hovered else "#94a3b8"
            bg = "background-color: rgba(0, 0, 0, 0.06); border-radius: 10px;" if self._is_hovered else "background: transparent; border: none;"

        _MkThemeEngine.apply_style_sheet(self, f"QLabel {{ {bg} }}")
        self.setPixmap(PhInfo.pixmap(size=13, color=color, weight="regular", dpr=3.0))

    def enterEvent(self, event):
        if not self._is_hovered:
            self._is_hovered = True
            self._update_appearance()
            if self.hint_text:
                InfoTooltipPopover.show_popover(self, self.hint_text, self._is_dark)
        super().enterEvent(event)

    def leaveEvent(self, event):
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(cursor_pos):
            return
        self._is_hovered = False
        self._update_appearance()
        InfoTooltipPopover.hide_popover()
        super().leaveEvent(event)


class AdvHeaderBar(QFrame):
    """Ultralytics-style interactive accordion header for advanced settings."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AdvHeaderBar")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.is_expanded = False
        self.setFixedHeight(38)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 12, 4)
        lay.setSpacing(8)

        # Gear Icon
        self.icon_gear = QLabel()
        lay.addWidget(self.icon_gear)

        # Title
        self.lbl_title = QLabel("超参数设置")
        lay.addWidget(self.lbl_title)

        # Subtle pill badge
        self.tag_status = QLabel("49 项")
        lay.addWidget(self.tag_status)

        lay.addStretch()

        # Chevron arrow on the right
        self.icon_arrow = QLabel()
        lay.addWidget(self.icon_arrow)

        self.apply_theme()

    def apply_theme(self):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#AdvHeaderBar {
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                }
                QFrame#AdvHeaderBar:hover {
                    background-color: #27272a;
                }
                QLabel {
                    background: transparent;
                }
            """)
            self.icon_gear.setPixmap(PhGearSix.pixmap(size=16, color="#f4f4f5", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 13px; font-weight: 600; color: #f4f4f5;")
            _MkThemeEngine.apply_style_sheet(self.tag_status, """
                background-color: #27272a;
                color: #a1a1aa;
                font-size: 11px;
                font-weight: 500;
                padding: 2px 7px;
                border-radius: 4px;
            """)
            self._arrow_color = "#a1a1aa"
        else:
            _MkThemeEngine.apply_style_sheet(self, """
                QFrame#AdvHeaderBar {
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                }
                QFrame#AdvHeaderBar:hover {
                    background-color: #f4f4f5;
                }
                QLabel {
                    background: transparent;
                }
            """)
            self.icon_gear.setPixmap(PhGearSix.pixmap(size=16, color="#18181b", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_title, "font-size: 13px; font-weight: 600; color: #18181b;")
            _MkThemeEngine.apply_style_sheet(self.tag_status, """
                background-color: #f4f4f5;
                color: #71717a;
                font-size: 11px;
                font-weight: 500;
                padding: 2px 7px;
                border-radius: 4px;
            """)
            self._arrow_color = "#71717a"
        self.set_expanded(self.is_expanded)

    def set_expanded(self, expanded: bool):
        self.is_expanded = expanded
        color = getattr(self, "_arrow_color", "#71717a")
        if expanded:
            self.icon_arrow.setPixmap(PhCaretUp.pixmap(size=14, color=color, weight="bold", dpr=3.0))
        else:
            self.icon_arrow.setPixmap(PhCaretDown.pixmap(size=14, color=color, weight="bold", dpr=3.0))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

class TrainCreateDialog(QDialog):
    task_started = Signal(int)  # emits task_id upon successful launch
    model_inspected = Signal(str, object, bool)

    def __init__(self, project_info, db_manager, task_manager, parent=None):
        super().__init__(parent)
        self.project = project_info
        self.task_type = project_info.get("task_type", "detect")
        if self.task_type not in MODEL_CATALOG:
            self.task_type = "detect"
        self.db = db_manager
        self.task_mgr = task_manager
        self.created_task_id = None
        self._custom_model_signature = None
        self.model_inspected.connect(self._on_custom_model_inspected)
        self._parent_widget = parent
        self._top_window = parent.window() if parent else None

        self.field_header_labels = []
        self.input_field_labels = []
        self.info_icons = []
        self.section_header_labels = []
        self.section_lines = []

        self.setWindowTitle(f"创建训练任务 - {self.project['name']}")
        # Frameless modern modal window (removes native OS titlebar)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        _MkThemeEngine.apply_style_sheet(self, "QDialog { background: transparent; background-color: transparent; border: none; }")

        self._is_closing = False
        self._is_showing_sub_dialog = False

        if self._parent_widget:
            self._parent_widget.installEventFilter(self)
        if self._top_window:
            self._top_window.installEventFilter(self)

        self._init_ui()
        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)

    def _get_target_card_width(self):
        if self._parent_widget and self._parent_widget.isVisible():
            return min(840, max(400, self._parent_widget.width() - 40))
        return 840

    def _get_target_heights(self):
        if self._parent_widget and self._parent_widget.isVisible():
            avail_h = self._parent_widget.height()
            collapsed = min(800, max(300, avail_h - 40))
            expanded = min(980, max(300, avail_h - 40))
        else:
            screen = self.screen() or QApplication.primaryScreen()
            if screen:
                avail_h = screen.availableGeometry().height()
                collapsed = min(800, avail_h - 60)
                expanded = min(980, avail_h - 60)
            else:
                collapsed = 800
                expanded = 980
        return collapsed, expanded

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_geometry_with_parent()

    def _sync_geometry_with_parent(self):
        if not self._parent_widget or not self._parent_widget.isVisible():
            return
        global_tl = self._parent_widget.mapToGlobal(QPoint(0, 0))
        p_w = self._parent_widget.width()
        p_h = self._parent_widget.height()
        self.setGeometry(global_tl.x(), global_tl.y(), p_w, p_h)

        if hasattr(self, 'container_card') and self.container_card:
            card_w = self._get_target_card_width()
            collapsed_h, expanded_h = self._get_target_heights()
            is_expanded = hasattr(self, 'adv_container') and self.adv_container.isVisible()
            target_h = expanded_h if is_expanded else collapsed_h
            if not getattr(self, '_card_anim', None) or self._card_anim.state() != QVariantAnimation.State.Running:
                self.container_card.setFixedSize(card_w, target_h)

    def _animate_card_height(self, start_h, end_h, on_finished=None):
        if hasattr(self, '_card_anim') and self._card_anim:
            self._card_anim.stop()
        self._card_anim = QVariantAnimation(self)
        self._card_anim.setDuration(220)
        self._card_anim.setStartValue(start_h)
        self._card_anim.setEndValue(end_h)
        self._card_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        card_w = self.container_card.width()
        def _on_val_changed(val):
            if hasattr(self, 'container_card') and self.container_card:
                self.container_card.setFixedSize(card_w, int(val))
        self._card_anim.valueChanged.connect(_on_val_changed)
        if on_finished:
            self._card_anim.finished.connect(on_finished)
        self._card_anim.start()

    def _animate_geometry(self, start_geom, end_geom, on_finished=None):
        if hasattr(self, '_resize_anim') and self._resize_anim:
            self._resize_anim.stop()
        self._resize_anim = QPropertyAnimation(self, b"geometry")
        self._resize_anim.setDuration(220)
        self._resize_anim.setStartValue(start_geom)
        self._resize_anim.setEndValue(end_geom)
        self._resize_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        if on_finished:
            self._resize_anim.finished.connect(on_finished)
        self._resize_anim.start()

    def _remove_event_filter(self):
        if self._parent_widget:
            try:
                self._parent_widget.removeEventFilter(self)
            except Exception:
                pass
        if getattr(self, '_top_window', None):
            try:
                self._top_window.removeEventFilter(self)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        if getattr(self, '_is_closing', False):
            return super().eventFilter(obj, event)
        if obj in (self._parent_widget, getattr(self, '_top_window', None)):
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
                self._sync_geometry_with_parent()
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
        painter.end()



    def _create_field_header(self, text, hint=""):
        w = QWidget()
        _MkThemeEngine.apply_style_sheet(w, "background: transparent; border: none;")
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        lbl = QLabel(text)
        self.field_header_labels.append(lbl)
        l.addWidget(lbl)
        if hint:
            ic = InfoIconButton(hint)
            self.info_icons.append(ic)
            l.addWidget(ic)
        l.addStretch()
        return w

    def _create_section_divider(self, title, action_widget=None):
        w = QWidget()
        _MkThemeEngine.apply_style_sheet(w, "background: transparent; border: none;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 12, 0, 4)
        lay.setSpacing(10)
        lbl = QLabel(title)
        self.section_header_labels.append(lbl)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        self.section_lines.append(line)
        lay.addWidget(lbl)
        lay.addWidget(line, stretch=1)
        if action_widget:
            lay.addWidget(action_widget)
        return w

    def _create_input_field(self, label_text, default_val, hint=""):
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        header = QWidget()
        _MkThemeEngine.apply_style_sheet(header, "background: transparent; border: none;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(0, 0, 0, 0)
        h_lay.setSpacing(4)
        lbl = QLabel(label_text)
        self.input_field_labels.append(lbl)
        h_lay.addWidget(lbl)
        if hint:
            ic = InfoIconButton(hint)
            self.info_icons.append(ic)
            h_lay.addWidget(ic)
        h_lay.addStretch()
        box.addWidget(header)
        
        inp = MkInput()
        inp.setText(str(default_val))
        _MkThemeEngine.apply_style_sheet(inp, "font-size: 12px; padding: 4px 8px;")
        box.addWidget(inp)
        
        container = QWidget()
        _MkThemeEngine.apply_style_sheet(container, "background: transparent; border: none;")
        container.setLayout(box)
        return container, inp

    def _create_switch_field(self, label_text, default_checked=False, hint=""):
        w = QWidget()
        _MkThemeEngine.apply_style_sheet(w, "background: transparent; border: none;")
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 4, 4, 4)
        h.setSpacing(6)
        lbl = QLabel(label_text)
        self.input_field_labels.append(lbl)
        h.addWidget(lbl)
        if hint:
            ic = InfoIconButton(hint)
            self.info_icons.append(ic)
            h.addWidget(ic)
        h.addStretch()
        sw = MkSwitch(checked=default_checked)
        h.addWidget(sw)
        return w, sw

    def _get_next_available_task_name(self) -> str:
        """
        Scan both the SQLite database tasks and the filesystem under runs/<task_type>/
        to calculate the next available auto-incremented experiment name (e.g. exp-1, exp-2, exp-3...).
        """
        from TrainingPlatform.core.env_manager import get_workspace_root
        workspace = get_workspace_root()
        task_type = self.task_type
        runs_task_dir = os.path.join(workspace, "runs", task_type)

        existing_names = set()

        # 1. Collect from database
        if hasattr(self, "db") and self.db:
            try:
                db_tasks = self.db.get_tasks(project_id=self.project['id'])
                for t in db_tasks:
                    tn = (t.get('task_name') or '').strip()
                    if tn:
                        existing_names.add(tn)
            except Exception:
                pass

        # 2. Collect from disk runs/<task_type>/
        if os.path.exists(runs_task_dir) and os.path.isdir(runs_task_dir):
            try:
                for entry in os.listdir(runs_task_dir):
                    if os.path.isdir(os.path.join(runs_task_dir, entry)):
                        existing_names.add(entry)
            except Exception:
                pass

        # 3. Find highest numeric suffix among exp-N, exp_N, expN
        max_idx = 0
        import re
        for name in existing_names:
            if re.match(r"^exp", name, re.IGNORECASE):
                for n_str in re.findall(r"\d+", name):
                    try:
                        num = int(n_str)
                        if num > max_idx:
                            max_idx = num
                    except ValueError:
                        pass

        next_idx = max_idx + 1
        while f"exp-{next_idx}" in existing_names or os.path.exists(os.path.join(runs_task_dir, f"exp-{next_idx}")):
            next_idx += 1

        return f"exp-{next_idx}"

    def _init_ui(self):
        # Root layout for overlay centering
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Dialog main container card with rounded corners & shadow
        self.container_card = QFrame(self)
        self.container_card.setObjectName("DialogContainer")
        _MkThemeEngine.apply_style_sheet(self.container_card, """
            QFrame#DialogContainer {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 48))
        shadow.setOffset(0, 6)
        self.container_card.setGraphicsEffect(shadow)
        self.root_layout.addWidget(self.container_card)

        card_w = self._get_target_card_width()
        collapsed_h, _ = self._get_target_heights()
        self.container_card.setFixedSize(card_w, collapsed_h)
        if not self._parent_widget:
            self.resize(card_w + 40, collapsed_h + 40)

        main_layout = QVBoxLayout(self.container_card)
        main_layout.setContentsMargins(28, 22, 28, 20)
        main_layout.setSpacing(12)

        # ── 1. Top Custom Header (Chinese First) ──
        self.header_widget = QWidget()
        self.header_widget.setObjectName("HeaderWidget")
        _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_lbl = QLabel("创建训练任务")
        header_layout.addWidget(self.title_lbl)

        self.arrow_icon = QLabel()
        self.arrow_icon.setPixmap(PhArrowUpRight.pixmap(size=14, color="#2563eb", weight="bold", dpr=3.0))
        _MkThemeEngine.apply_style_sheet(self.arrow_icon, "background: transparent; border: none;")
        header_layout.addWidget(self.arrow_icon)

        tag_text = self.task_type
        task_names_cn = {
            "detect": "目标检测 (detect)",
            "segment": "实例分割 (segment)",
            "classify": "图像分类 (classify)",
            "pose": "姿态估计 (pose)",
            "obb": "旋转检测 (obb)"
        }
        self.tag_lbl = QLabel(task_names_cn.get(tag_text.lower(), tag_text))
        header_layout.addWidget(self.tag_lbl)
        header_layout.addStretch()

        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        header_layout.addWidget(self.btn_close)

        main_layout.addWidget(self.header_widget)

        self.subtitle_lbl = QLabel(f"在项目 [{self.project['name']}] 下配置并启动本地 YOLO 视觉模型训练。")
        _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "background: transparent; background-color: transparent; border: none;")
        main_layout.addWidget(self.subtitle_lbl)

        # ── Scroll Area for Form Body (Clean floating scrollbar, zero Windows arrows) ──
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().valueChanged.connect(lambda: InfoTooltipPopover.hide_popover())

        self.scroll_container = QWidget()
        self.scroll_container.setObjectName("ScrollContentContainer")
        c_layout = QVBoxLayout(self.scroll_container)
        c_layout.setContentsMargins(0, 4, 8, 4)
        c_layout.setSpacing(16)
        c_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── 2. Top Grid: Base Model (Left) & Dataset (Right) ──
        top_grid = QHBoxLayout()
        top_grid.setSpacing(18)

        # Left Column: Base Model
        col_model = QVBoxLayout()
        col_model.setSpacing(6)
        col_model.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_model_header = QWidget()
        _MkThemeEngine.apply_style_sheet(lbl_model_header, "background: transparent; border: none;")
        l_mh = QHBoxLayout(lbl_model_header)
        l_mh.setContentsMargins(0, 0, 0, 0)
        l_mh.setSpacing(6)
        self.ic_cube = QLabel()
        self.lbl_model_txt = QLabel("基础模型 (Base Model)")
        l_mh.addWidget(self.ic_cube)
        l_mh.addWidget(self.lbl_model_txt)
        ic_m_info = InfoIconButton("选择预训练模型架构 (.pt) 或指定本地权重。YOLO 官方架构涵盖 detect、segment、pose、obb 等任务")
        self.info_icons.append(ic_m_info)
        l_mh.addWidget(ic_m_info)
        l_mh.addStretch()
        col_model.addWidget(lbl_model_header)

        self.combo_task_type = MkComboBox()
        for title, task in (("目标检测 Detect", "detect"), ("实例分割 Segment", "segment"),
                            ("关键点 Pose", "pose"), ("图像分类 Classify", "classify"), ("旋转框 OBB", "obb")):
            self.combo_task_type.addItem(title, task)
        self.combo_task_type.setCurrentIndex(self.combo_task_type.findData(self.task_type))
        col_model.addWidget(self.combo_task_type)
        self.combo_model = ModelPickerWidget(task_type=self.task_type, parent=self)
        self.combo_task_type.currentIndexChanged.connect(self._on_task_type_changed)
        col_model.addWidget(self.combo_model)

        # Custom weight row
        box_model_custom = QHBoxLayout()
        box_model_custom.setSpacing(6)
        self.input_custom_model = MkInput(placeholder="或指定本地自定义 .pt 权重路径...")
        _MkThemeEngine.apply_style_sheet(self.input_custom_model, "font-size: 12px;")
        btn_browse_model = MkButton("浏览权重...", btn_type="default")
        btn_browse_model.setIcon(PhFolderOpen.regular(size=13))
        _MkThemeEngine.apply_style_sheet(btn_browse_model, "font-size: 12px; padding: 4px 10px;")
        btn_browse_model.clicked.connect(self._browse_custom_model)
        box_model_custom.addWidget(self.input_custom_model, stretch=1)
        box_model_custom.addWidget(btn_browse_model)
        col_model.addLayout(box_model_custom)

        # Subtle model info card to balance vertical height with dataset code editor
        self.card_model_tip = QFrame()
        self.card_model_tip.setObjectName("ModelTipCard")
        tip_lay = QVBoxLayout(self.card_model_tip)
        tip_lay.setContentsMargins(10, 8, 10, 8)
        tip_lay.setSpacing(2)
        self.lbl_tip_title = QLabel("预训练权重与迁移学习")
        self.lbl_tip_sub = QLabel("支持官方 YOLO 架构权重与自定义检查点断点续训。")
        tip_lay.addWidget(self.lbl_tip_title)
        tip_lay.addWidget(self.lbl_tip_sub)
        col_model.addWidget(self.card_model_tip)

        top_grid.addLayout(col_model, stretch=4)

        # Right Column: Dataset (Code Editor & Upload)
        col_dataset = QVBoxLayout()
        col_dataset.setSpacing(6)
        col_dataset.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_ds_header = QWidget()
        _MkThemeEngine.apply_style_sheet(lbl_ds_header, "background: transparent; border: none;")
        l_dh = QHBoxLayout(lbl_ds_header)
        l_dh.setContentsMargins(0, 0, 0, 0)
        l_dh.setSpacing(6)
        self.ic_database = QLabel()
        self.lbl_dataset_txt = QLabel("训练数据集配置 (Dataset)")
        l_dh.addWidget(self.ic_database)
        l_dh.addWidget(self.lbl_dataset_txt)
        ic_d_info = InfoIconButton("配置用于模型训练的数据集。可直接在下方编辑 YAML 必填模板，或点击【上传 YAML...】自动解析并提取模板参数。")
        self.info_icons.append(ic_d_info)
        l_dh.addWidget(ic_d_info)
        l_dh.addStretch()

        self.btn_upload_yaml = QPushButton("上传 YAML...")
        self.btn_upload_yaml.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload_yaml.clicked.connect(self._upload_and_parse_yaml)
        l_dh.addWidget(self.btn_upload_yaml)

        self.btn_reset_yaml = QPushButton("重置模板")
        self.btn_reset_yaml.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_yaml.clicked.connect(self._reset_yaml_template)
        l_dh.addWidget(self.btn_reset_yaml)

        col_dataset.addWidget(lbl_ds_header)

        # Integrated YAML Code Editor
        self.yaml_editor = YamlCodeEditor(self, task_type=self.task_type)
        col_dataset.addWidget(self.yaml_editor)

        top_grid.addLayout(col_dataset, stretch=5)
        c_layout.addLayout(top_grid)

        # ── 3. Core 4 Hyperparameters (4 Columns in 1 Row) ──
        default_task_name = self._get_next_available_task_name()

        row_params = QHBoxLayout()
        row_params.setSpacing(12)
        row_params.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Col 1: Epochs ⓘ
        v_ep = QVBoxLayout()
        v_ep.setSpacing(6)
        v_ep.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_ep.addWidget(self._create_field_header("训练轮数 (Epochs)", "设置在训练数据集上的完整迭代轮数上限。建议以 100 轮作为基准基线；如果验证集指标仍在持续改善可适当调大，配合早停机制 (Patience) 可自动在指标停滞时提前终止。"))
        self.input_epochs = MkInput()
        self.input_epochs.setText(str(DEFAULT_TRAIN_SETTINGS["epochs"]))
        v_ep.addWidget(self.input_epochs)
        row_params.addLayout(v_ep, stretch=1)

        # Col 2: Batch Size ⓘ
        v_bs = QVBoxLayout()
        v_bs.setSpacing(6)
        v_bs.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_bs.addWidget(self._create_field_header("批次大小 (Batch)", "单批次训练样本数。设为 auto 将根据当前 GPU 可用显存动态分配最大吞吐批次；亦可指定固定数值 (如 16, 32, 64)。"))
        self.input_batch = MkInput()
        self.input_batch.setText(str(DEFAULT_TRAIN_SETTINGS["batch"]))
        v_bs.addWidget(self.input_batch)
        row_params.addLayout(v_bs, stretch=1)

        # Col 3: Image Size ⓘ
        v_img = QVBoxLayout()
        v_img.setSpacing(6)
        v_img.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_img.addWidget(self._create_field_header("图像尺寸 (imgsz)", "输入图像分辨率 (imgsz)。训练和验证使用的正方形尺寸，默认 640"))
        self.combo_imgsz = RoundedComboBox(is_dark=ThemeEngine.is_dark())
        self.combo_imgsz.addItems(["640", "320", "416", "512", "1024", "1280"])
        self.combo_imgsz.setCurrentText(str(DEFAULT_TRAIN_SETTINGS["imgsz"]))
        v_img.addWidget(self.combo_imgsz)
        row_params.addLayout(v_img, stretch=1)

        # Col 4: Task Name ⓘ
        v_nm = QVBoxLayout()
        v_nm.setSpacing(6)
        v_nm.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_nm.addWidget(self._create_field_header("实验名称 (Name)", "为这次训练运行命名。留空则自动生成名称。"))
        self.input_task_name = MkInput(placeholder=f"留空则自动递增 (当前为 {default_task_name})")
        self.input_task_name.setText("")
        v_nm.addWidget(self.input_task_name)
        row_params.addLayout(v_nm, stretch=1)

        c_layout.addLayout(row_params)

        # ── 4. Collapsible Advanced Settings (Unified Continuous Flow) ──
        self.adv_header_bar = AdvHeaderBar()
        self.adv_header_bar.clicked.connect(self._toggle_advanced_settings)
        c_layout.addWidget(self.adv_header_bar)

        # Advanced container (Unified view without navigation bar) - Lazy initialized
        self.adv_container = QFrame()
        self.adv_container.setObjectName("AdvContainer")
        self.adv_container.setVisible(False)
        c_layout.addWidget(self.adv_container)
        self._adv_built = False


        # ── 5. Device & Environment Summary Card ──
        self.card_device = QFrame()
        self.card_device.setObjectName("DeviceCard")
        l_dev = QVBoxLayout(self.card_device)
        l_dev.setContentsMargins(22, 14, 18, 14)
        l_dev.setSpacing(10)

        dev_top = QHBoxLayout()
        dev_top.setSpacing(8)

        # Tab Segmented Pill (referencing Fig 2)
        self.pill_toggle = SegmentedPillToggle([
            (PhDesktop, "本地单机训练"),
            (PhSliders, "运行环境选择")
        ], self.card_device, is_dark=ThemeEngine.is_dark())
        dev_top.addWidget(self.pill_toggle)
        dev_top.addStretch()

        # Device combo
        self.lbl_target_dev = QLabel("计算加速设备:")
        _MkThemeEngine.apply_style_sheet(self.lbl_target_dev, "background: transparent; border: none;")
        self.combo_device = RoundedComboBox(self.card_device, is_dark=ThemeEngine.is_dark())
        self.combo_device.addItems(["0 (GPU 0)", "CPU"])
        self.combo_device.setFixedWidth(185)
        dev_top.addWidget(self.lbl_target_dev)
        dev_top.addWidget(self.combo_device)

        l_dev.addLayout(dev_top)

        # ── Stacked Pages: Page 0 = Terminal CLI, Page 1 = Environment Selector & Console ──
        self.device_stack = QStackedWidget(self.card_device)

        # Page 0: Terminal Command Display Block (Read-only, copyable)
        self.page_cli = QWidget()
        l_cli = QVBoxLayout(self.page_cli)
        l_cli.setContentsMargins(0, 2, 0, 0)
        l_cli.setSpacing(6)

        cli_header = QHBoxLayout()
        cli_header.setSpacing(8)
        self.lbl_cli_title = QLabel("终端运行指令 (Run in terminal)")
        _MkThemeEngine.apply_style_sheet(self.lbl_cli_title, "font-size: 12px; font-weight: 600; color: #64748b; background: transparent; border: none;")
        cli_header.addWidget(self.lbl_cli_title)
        cli_header.addStretch()

        self.combo_cli_shell = RoundedComboBox(self.page_cli, is_dark=ThemeEngine.is_dark())
        self.combo_cli_shell.setFixedWidth(135)
        self.combo_cli_shell.addItems(["单行 (推荐/通用)", "CMD (^)", "PowerShell (`)", "Bash (\\)"])
        self.combo_cli_shell.currentIndexChanged.connect(self._update_cli_command)
        cli_header.addWidget(self.combo_cli_shell)

        l_cli.addLayout(cli_header)

        self.cli_viewer = CliCommandViewer(self.page_cli)
        if hasattr(self, "yaml_editor") and hasattr(self.yaml_editor, "sync_to_local_files"):
            self.cli_viewer._on_copy_hook = self.yaml_editor.sync_to_local_files
        l_cli.addWidget(self.cli_viewer)
        self.device_stack.addWidget(self.page_cli)

        # Page 1: Reusable Environment Selector Card
        self.env_selector = EnvironmentSelectorWidget(self.card_device, is_dark=ThemeEngine.is_dark())
        self.env_selector.env_changed.connect(self._on_env_changed)
        self.device_stack.addWidget(self.env_selector)

        l_dev.addWidget(self.device_stack)

        # Wire toggle to stack
        self.pill_toggle.tab_changed.connect(self.device_stack.setCurrentIndex)

        self.scroll.setWidget(self.scroll_container)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll_container.setAutoFillBackground(False)
        main_layout.addWidget(self.scroll, stretch=1)
        main_layout.addWidget(self.card_device)

        # ── 6. Bottom Action Bar ──
        self.footer_widget = QFrame()
        self.footer_widget.setObjectName("DialogFooter")
        btn_layout = QHBoxLayout(self.footer_widget)
        btn_layout.setContentsMargins(0, 12, 0, 0)
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_start = QPushButton("开始训练")
        self.btn_start.setIcon(PhPlay.fill(size=12, color="#ffffff"))
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.handle_start_training)
        btn_layout.addWidget(self.btn_start)

        main_layout.addWidget(self.footer_widget)

        self._connect_cli_command_signals()
        self._update_cli_command()

    def _ensure_adv_built(self):
        """Lazily initialize advanced hyperparameter inputs on first expansion."""
        if getattr(self, "_adv_built", False):
            return
        self._adv_built = True

        _MkThemeEngine.apply_style_sheet(self.adv_container, """
            QFrame#AdvContainer {
                background-color: transparent;
                border: none;
            }
            QFrame#AdvContainer QWidget {
                background: transparent;
            }
        """)
        l_adv = QVBoxLayout(self.adv_container)
        l_adv.setContentsMargins(0, 8, 0, 8)
        l_adv.setSpacing(12)

        # Reset button inside advanced container
        self.btn_reset_defaults = QPushButton("恢复默认参数")
        is_dark = ThemeEngine.is_dark()
        self.btn_reset_defaults.setIcon(PhArrowCounterClockwise.regular(size=12, color="#cbd5e1" if is_dark else "#64748b"))
        self.btn_reset_defaults.setToolTip("恢复官方推荐默认超参数配置")
        self.btn_reset_defaults.setCursor(Qt.CursorShape.PointingHandCursor)
        if is_dark:
            _MkThemeEngine.apply_style_sheet(self.btn_reset_defaults, """
                QPushButton {
                    border: 1px solid #3f3f46;
                    background-color: #27272a;
                    color: #cbd5e1;
                    font-size: 12px;
                    font-weight: 500;
                    border-radius: 6px;
                    padding: 3px 10px;
                }
                QPushButton:hover {
                    background-color: #3f3f46;
                    color: #ffffff;
                    border-color: #52525b;
                }
            """)
        else:
            _MkThemeEngine.apply_style_sheet(self.btn_reset_defaults, """
                QPushButton {
                    border: 1px solid #e2e8f0;
                    background-color: #ffffff;
                    color: #64748b;
                    font-size: 12px;
                    font-weight: 500;
                    border-radius: 6px;
                    padding: 3px 10px;
                }
                QPushButton:hover {
                    background-color: #f8fafc;
                    color: #0f172a;
                    border-color: #cbd5e1;
                }
            """)
        self.btn_reset_defaults.clicked.connect(self.reset_to_defaults)

        # ── Section 1: 优化器与学习率 ──
        l_adv.addWidget(self._create_section_divider(
            "一、 优化器与学习率配置 (Optimizer & LR)", 
            action_widget=self.btn_reset_defaults
        ))
        grid_opt = QGridLayout()
        grid_opt.setHorizontalSpacing(14)
        grid_opt.setVerticalSpacing(8)

        # Optimizer
        box_opt = QVBoxLayout()
        box_opt.setContentsMargins(0, 0, 0, 0)
        box_opt.setSpacing(4)
        header_opt = self._create_field_header("优化器算法 (optimizer)", "选择优化器算法。可选 auto, SGD, MuSGD, Adam, AdamW, NAdam, RAdam, RMSProp 等，auto 会根据模型自动选择")
        self.combo_optimizer = RoundedComboBox(is_dark=is_dark)
        self.combo_optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp", "NAdam"])
        self.combo_optimizer.setCurrentText(str(DEFAULT_TRAIN_SETTINGS["optimizer"]))
        box_opt.addWidget(header_opt)
        box_opt.addWidget(self.combo_optimizer)
        w_opt = QWidget()
        _MkThemeEngine.apply_style_sheet(w_opt, "background: transparent; border: none;")
        w_opt.setLayout(box_opt)
        grid_opt.addWidget(w_opt, 0, 0)

        w, self.input_lr0 = self._create_input_field("初始学习率 (lr0)", DEFAULT_TRAIN_SETTINGS["lr0"], "初始学习率 (lr0)。SGD 推荐 0.01，Adam/AdamW 推荐 0.001")
        grid_opt.addWidget(w, 0, 1)

        w, self.input_lrf = self._create_input_field("最终学习率比例 (lrf)", DEFAULT_TRAIN_SETTINGS["lrf"], "最终学习率比率 (lrf)。最终学习率 = lr0 * lrf，默认 0.01")
        grid_opt.addWidget(w, 0, 2)

        w, self.input_momentum = self._create_input_field("动量系数 (momentum)", DEFAULT_TRAIN_SETTINGS["momentum"], "动量系数 (momentum)。控制梯度更新的惯性大小（对应 SGD 的动量系数，或 Adam/AdamW 算法中的 β1 衰减参数），默认 0.937")
        grid_opt.addWidget(w, 1, 0)

        w, self.input_weight_decay = self._create_input_field("权重衰减 (weight_decay)", DEFAULT_TRAIN_SETTINGS["weight_decay"], "权重衰减系数 (weight_decay)。L2 正则化权重衰减系数，防止过拟合，默认 0.0005")
        grid_opt.addWidget(w, 1, 1)

        w_sw_cos, self.sw_cos_lr = self._create_switch_field("余弦退火调度 (cos_lr)", DEFAULT_TRAIN_SETTINGS["cos_lr"], "余弦退火学习率调度器 (cos_lr)。启用余弦退火动态平滑衰减学习率")
        grid_opt.addWidget(w_sw_cos, 1, 2)

        w, self.input_warmup_epochs = self._create_input_field("预热轮数 (warmup_epochs)", DEFAULT_TRAIN_SETTINGS["warmup_epochs"], "学习率预热轮数 (warmup_epochs)。训练初期平滑预热学习率轮数，默认 3.0")
        grid_opt.addWidget(w, 2, 0)

        w, self.input_warmup_momentum = self._create_input_field("预热初始动量 (warmup_momentum)", DEFAULT_TRAIN_SETTINGS["warmup_momentum"], "预热阶段初始动量 (warmup_momentum)，默认 0.8")
        grid_opt.addWidget(w, 2, 1)

        w, self.input_warmup_bias_lr = self._create_input_field("偏置学习率 (warmup_bias_lr)", DEFAULT_TRAIN_SETTINGS["warmup_bias_lr"], "预热阶段偏置项学习率 (warmup_bias_lr)，默认 0.1")
        grid_opt.addWidget(w, 2, 2)

        l_adv.addLayout(grid_opt)

        # ── Section 2: 训练策略与控制 ──
        l_adv.addWidget(self._create_section_divider("二、 训练策略与加速控制 (Training Strategy)"))
        grid_strat = QGridLayout()
        grid_strat.setHorizontalSpacing(14)
        grid_strat.setVerticalSpacing(8)

        w, self.input_patience = self._create_input_field("早停等待轮数 (patience)", DEFAULT_TRAIN_SETTINGS["patience"], "早停机制等待轮数 (patience)。连续 N 轮验证集指标未提升则提前终止训练，默认 100")
        grid_strat.addWidget(w, 0, 0)

        w, self.input_close_mosaic = self._create_input_field("关闭 Mosaic 轮数 (close_mosaic)", DEFAULT_TRAIN_SETTINGS["close_mosaic"], "最后 N 轮禁用 Mosaic 数据增强 (close_mosaic)。稳定收敛提升精度，0 为全程启用，默认 10")
        grid_strat.addWidget(w, 0, 1)

        w, self.input_save_period = self._create_input_field("模型保存间隔 (save_period)", DEFAULT_TRAIN_SETTINGS["save_period"], "权重检查点保存间隔 (save_period)。每隔 N 轮保存一次，-1 表示仅保存 best.pt 和 last.pt")
        grid_strat.addWidget(w, 0, 2)

        w, self.input_seed = self._create_input_field("随机数种子 (seed)", DEFAULT_TRAIN_SETTINGS["seed"], "随机数种子 (seed)。固定数值 (如 0) 保证训练实验完全可复现")
        grid_strat.addWidget(w, 1, 0)

        w, self.input_fraction = self._create_input_field("数据集采样比例 (fraction)", DEFAULT_TRAIN_SETTINGS["fraction"], "参与训练的数据集比例 (fraction)。1.0 为全部数据，小数 (如 0.1) 用于极速验证")
        grid_strat.addWidget(w, 1, 1)

        w, self.input_nbs = self._create_input_field("标称批次大小 (nbs)", DEFAULT_TRAIN_SETTINGS["nbs"], "标称批次大小 (nbs)。用于损失归一化，默认 64")
        grid_strat.addWidget(w, 1, 2)

        w, self.input_freeze = self._create_input_field("冻结骨干层数 (freeze)", "", "冻结模型骨干层数 (freeze)。冻结 Backbone 前 N 层微调，留空为不冻结")
        grid_strat.addWidget(w, 2, 0)

        w, self.input_multi_scale = self._create_input_field("多尺度扰动幅度 (multi_scale)", DEFAULT_TRAIN_SETTINGS["multi_scale"], "多尺度训练变化范围 (multi_scale)。按 imgsz 的比例动态缩放训练尺寸，0.0 为禁用")
        grid_strat.addWidget(w, 2, 1)

        l_adv.addLayout(grid_strat)

        # Switches in 3 columns
        grid_sw = QGridLayout()
        grid_sw.setHorizontalSpacing(14)
        grid_sw.setVerticalSpacing(6)

        w, self.sw_amp = self._create_switch_field("自动混合精度 (amp)", DEFAULT_TRAIN_SETTINGS["amp"], "自动混合精度训练 (amp)。利用 FP16 显著加速训练并节省显存")
        grid_sw.addWidget(w, 0, 0)

        w, self.sw_deterministic = self._create_switch_field("确定性算法 (deterministic)", DEFAULT_TRAIN_SETTINGS["deterministic"], "启用确定性算法 (deterministic)。保证可复现性，但可能轻微降低运行速度")
        grid_sw.addWidget(w, 0, 1)

        w, self.sw_pretrained = self._create_switch_field("加载预训练权重 (pretrained)", DEFAULT_TRAIN_SETTINGS["pretrained"], "加载预训练模型权重 (pretrained)。基于官方或指定权重开始微调训练")
        grid_sw.addWidget(w, 0, 2)

        w, self.sw_cls_remap = self._create_switch_field("类别自动重映射 (cls_remap)", DEFAULT_TRAIN_SETTINGS["cls_remap"], "自动类别映射 (cls_remap)。自动根据数据集标签对模型输出类别进行智能对齐映射")
        grid_sw.addWidget(w, 1, 0)

        w, self.sw_rect = self._create_switch_field("矩形训练批次 (rect)", DEFAULT_TRAIN_SETTINGS["rect"], "矩形批次训练 (rect)。按图像长宽比打包批次，减少无效 padding 黑边提速")
        grid_sw.addWidget(w, 1, 1)

        w, self.sw_single_cls = self._create_switch_field("强制单类别模式 (single_cls)", DEFAULT_TRAIN_SETTINGS["single_cls"], "强制单类别模式 (single_cls)。将数据集中所有类别统一视为单一目标类别")
        grid_sw.addWidget(w, 1, 2)

        w, self.sw_compile = self._create_switch_field("模型图编译 (compile)", DEFAULT_TRAIN_SETTINGS["compile"], "启用 PyTorch 2.x torch.compile() 图编译加速训练")
        grid_sw.addWidget(w, 2, 0)

        w, self.sw_resume = self._create_switch_field("断点续训恢复 (resume)", DEFAULT_TRAIN_SETTINGS["resume"], "断点续训 (resume)。自动从上次训练中断的检查点 last.pt 恢复训练")
        grid_sw.addWidget(w, 2, 1)

        w, self.sw_profile = self._create_switch_field("耗时性能剖析 (profile)", DEFAULT_TRAIN_SETTINGS["profile"], "耗时性能剖析 (profile)。在训练中对 ONNX/TensorRT 速度与瓶颈进行分析")
        grid_sw.addWidget(w, 2, 2)

        l_adv.addLayout(grid_sw)

        # ── Section 3: 损失权重与检测 ──
        l_adv.addWidget(self._create_section_divider("三、 损失权重与检测头参数 (Loss & Detection)"))
        grid_loss = QGridLayout()
        grid_loss.setHorizontalSpacing(14)
        grid_loss.setVerticalSpacing(8)

        w, self.input_box = self._create_input_field("边界框回归损失权重 (box)", DEFAULT_TRAIN_SETTINGS["box"], "边界框定位损失增益权重 (box)。控制定位误差惩罚比重，默认 7.5")
        grid_loss.addWidget(w, 0, 0)

        w, self.input_cls = self._create_input_field("类别分类损失权重 (cls)", DEFAULT_TRAIN_SETTINGS["cls"], "分类损失增益权重 (cls)。控制类别分类误差比重，默认 0.5")
        grid_loss.addWidget(w, 0, 1)

        w, self.input_dfl = self._create_input_field("分布焦点细化损失 (dfl)", DEFAULT_TRAIN_SETTINGS["dfl"], "分布焦点损失权重 (dfl)。微调预测框边界分布精细度，默认 1.5")
        grid_loss.addWidget(w, 0, 2)

        w, self.input_cls_pw = self._create_input_field("正样本分类权重 (cls_pw)", DEFAULT_TRAIN_SETTINGS["cls_pw"], "类别不平衡正样本权重幂 (cls_pw)。0.0 表示禁用，1.0 表示完全反频率加权")
        grid_loss.addWidget(w, 1, 0)

        w, self.input_max_det = self._create_input_field("单图最大检出数 (max_det)", DEFAULT_TRAIN_SETTINGS["max_det"], "单张图片允许的最大检测目标数量 (max_det)。默认 300")
        grid_loss.addWidget(w, 1, 1)

        l_adv.addLayout(grid_loss)

        # ── Section 4: 图像数据增强超参数 ──
        l_adv.addWidget(self._create_section_divider("四、 图像数据增强超参数 (Image Augmentation)"))
        grid_aug = QGridLayout()
        grid_aug.setHorizontalSpacing(14)
        grid_aug.setVerticalSpacing(8)

        w, self.input_mosaic = self._create_input_field("Mosaic 拼接概率 (mosaic)", DEFAULT_TRAIN_SETTINGS["mosaic"], "Mosaic 4图拼接数据增强概率 (mosaic)。默认 1.0 (100% 启用)")
        grid_aug.addWidget(w, 0, 0)

        w, self.input_mixup = self._create_input_field("MixUp 融合概率 (mixup)", DEFAULT_TRAIN_SETTINGS["mixup"], "MixUp 图像透明混合增强概率 (mixup)。默认 0.0")
        grid_aug.addWidget(w, 0, 1)

        w, self.input_cutmix = self._create_input_field("CutMix 剪贴概率 (cutmix)", DEFAULT_TRAIN_SETTINGS["cutmix"], "CutMix 图像局部剪贴替换增强概率 (cutmix)。默认 0.0")
        grid_aug.addWidget(w, 0, 2)

        w, self.input_hsv_h = self._create_input_field("HSV 色相扰动 (hsv_h)", DEFAULT_TRAIN_SETTINGS["hsv_h"], "HSV 颜色空间色相变化幅度 (hsv_h)。默认 0.015")
        grid_aug.addWidget(w, 1, 0)

        w, self.input_hsv_s = self._create_input_field("HSV 饱和度扰动 (hsv_s)", DEFAULT_TRAIN_SETTINGS["hsv_s"], "HSV 颜色空间饱和度变化幅度 (hsv_s)。默认 0.7")
        grid_aug.addWidget(w, 1, 1)

        w, self.input_hsv_v = self._create_input_field("HSV 明度扰动 (hsv_v)", DEFAULT_TRAIN_SETTINGS["hsv_v"], "HSV 颜色空间明度/亮度变化幅度 (hsv_v)。默认 0.4")
        grid_aug.addWidget(w, 1, 2)

        w, self.input_degrees = self._create_input_field("随机旋转角度 (degrees)", DEFAULT_TRAIN_SETTINGS["degrees"], "随机旋转角度范围 (degrees)。以度为单位 (+/- deg)，默认 0.0")
        grid_aug.addWidget(w, 2, 0)

        w, self.input_translate = self._create_input_field("随机平移变换 (translate)", DEFAULT_TRAIN_SETTINGS["translate"], "水平与垂直平移比例范围 (translate)。(+/- fraction)，默认 0.1")
        grid_aug.addWidget(w, 2, 1)

        w, self.input_scale = self._create_input_field("尺度增益缩放 (scale)", DEFAULT_TRAIN_SETTINGS["scale"], "图像尺度缩放增益 (scale)。(+/- gain)，默认 0.5")
        grid_aug.addWidget(w, 2, 2)

        w, self.input_shear = self._create_input_field("随机剪切角度 (shear)", DEFAULT_TRAIN_SETTINGS["shear"], "错切剪切仿射变换角度 (shear)。(+/- deg)，默认 0.0")
        grid_aug.addWidget(w, 3, 0)

        w, self.input_perspective = self._create_input_field("透视畸变系数 (perspective)", DEFAULT_TRAIN_SETTINGS["perspective"], "透视变换扭曲系数 (perspective)。通常取值 0.0 ~ 0.001，默认 0.0")
        grid_aug.addWidget(w, 3, 1)

        w, self.input_bgr = self._create_input_field("BGR 通道交换 (bgr)", DEFAULT_TRAIN_SETTINGS["bgr"], "RGB 与 BGR 通道随机交换概率 (bgr)。默认 0.0")
        grid_aug.addWidget(w, 3, 2)

        w, self.input_fliplr = self._create_input_field("水平镜像翻转 (fliplr)", DEFAULT_TRAIN_SETTINGS["fliplr"], "左右水平翻转增强概率 (fliplr)。默认 0.5 (50% 概率翻转)")
        grid_aug.addWidget(w, 4, 0)

        w, self.input_flipud = self._create_input_field("垂直镜像翻转 (flipud)", DEFAULT_TRAIN_SETTINGS["flipud"], "上下垂直翻转增强概率 (flipud)。默认 0.0")
        grid_aug.addWidget(w, 4, 1)

        l_adv.addLayout(grid_aug)

        for g in (grid_opt, grid_strat, grid_sw, grid_loss, grid_aug):
            g.setColumnStretch(0, 1)
            g.setColumnStretch(1, 1)
            g.setColumnStretch(2, 1)

        # Style newly created elements according to current theme
        self.apply_theme()

        # Connect CLI signals for newly created fields
        adv_inputs = [
            getattr(self, "input_lr0", None), getattr(self, "input_lrf", None),
            getattr(self, "input_momentum", None), getattr(self, "input_weight_decay", None),
            getattr(self, "input_warmup_epochs", None), getattr(self, "input_warmup_momentum", None),
            getattr(self, "input_warmup_bias_lr", None), getattr(self, "input_patience", None),
            getattr(self, "input_close_mosaic", None), getattr(self, "input_save_period", None),
            getattr(self, "input_seed", None), getattr(self, "input_fraction", None),
            getattr(self, "input_nbs", None), getattr(self, "input_freeze", None),
            getattr(self, "input_multi_scale", None), getattr(self, "input_box", None),
            getattr(self, "input_cls", None), getattr(self, "input_dfl", None),
            getattr(self, "input_cls_pw", None), getattr(self, "input_max_det", None),
            getattr(self, "input_mosaic", None), getattr(self, "input_mixup", None),
            getattr(self, "input_cutmix", None), getattr(self, "input_bgr", None),
            getattr(self, "input_hsv_h", None), getattr(self, "input_hsv_s", None),
            getattr(self, "input_hsv_v", None), getattr(self, "input_degrees", None),
            getattr(self, "input_translate", None), getattr(self, "input_scale", None),
            getattr(self, "input_shear", None), getattr(self, "input_perspective", None),
            getattr(self, "input_fliplr", None), getattr(self, "input_flipud", None),
        ]
        for inp in adv_inputs:
            if inp is not None:
                inp.textChanged.connect(self._update_cli_command)

        if hasattr(self, "combo_optimizer"):
            self.combo_optimizer.currentIndexChanged.connect(self._update_cli_command)

        adv_switches = [
            getattr(self, "sw_cos_lr", None), getattr(self, "sw_amp", None),
            getattr(self, "sw_deterministic", None), getattr(self, "sw_pretrained", None),
            getattr(self, "sw_cls_remap", None), getattr(self, "sw_rect", None),
            getattr(self, "sw_single_cls", None), getattr(self, "sw_compile", None),
            getattr(self, "sw_resume", None), getattr(self, "sw_profile", None)
        ]
        for sw in adv_switches:
            if sw is not None:
                if hasattr(sw, "stateChanged"):
                    sw.stateChanged.connect(self._update_cli_command)
                elif hasattr(sw, "toggled"):
                    sw.toggled.connect(self._update_cli_command)

    def _on_env_changed(self, env_info: dict):
        """Called when an environment probe finishes in EnvironmentSelectorWidget."""
        if env_info.get("cuda_available"):
            dev_names = env_info.get("device_names", [])
            gpu_name = dev_names[0] if dev_names else "GPU 0"
            gpu_model = format_gpu_model_name(gpu_name)
            gpu_label = f"0 ({gpu_model})"
            self.combo_device.blockSignals(True)
            self.combo_device.clear()
            self.combo_device.addItems([gpu_label, "CPU"])
            self.combo_device.setCurrentIndex(0)
            self.combo_device.blockSignals(False)
        elif env_info.get("torch_installed"):
            self.combo_device.blockSignals(True)
            self.combo_device.clear()
            self.combo_device.addItems(["CPU"])
            self.combo_device.setCurrentIndex(0)
            self.combo_device.blockSignals(False)
        self._update_cli_command()

    def reset_to_defaults(self):
        """Reset all training settings & augmentations back to exact default specifications"""
        self._ensure_adv_built()
        # Core
        if hasattr(self, "combo_model") and hasattr(self.combo_model, "reset_to_default"):
            self.combo_model.reset_to_default()
        self.input_epochs.setText(str(DEFAULT_TRAIN_SETTINGS["epochs"]))
        self.input_batch.setText("auto")
        self.combo_imgsz.setCurrentText(str(DEFAULT_TRAIN_SETTINGS["imgsz"]))
        self.input_task_name.setText("")
        # Optimizer & LR
        self.combo_optimizer.setCurrentText(str(DEFAULT_TRAIN_SETTINGS["optimizer"]))
        self.input_lr0.setText(str(DEFAULT_TRAIN_SETTINGS["lr0"]))
        self.input_lrf.setText(str(DEFAULT_TRAIN_SETTINGS["lrf"]))
        self.input_momentum.setText(str(DEFAULT_TRAIN_SETTINGS["momentum"]))
        self.input_weight_decay.setText(str(DEFAULT_TRAIN_SETTINGS["weight_decay"]))
        self.input_warmup_epochs.setText(str(DEFAULT_TRAIN_SETTINGS["warmup_epochs"]))
        self.input_warmup_momentum.setText(str(DEFAULT_TRAIN_SETTINGS["warmup_momentum"]))
        self.input_warmup_bias_lr.setText(str(DEFAULT_TRAIN_SETTINGS["warmup_bias_lr"]))
        self.sw_cos_lr.setChecked(DEFAULT_TRAIN_SETTINGS["cos_lr"])
        # Strategy
        self.input_patience.setText(str(DEFAULT_TRAIN_SETTINGS["patience"]))
        self.input_close_mosaic.setText(str(DEFAULT_TRAIN_SETTINGS["close_mosaic"]))
        self.input_save_period.setText(str(DEFAULT_TRAIN_SETTINGS["save_period"]))
        self.input_seed.setText(str(DEFAULT_TRAIN_SETTINGS["seed"]))
        self.input_fraction.setText(str(DEFAULT_TRAIN_SETTINGS["fraction"]))
        self.input_nbs.setText(str(DEFAULT_TRAIN_SETTINGS["nbs"]))
        self.input_freeze.setText("")
        self.input_multi_scale.setText(str(DEFAULT_TRAIN_SETTINGS["multi_scale"]))
        self.sw_amp.setChecked(DEFAULT_TRAIN_SETTINGS["amp"])
        self.sw_deterministic.setChecked(DEFAULT_TRAIN_SETTINGS["deterministic"])
        self.sw_pretrained.setChecked(DEFAULT_TRAIN_SETTINGS["pretrained"])
        self.sw_cls_remap.setChecked(DEFAULT_TRAIN_SETTINGS["cls_remap"])
        self.sw_rect.setChecked(DEFAULT_TRAIN_SETTINGS["rect"])
        self.sw_single_cls.setChecked(DEFAULT_TRAIN_SETTINGS["single_cls"])
        self.sw_compile.setChecked(DEFAULT_TRAIN_SETTINGS["compile"])
        self.sw_resume.setChecked(DEFAULT_TRAIN_SETTINGS["resume"])
        self.sw_profile.setChecked(DEFAULT_TRAIN_SETTINGS["profile"])
        # Loss & Detection
        self.input_box.setText(str(DEFAULT_TRAIN_SETTINGS["box"]))
        self.input_cls.setText(str(DEFAULT_TRAIN_SETTINGS["cls"]))
        self.input_dfl.setText(str(DEFAULT_TRAIN_SETTINGS["dfl"]))
        self.input_cls_pw.setText(str(DEFAULT_TRAIN_SETTINGS["cls_pw"]))
        self.input_max_det.setText(str(DEFAULT_TRAIN_SETTINGS["max_det"]))
        # Augmentation
        self.input_mosaic.setText(str(DEFAULT_TRAIN_SETTINGS["mosaic"]))
        self.input_mixup.setText(str(DEFAULT_TRAIN_SETTINGS["mixup"]))
        self.input_cutmix.setText(str(DEFAULT_TRAIN_SETTINGS["cutmix"]))
        self.input_bgr.setText(str(DEFAULT_TRAIN_SETTINGS["bgr"]))
        self.input_hsv_h.setText(str(DEFAULT_TRAIN_SETTINGS["hsv_h"]))
        self.input_hsv_s.setText(str(DEFAULT_TRAIN_SETTINGS["hsv_s"]))
        self.input_hsv_v.setText(str(DEFAULT_TRAIN_SETTINGS["hsv_v"]))
        self.input_degrees.setText(str(DEFAULT_TRAIN_SETTINGS["degrees"]))
        self.input_translate.setText(str(DEFAULT_TRAIN_SETTINGS["translate"]))
        self.input_scale.setText(str(DEFAULT_TRAIN_SETTINGS["scale"]))
        self.input_shear.setText(str(DEFAULT_TRAIN_SETTINGS["shear"]))
        self.input_perspective.setText(str(DEFAULT_TRAIN_SETTINGS["perspective"]))
        self.input_fliplr.setText(str(DEFAULT_TRAIN_SETTINGS["fliplr"]))
        self.input_flipud.setText(str(DEFAULT_TRAIN_SETTINGS["flipud"]))
        self._update_cli_command()
        MkMessage.info(self, "已重置所有参数为默认配置！")

    def _get_modified_hyperparameters(self) -> dict:
        """Inspect all advanced settings and return dict of parameters that differ from DEFAULT_TRAIN_SETTINGS."""
        if not getattr(self, "_adv_built", False):
            return {}
        modified = {}

        # 1. Optimizer & LR
        try:
            opt = self.combo_optimizer.currentText()
            if opt != DEFAULT_TRAIN_SETTINGS["optimizer"]:
                modified["optimizer"] = opt

            lr0_val = float(self.input_lr0.text().strip())
            if abs(lr0_val - DEFAULT_TRAIN_SETTINGS["lr0"]) > 1e-6:
                modified["lr0"] = lr0_val

            lrf_val = float(self.input_lrf.text().strip())
            if abs(lrf_val - DEFAULT_TRAIN_SETTINGS["lrf"]) > 1e-6:
                modified["lrf"] = lrf_val

            mom_val = float(self.input_momentum.text().strip())
            if abs(mom_val - DEFAULT_TRAIN_SETTINGS["momentum"]) > 1e-6:
                modified["momentum"] = mom_val

            wd_val = float(self.input_weight_decay.text().strip())
            if abs(wd_val - DEFAULT_TRAIN_SETTINGS["weight_decay"]) > 1e-6:
                modified["weight_decay"] = wd_val

            we_val = float(self.input_warmup_epochs.text().strip())
            if abs(we_val - DEFAULT_TRAIN_SETTINGS["warmup_epochs"]) > 1e-6:
                modified["warmup_epochs"] = we_val

            wm_val = float(self.input_warmup_momentum.text().strip())
            if abs(wm_val - DEFAULT_TRAIN_SETTINGS["warmup_momentum"]) > 1e-6:
                modified["warmup_momentum"] = wm_val

            wb_val = float(self.input_warmup_bias_lr.text().strip())
            if abs(wb_val - DEFAULT_TRAIN_SETTINGS["warmup_bias_lr"]) > 1e-6:
                modified["warmup_bias_lr"] = wb_val

            cos_lr_checked = getattr(self.sw_cos_lr, "checked", False)
            if bool(cos_lr_checked) != bool(DEFAULT_TRAIN_SETTINGS["cos_lr"]):
                modified["cos_lr"] = bool(cos_lr_checked)
        except (ValueError, AttributeError):
            pass

        # 2. Strategy & Regularization
        try:
            pat_val = int(self.input_patience.text().strip())
            if pat_val != DEFAULT_TRAIN_SETTINGS["patience"]:
                modified["patience"] = pat_val

            cm_val = int(self.input_close_mosaic.text().strip())
            if cm_val != DEFAULT_TRAIN_SETTINGS["close_mosaic"]:
                modified["close_mosaic"] = cm_val

            sp_val = int(self.input_save_period.text().strip())
            if sp_val != DEFAULT_TRAIN_SETTINGS["save_period"]:
                modified["save_period"] = sp_val

            seed_val = int(self.input_seed.text().strip())
            if seed_val != DEFAULT_TRAIN_SETTINGS["seed"]:
                modified["seed"] = seed_val

            frac_val = float(self.input_fraction.text().strip())
            if abs(frac_val - DEFAULT_TRAIN_SETTINGS["fraction"]) > 1e-6:
                modified["fraction"] = frac_val

            nbs_val = int(self.input_nbs.text().strip())
            if nbs_val != DEFAULT_TRAIN_SETTINGS["nbs"]:
                modified["nbs"] = nbs_val

            frz_txt = self.input_freeze.text().strip()
            if frz_txt:
                try:
                    modified["freeze"] = int(frz_txt)
                except ValueError:
                    modified["freeze"] = frz_txt

            ms_val = float(self.input_multi_scale.text().strip())
            if abs(ms_val - DEFAULT_TRAIN_SETTINGS["multi_scale"]) > 1e-6:
                modified["multi_scale"] = ms_val

            for attr, key in [
                ("sw_amp", "amp"),
                ("sw_deterministic", "deterministic"),
                ("sw_pretrained", "pretrained"),
                ("sw_cls_remap", "cls_remap"),
                ("sw_rect", "rect"),
                ("sw_single_cls", "single_cls"),
                ("sw_compile", "compile"),
                ("sw_resume", "resume"),
                ("sw_profile", "profile"),
            ]:
                sw = getattr(self, attr, None)
                if sw:
                    chk = getattr(sw, "checked", False)
                    if bool(chk) != bool(DEFAULT_TRAIN_SETTINGS[key]):
                        modified[key] = bool(chk)
        except (ValueError, AttributeError):
            pass

        # 3. Loss & Detection
        try:
            box_val = float(self.input_box.text().strip())
            if abs(box_val - DEFAULT_TRAIN_SETTINGS["box"]) > 1e-6:
                modified["box"] = box_val

            cls_val = float(self.input_cls.text().strip())
            if abs(cls_val - DEFAULT_TRAIN_SETTINGS["cls"]) > 1e-6:
                modified["cls"] = cls_val

            dfl_val = float(self.input_dfl.text().strip())
            if abs(dfl_val - DEFAULT_TRAIN_SETTINGS["dfl"]) > 1e-6:
                modified["dfl"] = dfl_val

            cls_pw_val = float(self.input_cls_pw.text().strip())
            if abs(cls_pw_val - DEFAULT_TRAIN_SETTINGS["cls_pw"]) > 1e-6:
                modified["cls_pw"] = cls_pw_val

            max_det_val = int(self.input_max_det.text().strip())
            if max_det_val != DEFAULT_TRAIN_SETTINGS["max_det"]:
                modified["max_det"] = max_det_val
        except (ValueError, AttributeError):
            pass

        # 4. Image Augmentations
        try:
            mosaic_val = float(self.input_mosaic.text().strip())
            if abs(mosaic_val - DEFAULT_TRAIN_SETTINGS["mosaic"]) > 1e-6:
                modified["mosaic"] = mosaic_val

            mixup_val = float(self.input_mixup.text().strip())
            if abs(mixup_val - DEFAULT_TRAIN_SETTINGS["mixup"]) > 1e-6:
                modified["mixup"] = mixup_val

            cutmix_val = float(self.input_cutmix.text().strip())
            if abs(cutmix_val - DEFAULT_TRAIN_SETTINGS["cutmix"]) > 1e-6:
                modified["cutmix"] = cutmix_val

            bgr_val = float(self.input_bgr.text().strip())
            if abs(bgr_val - DEFAULT_TRAIN_SETTINGS["bgr"]) > 1e-6:
                modified["bgr"] = bgr_val

            hsv_h_val = float(self.input_hsv_h.text().strip())
            if abs(hsv_h_val - DEFAULT_TRAIN_SETTINGS["hsv_h"]) > 1e-6:
                modified["hsv_h"] = hsv_h_val

            hsv_s_val = float(self.input_hsv_s.text().strip())
            if abs(hsv_s_val - DEFAULT_TRAIN_SETTINGS["hsv_s"]) > 1e-6:
                modified["hsv_s"] = hsv_s_val

            hsv_v_val = float(self.input_hsv_v.text().strip())
            if abs(hsv_v_val - DEFAULT_TRAIN_SETTINGS["hsv_v"]) > 1e-6:
                modified["hsv_v"] = hsv_v_val

            degrees_val = float(self.input_degrees.text().strip())
            if abs(degrees_val - DEFAULT_TRAIN_SETTINGS["degrees"]) > 1e-6:
                modified["degrees"] = degrees_val

            translate_val = float(self.input_translate.text().strip())
            if abs(translate_val - DEFAULT_TRAIN_SETTINGS["translate"]) > 1e-6:
                modified["translate"] = translate_val

            scale_val = float(self.input_scale.text().strip())
            if abs(scale_val - DEFAULT_TRAIN_SETTINGS["scale"]) > 1e-6:
                modified["scale"] = scale_val

            shear_val = float(self.input_shear.text().strip())
            if abs(shear_val - DEFAULT_TRAIN_SETTINGS["shear"]) > 1e-6:
                modified["shear"] = shear_val

            perspective_val = float(self.input_perspective.text().strip())
            if abs(perspective_val - DEFAULT_TRAIN_SETTINGS["perspective"]) > 1e-6:
                modified["perspective"] = perspective_val

            fliplr_val = float(self.input_fliplr.text().strip())
            if abs(fliplr_val - DEFAULT_TRAIN_SETTINGS["fliplr"]) > 1e-6:
                modified["fliplr"] = fliplr_val

            flipud_val = float(self.input_flipud.text().strip())
            if abs(flipud_val - DEFAULT_TRAIN_SETTINGS["flipud"]) > 1e-6:
                modified["flipud"] = flipud_val
        except (ValueError, AttributeError):
            pass

        # 5. Device (if user chooses CPU)
        if hasattr(self, "combo_device"):
            dev_text = self.combo_device.currentText()
            if "CPU" in dev_text:
                modified["device"] = "cpu"

        return modified

    def _update_cli_command(self):
        if not hasattr(self, "cli_viewer") or not self.cli_viewer:
            return

        # 1. model
        custom_model = self.input_custom_model.text().strip() if hasattr(self, "input_custom_model") else ""
        if custom_model:
            model_str = custom_model
        else:
            model_str = self.combo_model.currentData() if hasattr(self, "combo_model") else ""
            if not model_str and hasattr(self, "combo_model"):
                model_str = self.combo_model.currentText().strip()
            if not model_str:
                model_str = "yolo26n.pt"

        # 3. task
        task_str = self.task_type

        # 2. data (absolute path to the actual YAML file)
        is_pose = (task_str or "detect").lower() in ("pose", "keypoint", "kpt", "keypoints")
        cfg_filename = "data-pose.yaml" if is_pose else "data.yaml"
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        tp_dir = os.path.normpath(os.path.join(cur_dir, "..", "..", ".."))
        default_yaml_abs = os.path.normpath(os.path.join(tp_dir, "cfg", "datasets", cfg_filename)).replace('\\', '/')
        data_str = getattr(self, "_uploaded_yaml_path", None) or default_yaml_abs

        # 4. epochs
        try:
            epochs_val = int(self.input_epochs.text().strip())
        except (ValueError, AttributeError):
            epochs_val = 100

        # 5. batch
        batch_raw = self.input_batch.text().strip().lower() if hasattr(self, "input_batch") else "auto"
        if batch_raw in ("auto", "-1"):
            batch_val = -1
        else:
            try:
                batch_val = int(batch_raw)
            except ValueError:
                batch_val = -1

        # 6. imgsz
        try:
            imgsz_val = int(self.combo_imgsz.currentText().strip())
        except (ValueError, AttributeError):
            imgsz_val = 640

        task_name = self.input_task_name.text().strip() if hasattr(self, "input_task_name") else ""
        if not task_name:
            task_name = self._get_next_available_task_name()

        shell_fmt = self.combo_cli_shell.currentText() if hasattr(self, "combo_cli_shell") else "单行"
        is_single_line = "单行" in shell_fmt
        if "^" in shell_fmt:
            continuation = " ^"
        elif "`" in shell_fmt:
            continuation = " `"
        elif "\\" in shell_fmt:
            continuation = " \\"
        else:
            continuation = ""

        extra_params = self._get_modified_hyperparameters()

        if is_single_line:
            params = [
                "yolo train",
                f'model="{model_str.strip()}"',
                f'data="{data_str.strip()}"',
                f'task="{task_str.strip()}"',
                f'name="{task_name.strip()}"',
                f'epochs={epochs_val}',
                f'batch={batch_val}',
                f'imgsz={imgsz_val}'
            ]
            if extra_params:
                for k, v in extra_params.items():
                    if isinstance(v, str):
                        params.append(f'{k}="{v.strip()}"')
                    else:
                        params.append(f'{k}={v}')
            full_cmd = " ".join([p.strip() for p in params if p and p.strip()]).strip()
        else:
            lines = [
                f"yolo train{continuation}",
                f'  model="{model_str.strip()}"{continuation}',
                f'  data="{data_str.strip()}"{continuation}',
                f'  task="{task_str.strip()}"{continuation}',
                f'  name="{task_name.strip()}"{continuation}',
                f'  epochs={epochs_val}{continuation}',
                f'  batch={batch_val}{continuation}',
                f'  imgsz={imgsz_val}'
            ]
            if extra_params:
                lines[-1] += continuation
                items = list(extra_params.items())
                for i, (k, v) in enumerate(items):
                    is_last = (i == len(items) - 1)
                    suffix = "" if is_last else continuation
                    if isinstance(v, str):
                        lines.append(f'  {k}="{v.strip()}"{suffix}')
                    else:
                        lines.append(f'  {k}={v}{suffix}')
            clean_lines = [l.rstrip() for l in lines]
            full_cmd = "\n".join(clean_lines).strip()

        while full_cmd.endswith(" ") or full_cmd.endswith("\t"):
            full_cmd = full_cmd[:-1]

        self.cli_viewer.set_command_text(full_cmd)

    def _connect_cli_command_signals(self):
        """Connect all interactive controls to _update_cli_command."""
        # Core
        if hasattr(self, "combo_model"):
            self.combo_model.currentIndexChanged.connect(self._update_cli_command)
        if hasattr(self, "input_custom_model"):
            self.input_custom_model.textChanged.connect(self._update_cli_command)
        if hasattr(self, "input_epochs"):
            self.input_epochs.textChanged.connect(self._update_cli_command)
        if hasattr(self, "input_batch"):
            self.input_batch.textChanged.connect(self._update_cli_command)
        if hasattr(self, "combo_imgsz"):
            self.combo_imgsz.currentTextChanged.connect(self._update_cli_command)
        if hasattr(self, "combo_device"):
            self.combo_device.currentIndexChanged.connect(self._update_cli_command)
        if hasattr(self, "yaml_editor") and hasattr(self.yaml_editor, "editor"):
            self.yaml_editor.editor.textChanged.connect(self._update_cli_command)

        # Advanced inputs
        adv_inputs = [
            getattr(self, "input_lr0", None), getattr(self, "input_lrf", None),
            getattr(self, "input_momentum", None), getattr(self, "input_weight_decay", None),
            getattr(self, "input_warmup_epochs", None), getattr(self, "input_warmup_momentum", None),
            getattr(self, "input_warmup_bias_lr", None), getattr(self, "input_patience", None),
            getattr(self, "input_close_mosaic", None), getattr(self, "input_save_period", None),
            getattr(self, "input_seed", None), getattr(self, "input_fraction", None),
            getattr(self, "input_nbs", None), getattr(self, "input_freeze", None),
            getattr(self, "input_multi_scale", None), getattr(self, "input_box", None),
            getattr(self, "input_cls", None), getattr(self, "input_dfl", None),
            getattr(self, "input_cls_pw", None), getattr(self, "input_max_det", None),
            getattr(self, "input_mosaic", None), getattr(self, "input_mixup", None),
            getattr(self, "input_cutmix", None), getattr(self, "input_bgr", None),
            getattr(self, "input_hsv_h", None), getattr(self, "input_hsv_s", None),
            getattr(self, "input_hsv_v", None), getattr(self, "input_degrees", None),
            getattr(self, "input_translate", None), getattr(self, "input_scale", None),
            getattr(self, "input_shear", None), getattr(self, "input_perspective", None),
            getattr(self, "input_fliplr", None), getattr(self, "input_flipud", None),
        ]
        for inp in adv_inputs:
            if inp is not None:
                inp.textChanged.connect(self._update_cli_command)

        if hasattr(self, "combo_optimizer"):
            self.combo_optimizer.currentIndexChanged.connect(self._update_cli_command)

        adv_switches = [
            getattr(self, "sw_cos_lr", None), getattr(self, "sw_amp", None),
            getattr(self, "sw_deterministic", None), getattr(self, "sw_pretrained", None),
            getattr(self, "sw_cls_remap", None), getattr(self, "sw_rect", None),
            getattr(self, "sw_single_cls", None), getattr(self, "sw_compile", None),
            getattr(self, "sw_resume", None), getattr(self, "sw_profile", None)
        ]
        for sw in adv_switches:
            if sw is not None:
                if hasattr(sw, "stateChanged"):
                    sw.stateChanged.connect(self._update_cli_command)
                elif hasattr(sw, "toggled"):
                    sw.toggled.connect(self._update_cli_command)

    def _on_task_type_changed(self, index):
        self.task_type = self.combo_task_type.currentData()
        self.tag_lbl.setText(self.combo_task_type.currentText())
        self.combo_model.task_type = self.task_type
        if self.combo_model._popup is not None:
            self.combo_model._popup.deleteLater()
            self.combo_model._popup = None
        self.combo_model.reset_to_default()
        if hasattr(self, "yaml_editor"):
            self.yaml_editor.task_type = self.task_type
            self.yaml_editor.set_yaml_text(format_filtered_yaml(self.yaml_editor.get_yaml_text(), self.task_type))
        self._update_cli_command()

    def _inspect_custom_model(self, path, start_after=False):
        import threading
        from TrainingPlatform.core.checkpoint_metadata import load_checkpoint_metadata
        python_exe = self.env_selector.get_selected_python()
        self.btn_start.setEnabled(False)

        def inspect_model():
            try:
                result = load_checkpoint_metadata(path, python_exe)
            except Exception as exc:
                result = {"error": str(exc)}
            try:
                self.model_inspected.emit(path, result, start_after)
            except RuntimeError:
                pass  # The dialog was closed while its worker was reading.

        threading.Thread(target=inspect_model, daemon=True).start()

    def _on_custom_model_inspected(self, path, result, start_after):
        self.btn_start.setEnabled(True)
        if self.input_custom_model.text().strip() != path:
            return
        if result.get("error"):
            MkMessage.error(self, result["error"])
            return
        index = self.combo_task_type.findData(result.get("task"))
        if index < 0:
            MkMessage.error(self, "当前训练对话框尚不支持该模型的任务类型")
            return
        self.combo_task_type.setCurrentIndex(index)
        self._custom_model_signature = (path, os.stat(path).st_mtime_ns, os.path.getsize(path))
        self._custom_model_task = result["task"]
        if start_after:
            self.handle_start_training()

    def _populate_model_options(self):
        """Populate models catalog (handled dynamically by ModelPickerWidget)."""
        pass

    def _browse_custom_model(self):
        self._is_showing_sub_dialog = True
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择本地 PyTorch 权重文件", "", "PyTorch Models (*.pt);;All Files (*.*)")
            if file_path:
                self.input_custom_model.setText(file_path)
                self._inspect_custom_model(file_path)
                MkMessage.info(self, f"已选择本地权重: {os.path.basename(file_path)}")
        finally:
            self._is_showing_sub_dialog = False

    def _upload_and_parse_yaml(self):
        self._is_showing_sub_dialog = True
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择数据集配置文件 (data.yaml)", "", "YAML 文件 (*.yaml *.yml);;所有文件 (*.*)"
            )
            if file_path:
                self._uploaded_yaml_path = file_path.replace('\\', '/')
                formatted_yaml = format_filtered_yaml(file_path, self.task_type)
                self.yaml_editor.set_yaml_text(formatted_yaml)
                self._update_cli_command()
                MkMessage.success(self, f"已成功解析并提取必要参数: {os.path.basename(file_path)}")
        except Exception as e:
            MkMessage.error(self, f"解析 YAML 文件失败: {str(e)}")
        finally:
            self._is_showing_sub_dialog = False

    def _reset_yaml_template(self):
        self._uploaded_yaml_path = None
        self.yaml_editor.reset_to_default_template()
        self._update_cli_command()
        MkMessage.info(self, "已重置为当前任务类型的默认 YAML 模板")

    def _toggle_advanced_settings(self):
        InfoTooltipPopover.hide_popover()
        is_visible = self.adv_container.isVisible()
        new_state = not is_visible
        if new_state:
            self._ensure_adv_built()
        
        collapsed_h, expanded_h = self._get_target_heights()
        target_h = expanded_h if new_state else collapsed_h
        self.adv_header_bar.set_expanded(new_state)

        if self._parent_widget:
            cur_h = self.container_card.height()
            if new_state:
                self.adv_container.setVisible(True)
                self._animate_card_height(cur_h, target_h)
            else:
                self.scroll.verticalScrollBar().setValue(0)
                def _on_collapse_done():
                    self.adv_container.setVisible(False)
                self._animate_card_height(cur_h, target_h, on_finished=_on_collapse_done)
        else:
            cur_geom = self.geometry()
            diff = target_h - cur_geom.height()
            new_y = cur_geom.y() - diff // 2

            screen = self.screen() or QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                if new_y < avail.top() + 30:
                    new_y = avail.top() + 30
                if new_y + target_h > avail.bottom() - 30:
                    new_y = max(avail.top() + 30, avail.bottom() - 30 - target_h)
            else:
                new_y = max(30, new_y)

            target_geom = QRect(cur_geom.x(), new_y, cur_geom.width(), target_h)

            if new_state:
                self.adv_container.setVisible(True)
                self._animate_geometry(cur_geom, target_geom)
            else:
                self.scroll.verticalScrollBar().setValue(0)
                self.adv_container.setVisible(False)
                self._animate_geometry(cur_geom, target_geom)

    def handle_start_training(self):
        task_name = self.input_task_name.text().strip()
        if not task_name:
            task_name = self._get_next_available_task_name()

        # 1. Resolve Model
        custom_model = self.input_custom_model.text().strip()
        if custom_model:
            if not os.path.exists(custom_model):
                MkMessage.error(self, f"指定的本地权重文件不存在:\n{custom_model}")
                return
            model_name = custom_model
            signature = (custom_model, os.stat(custom_model).st_mtime_ns, os.path.getsize(custom_model))
            if signature != self._custom_model_signature:
                self._inspect_custom_model(custom_model, start_after=True)
                return
            self.combo_task_type.setCurrentIndex(self.combo_task_type.findData(self._custom_model_task))
        else:
            model_name = self.combo_model.currentData()
            if not model_name:
                model_name = self.combo_model.currentText().split()[0]

        # 2. Resolve Dataset from YAML Editor
        yaml_text = self.yaml_editor.get_yaml_text().strip()
        if not yaml_text:
            MkMessage.warning(self, "数据集配置不能为空，请填写或上传数据集 YAML！")
            return

        try:
            ds_data = yaml.safe_load(yaml_text)
            if not isinstance(ds_data, dict):
                raise ValueError("YAML 格式不正确，顶级结构应为键值对字典")
        except Exception as e:
            MkMessage.error(self, f"数据集 YAML 语法格式错误，请检查:\n{str(e)}")
            return

        train_path = ds_data.get('train', '')
        val_path = ds_data.get('val', '')
        if not train_path or not str(train_path).strip():
            MkMessage.warning(self, "数据集配置缺少训练集路径 (train)！")
            return
        if not val_path or not str(val_path).strip():
            MkMessage.warning(self, "数据集配置缺少验证集路径 (val)！")
            return
        if not ds_data.get('names'):
            MkMessage.warning(self, "数据集配置缺少类别名称定义 (names)！")
            return

        # Save the edited YAML to project root datasets directory
        try:
            from TrainingPlatform.core.env_manager import get_workspace_root
            workspace = get_workspace_root()
            datasets_dir = os.path.join(workspace, "datasets", f"proj_{self.project['id']}")
            os.makedirs(datasets_dir, exist_ok=True)
            yaml_filepath = os.path.join(datasets_dir, f"data_{task_name}.yaml").replace('\\', '/')
            with open(yaml_filepath, 'w', encoding='utf-8') as f:
                f.write(yaml_text)

            # Also keep TrainingPlatform/cfg/datasets/data.yaml in sync if default template was edited
            task_str = self.task_type
            is_pose = (task_str or "detect").lower() in ("pose", "keypoint", "kpt", "keypoints")
            cfg_filename = "data-pose.yaml" if is_pose else "data.yaml"
            cfg_yaml_path = os.path.join(workspace, "TrainingPlatform", "cfg", "datasets", cfg_filename)
            try:
                os.makedirs(os.path.dirname(cfg_yaml_path), exist_ok=True)
                with open(cfg_yaml_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_text)
            except Exception:
                pass

            ds_names = ds_data.get('names', {})
            classes_list = []
            if isinstance(ds_names, dict):
                classes_list = [ds_names[k] for k in sorted(ds_names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
            elif isinstance(ds_names, list):
                classes_list = ds_names
            class_count = ds_data.get('nc', len(classes_list))

            dataset_id = self.db.add_dataset(
                project_id=self.project['id'],
                name=f"dataset_{task_name}",
                path=yaml_filepath,
                fmt='yolo',
                class_count=class_count,
                classes=classes_list
            )
        except Exception as e:
            MkMessage.error(self, f"保存数据集 YAML 配置文件失败:\n{str(e)}")
            return

        # 3. Parse Full Hyperparameters & Augmentations
        try:
            # Core
            epochs = int(self.input_epochs.text().strip())
            batch_text = self.input_batch.text().strip().lower()
            batch = -1 if batch_text in ('auto', '-1') else int(batch_text)
            imgsz = int(self.combo_imgsz.currentText())

            if getattr(self, "_adv_built", False):
                # Optimizer & LR
                optimizer = self.combo_optimizer.currentText()
                lr0 = float(self.input_lr0.text().strip())
                lrf = float(self.input_lrf.text().strip())
                momentum = float(self.input_momentum.text().strip())
                weight_decay = float(self.input_weight_decay.text().strip())
                warmup_epochs = float(self.input_warmup_epochs.text().strip())
                warmup_momentum = float(self.input_warmup_momentum.text().strip())
                warmup_bias_lr = float(self.input_warmup_bias_lr.text().strip())
                cos_lr = self.sw_cos_lr.checked

                # Strategy
                patience = int(self.input_patience.text().strip())
                close_mosaic = int(self.input_close_mosaic.text().strip())
                save_period = int(self.input_save_period.text().strip())
                seed = int(self.input_seed.text().strip())
                fraction = float(self.input_fraction.text().strip())
                nbs = int(self.input_nbs.text().strip())
                freeze_txt = self.input_freeze.text().strip()
                freeze = int(freeze_txt) if freeze_txt else None
                multi_scale = float(self.input_multi_scale.text().strip())
                amp = self.sw_amp.checked
                deterministic = self.sw_deterministic.checked
                pretrained = self.sw_pretrained.checked
                cls_remap = self.sw_cls_remap.checked
                rect = self.sw_rect.checked
                single_cls = self.sw_single_cls.checked
                compile_val = self.sw_compile.checked
                resume = self.sw_resume.checked
                profile = self.sw_profile.checked

                # Loss & Detection
                box = float(self.input_box.text().strip())
                cls_val = float(self.input_cls.text().strip())
                dfl = float(self.input_dfl.text().strip())
                cls_pw = float(self.input_cls_pw.text().strip())
                max_det = int(self.input_max_det.text().strip())

                # Image Augmentations
                mosaic = float(self.input_mosaic.text().strip())
                mixup = float(self.input_mixup.text().strip())
                cutmix = float(self.input_cutmix.text().strip())
                bgr = float(self.input_bgr.text().strip())
                hsv_h = float(self.input_hsv_h.text().strip())
                hsv_s = float(self.input_hsv_s.text().strip())
                hsv_v = float(self.input_hsv_v.text().strip())
                degrees = float(self.input_degrees.text().strip())
                translate = float(self.input_translate.text().strip())
                scale = float(self.input_scale.text().strip())
                shear = float(self.input_shear.text().strip())
                perspective = float(self.input_perspective.text().strip())
                fliplr = float(self.input_fliplr.text().strip())
                flipud = float(self.input_flipud.text().strip())
            else:
                # Fast path: defaults preserved when advanced section remains collapsed
                optimizer = DEFAULT_TRAIN_SETTINGS["optimizer"]
                lr0 = float(DEFAULT_TRAIN_SETTINGS["lr0"])
                lrf = float(DEFAULT_TRAIN_SETTINGS["lrf"])
                momentum = float(DEFAULT_TRAIN_SETTINGS["momentum"])
                weight_decay = float(DEFAULT_TRAIN_SETTINGS["weight_decay"])
                warmup_epochs = float(DEFAULT_TRAIN_SETTINGS["warmup_epochs"])
                warmup_momentum = float(DEFAULT_TRAIN_SETTINGS["warmup_momentum"])
                warmup_bias_lr = float(DEFAULT_TRAIN_SETTINGS["warmup_bias_lr"])
                cos_lr = bool(DEFAULT_TRAIN_SETTINGS["cos_lr"])

                patience = int(DEFAULT_TRAIN_SETTINGS["patience"])
                close_mosaic = int(DEFAULT_TRAIN_SETTINGS["close_mosaic"])
                save_period = int(DEFAULT_TRAIN_SETTINGS["save_period"])
                seed = int(DEFAULT_TRAIN_SETTINGS["seed"])
                fraction = float(DEFAULT_TRAIN_SETTINGS["fraction"])
                nbs = int(DEFAULT_TRAIN_SETTINGS["nbs"])
                freeze = None
                multi_scale = float(DEFAULT_TRAIN_SETTINGS["multi_scale"])
                amp = bool(DEFAULT_TRAIN_SETTINGS["amp"])
                deterministic = bool(DEFAULT_TRAIN_SETTINGS["deterministic"])
                pretrained = bool(DEFAULT_TRAIN_SETTINGS["pretrained"])
                cls_remap = bool(DEFAULT_TRAIN_SETTINGS["cls_remap"])
                rect = bool(DEFAULT_TRAIN_SETTINGS["rect"])
                single_cls = bool(DEFAULT_TRAIN_SETTINGS["single_cls"])
                compile_val = bool(DEFAULT_TRAIN_SETTINGS["compile"])
                resume = bool(DEFAULT_TRAIN_SETTINGS["resume"])
                profile = bool(DEFAULT_TRAIN_SETTINGS["profile"])

                box = float(DEFAULT_TRAIN_SETTINGS["box"])
                cls_val = float(DEFAULT_TRAIN_SETTINGS["cls"])
                dfl = float(DEFAULT_TRAIN_SETTINGS["dfl"])
                cls_pw = float(DEFAULT_TRAIN_SETTINGS["cls_pw"])
                max_det = int(DEFAULT_TRAIN_SETTINGS["max_det"])

                mosaic = float(DEFAULT_TRAIN_SETTINGS["mosaic"])
                mixup = float(DEFAULT_TRAIN_SETTINGS["mixup"])
                cutmix = float(DEFAULT_TRAIN_SETTINGS["cutmix"])
                bgr = float(DEFAULT_TRAIN_SETTINGS["bgr"])
                hsv_h = float(DEFAULT_TRAIN_SETTINGS["hsv_h"])
                hsv_s = float(DEFAULT_TRAIN_SETTINGS["hsv_s"])
                hsv_v = float(DEFAULT_TRAIN_SETTINGS["hsv_v"])
                degrees = float(DEFAULT_TRAIN_SETTINGS["degrees"])
                translate = float(DEFAULT_TRAIN_SETTINGS["translate"])
                scale = float(DEFAULT_TRAIN_SETTINGS["scale"])
                shear = float(DEFAULT_TRAIN_SETTINGS["shear"])
                perspective = float(DEFAULT_TRAIN_SETTINGS["perspective"])
                fliplr = float(DEFAULT_TRAIN_SETTINGS["fliplr"])
                flipud = float(DEFAULT_TRAIN_SETTINGS["flipud"])


        except ValueError as e:
            MkMessage.error(self, f"参数数值格式不正确，请检查输入: {str(e)}")
            return

        device_raw = "0" if "0" in self.combo_device.currentText() else "cpu"

        config_dict = {
            "task": self.task_type,
            # Core
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device_raw,
            # Optimizer & LR
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "cos_lr": cos_lr,
            # Strategy
            "patience": patience,
            "close_mosaic": close_mosaic,
            "save_period": save_period,
            "seed": seed,
            "fraction": fraction,
            "nbs": nbs,
            "freeze": freeze,
            "multi_scale": multi_scale,
            "amp": amp,
            "deterministic": deterministic,
            "pretrained": pretrained,
            "cls_remap": cls_remap,
            "rect": rect,
            "single_cls": single_cls,
            "compile": compile_val,
            "resume": resume,
            "profile": profile,
            # Loss & Detection
            "box": box,
            "cls": cls_val,
            "dfl": dfl,
            "cls_pw": cls_pw,
            "max_det": max_det,
            # Augmentation
            "mosaic": mosaic,
            "mixup": mixup,
            "cutmix": cutmix,
            "bgr": bgr,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "fliplr": fliplr,
            "flipud": flipud,
        }

        if hasattr(self, "env_selector") and self.env_selector:
            config_dict["python_exe"] = self.env_selector.get_selected_python()

        if hasattr(self, "cli_viewer") and self.cli_viewer:
            config_dict["cli_command"] = self.cli_viewer.get_command_text().strip()

        # 4. Save Task to DB
        try:
            task_id = self.db.add_task(
                task_name=task_name,
                project_id=self.project['id'],
                dataset_id=dataset_id,
                model_name=model_name,
                config_dict=config_dict
            )
        except Exception as e:
            MkMessage.error(self, f"创建任务失败: {str(e)}")
            return

        # 5. Start via TaskManager
        success, msg = self.task_mgr.start_task(task_id)
        if success:
            self.created_task_id = task_id
            self.task_started.emit(task_id)
            self.accept()
        else:
            MkMessage.error(self, f"启动训练失败: {msg}")

    # ── Window Dragging & Outside Click Handling for Frameless Dialog ──
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, 'container_card') and self.container_card:
                if not self.container_card.geometry().contains(event.position().toPoint()):
                    self.reject()
                    event.accept()
                    return
            if not self._parent_widget and event.position().y() < 80:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        InfoTooltipPopover.hide_popover()
        if not self._parent_widget and event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, '_drag_pos'):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, '_drag_pos'):
            delattr(self, '_drag_pos')
        super().mouseReleaseEvent(event)

    def _disconnect_theme(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ThemeEngine.instance().themeChanged.disconnect(self.apply_theme)
        except Exception:
            pass

    def closeEvent(self, event):
        InfoTooltipPopover.hide_popover()
        self._remove_event_filter()
        self._disconnect_theme()
        super().closeEvent(event)

    def reject(self):
        if getattr(self, '_is_closing', False):
            return
        self._is_closing = True
        InfoTooltipPopover.hide_popover()
        self._remove_event_filter()
        self._disconnect_theme()
        super().reject()

    def accept(self):
        if getattr(self, '_is_closing', False):
            return
        self._is_closing = True
        InfoTooltipPopover.hide_popover()
        self._remove_event_filter()
        self._disconnect_theme()
        super().accept()

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()

        if is_dark:
            # 1. Dialog main container card
            _MkThemeEngine.apply_style_sheet(self.container_card, """
                QFrame#DialogContainer {
                    background-color: #1e1e20;
                    border: 1px solid #303036;
                    border-radius: 14px;
                }
                QFrame#DialogContainer > QWidget {
                    background-color: transparent;
                    background: transparent;
                }
                QFrame#DialogContainer QLabel {
                    background: transparent;
                    background-color: transparent;
                    border: none;
                }
            """)

            # 2. Scroll area & clean scrollbar
            _MkThemeEngine.apply_style_sheet(self.scroll, """
                QScrollArea {
                    border: none;
                    background-color: transparent;
                    background: transparent;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 6px;
                    margin: 0px 0px 0px 0px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical {
                    background: #3f3f46;
                    min-height: 25px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #52525b;
                }
                QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                    border: none;
                    background: none;
                    height: 0px;
                    width: 0px;
                }
                QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                    border: none;
                    background: none;
                    width: 0px;
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.scroll.viewport(), "background-color: transparent; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.scroll_container, "background-color: transparent; background: transparent; border: none;")

            # 3. Top Header
            _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 18px; font-weight: 700; color: #f4f4f5; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.arrow_icon, "background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "color: #a1a1aa; font-size: 13px; margin-top: -4px; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.tag_lbl, """
                background: #1e293b; 
                color: #60a5fa; 
                font-weight: 600; 
                padding: 2px 8px; 
                border-radius: 4px; 
                font-size: 11px;
                border: 1px solid #2563eb;
            """)
            self.btn_close.setIcon(PhX.bold(size=15, color="#a1a1aa"))
            _MkThemeEngine.apply_style_sheet(self.btn_close, """
                QPushButton {
                    border: none;
                    background-color: transparent;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #27272a;
                }
            """)

            # 4. Top Grid (Model & Dataset)
            self.ic_cube.setPixmap(PhCube.pixmap(size=14, color="#f4f4f5", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_model_txt, "font-size: 13px; font-weight: 600; color: #f4f4f5;")
            self.ic_database.setPixmap(PhDatabase.pixmap(size=14, color="#f4f4f5", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_dataset_txt, "font-size: 13px; font-weight: 600; color: #f4f4f5;")
            if hasattr(self, 'card_model_tip') and self.card_model_tip:
                _MkThemeEngine.apply_style_sheet(self.card_model_tip, """
                    QFrame#ModelTipCard {
                        background-color: #141416;
                        border: 1px solid #27272a;
                        border-radius: 8px;
                    }
                """)
                _MkThemeEngine.apply_style_sheet(self.lbl_tip_title, "font-size: 11px; font-weight: 600; color: #cbd5e1; border: none; background: transparent;")
                _MkThemeEngine.apply_style_sheet(self.lbl_tip_sub, "font-size: 11px; color: #71717a; border: none; background: transparent;")
            if hasattr(self, 'btn_upload_yaml') and self.btn_upload_yaml:
                self.btn_upload_yaml.setIcon(PhFolderOpen.regular(size=12, color="#f4f4f5"))
                _MkThemeEngine.apply_style_sheet(self.btn_upload_yaml, """
                    QPushButton {
                        border: 1px solid #3f3f46;
                        background-color: #27272a;
                        color: #f4f4f5;
                        font-size: 11px;
                        font-weight: 500;
                        border-radius: 4px;
                        padding: 2px 8px;
                    }
                    QPushButton:hover {
                        background-color: #3f3f46;
                        border-color: #52525b;
                    }
                """)
            if hasattr(self, 'btn_reset_yaml') and self.btn_reset_yaml:
                self.btn_reset_yaml.setIcon(PhArrowCounterClockwise.regular(size=12, color="#a1a1aa"))
                _MkThemeEngine.apply_style_sheet(self.btn_reset_yaml, """
                    QPushButton {
                        border: 1px solid #3f3f46;
                        background-color: #27272a;
                        color: #a1a1aa;
                        font-size: 11px;
                        font-weight: 500;
                        border-radius: 4px;
                        padding: 2px 8px;
                    }
                    QPushButton:hover {
                        background-color: #3f3f46;
                        color: #f4f4f5;
                        border-color: #52525b;
                    }
                """)
            if hasattr(self, 'yaml_editor') and self.yaml_editor:
                self.yaml_editor.apply_theme(True)
            if hasattr(self, 'combo_model') and hasattr(self.combo_model, 'apply_theme'):
                self.combo_model.apply_theme(True)

            # 5. Dynamic Form Labels & Dividers
            for lbl in self.field_header_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 13px; font-weight: 600; color: #f4f4f5;")
            for lbl in self.input_field_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 12px; font-weight: 500; color: #cbd5e1;")
            for ic in self.info_icons:
                ic.set_theme(True)
            for lbl in self.section_header_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 13px; font-weight: 700; color: #f1f5f9;")
            for line in self.section_lines:
                _MkThemeEngine.apply_style_sheet(line, "color: #333338; background-color: #333338; height: 1px; border: none;")

            # 6. Accordion Header Bar & Advanced Container
            if hasattr(self, "adv_header_bar") and self.adv_header_bar:
                self.adv_header_bar.apply_theme()

            if getattr(self, "_adv_built", False):
                _MkThemeEngine.apply_style_sheet(self.adv_container, """
                    QFrame#AdvContainer {
                        background-color: transparent;
                        border: none;
                    }
                    QFrame#AdvContainer QWidget {
                        background: transparent;
                    }
                """)
                if hasattr(self, "btn_reset_defaults") and self.btn_reset_defaults:
                    self.btn_reset_defaults.setIcon(PhArrowCounterClockwise.regular(size=12, color="#cbd5e1"))
                    _MkThemeEngine.apply_style_sheet(self.btn_reset_defaults, """
                        QPushButton {
                            border: 1px solid #3f3f46;
                            background-color: #27272a;
                            color: #cbd5e1;
                            font-size: 12px;
                            font-weight: 500;
                            border-radius: 6px;
                            padding: 3px 10px;
                        }
                        QPushButton:hover {
                            background-color: #3f3f46;
                            color: #ffffff;
                            border-color: #52525b;
                        }
                    """)


            # 7. Device & Environment Card
            _MkThemeEngine.apply_style_sheet(self.card_device, """
                QFrame#DeviceCard {
                    background-color: #141416;
                    border: 1px solid #2e2e32;
                    border-radius: 10px;
                }
                QFrame#DeviceCard QLabel {
                    background: transparent;
                    background-color: transparent;
                    border: none;
                }
            """)
            if hasattr(self, "pill_toggle") and self.pill_toggle:
                self.pill_toggle.set_dark_mode(True)
            if hasattr(self, "env_selector") and self.env_selector:
                self.env_selector.set_dark_mode(True)
            for cb in [getattr(self, "combo_device", None), getattr(self, "combo_imgsz", None), getattr(self, "combo_optimizer", None)]:
                if cb and hasattr(cb, "set_dark_mode"):
                    cb.set_dark_mode(True)
            _MkThemeEngine.apply_style_sheet(self.lbl_target_dev, "font-size: 12px; color: #a1a1aa; background: transparent; border: none;")
            if hasattr(self, "cli_viewer") and self.cli_viewer:
                self.cli_viewer.apply_theme(True)
            if hasattr(self, "lbl_cli_title") and self.lbl_cli_title:
                _MkThemeEngine.apply_style_sheet(self.lbl_cli_title, "font-size: 12px; font-weight: 600; color: #a1a1aa; background: transparent; border: none;")

            # 8. Footer & Action Buttons
            _MkThemeEngine.apply_style_sheet(self.footer_widget, """
                QFrame#DialogFooter {
                    border-top: 1px solid #2e2e32;
                    background-color: transparent;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_cancel, """
                QPushButton {
                    background-color: #27272a;
                    border: 1px solid #3f3f46;
                    color: #f4f4f5;
                    font-weight: 500;
                    font-size: 13px;
                    border-radius: 6px;
                    padding: 7px 18px;
                }
                QPushButton:hover {
                    background-color: #3f3f46;
                    border-color: #52525b;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_start, """
                QPushButton {
                    background-color: #2563eb;
                    border: 1px solid #2563eb;
                    color: #ffffff;
                    font-weight: 600;
                    font-size: 13px;
                    border-radius: 6px;
                    padding: 7px 22px;
                }
                QPushButton:hover {
                    background-color: #1d4ed8;
                    border-color: #1d4ed8;
                }
            """)

        else:
            # Light Mode (Default)
            # 1. Dialog main container card
            _MkThemeEngine.apply_style_sheet(self.container_card, """
                QFrame#DialogContainer {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 14px;
                }
                QFrame#DialogContainer > QWidget {
                    background-color: transparent;
                    background: transparent;
                }
                QFrame#DialogContainer QLabel {
                    background: transparent;
                    background-color: transparent;
                    border: none;
                }
            """)

            # 2. Scroll area & clean scrollbar
            _MkThemeEngine.apply_style_sheet(self.scroll, """
                QScrollArea {
                    border: none;
                    background-color: transparent;
                    background: transparent;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 6px;
                    margin: 0px 0px 0px 0px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical {
                    background: #cbd5e1;
                    min-height: 25px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #94a3b8;
                }
                QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                    border: none;
                    background: none;
                    height: 0px;
                    width: 0px;
                }
                QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                    border: none;
                    background: none;
                    width: 0px;
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.scroll.viewport(), "background-color: transparent; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.scroll_container, "background-color: transparent; background: transparent; border: none;")

            # 3. Top Header
            _MkThemeEngine.apply_style_sheet(self.header_widget, "background: transparent; background-color: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.title_lbl, "font-size: 18px; font-weight: 700; color: #09090b; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.arrow_icon, "background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.subtitle_lbl, "color: #71717a; font-size: 13px; margin-top: -4px; background: transparent; border: none;")
            _MkThemeEngine.apply_style_sheet(self.tag_lbl, """
                background: #eff6ff; 
                color: #2563eb; 
                font-weight: 600; 
                padding: 2px 8px; 
                border-radius: 4px; 
                font-size: 11px;
                border: 1px solid #dbeafe;
            """)
            self.btn_close.setIcon(PhX.bold(size=15, color="#71717a"))
            _MkThemeEngine.apply_style_sheet(self.btn_close, """
                QPushButton {
                    border: none;
                    background-color: transparent;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #f4f4f5;
                }
            """)

            # 4. Top Grid (Model & Dataset)
            self.ic_cube.setPixmap(PhCube.pixmap(size=14, color="#18181b", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_model_txt, "font-size: 13px; font-weight: 600; color: #18181b;")
            self.ic_database.setPixmap(PhDatabase.pixmap(size=14, color="#18181b", weight="bold", dpr=3.0))
            _MkThemeEngine.apply_style_sheet(self.lbl_dataset_txt, "font-size: 13px; font-weight: 600; color: #18181b;")
            if hasattr(self, 'card_model_tip') and self.card_model_tip:
                _MkThemeEngine.apply_style_sheet(self.card_model_tip, """
                    QFrame#ModelTipCard {
                        background-color: #f8fafc;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                    }
                """)
                _MkThemeEngine.apply_style_sheet(self.lbl_tip_title, "font-size: 11px; font-weight: 600; color: #475569; border: none; background: transparent;")
                _MkThemeEngine.apply_style_sheet(self.lbl_tip_sub, "font-size: 11px; color: #94a3b8; border: none; background: transparent;")
            if hasattr(self, 'btn_upload_yaml') and self.btn_upload_yaml:
                self.btn_upload_yaml.setIcon(PhFolderOpen.regular(size=12, color="#0f172a"))
                _MkThemeEngine.apply_style_sheet(self.btn_upload_yaml, """
                    QPushButton {
                        border: 1px solid #e2e8f0;
                        background-color: #ffffff;
                        color: #0f172a;
                        font-size: 11px;
                        font-weight: 500;
                        border-radius: 4px;
                        padding: 2px 8px;
                    }
                    QPushButton:hover {
                        background-color: #f8fafc;
                        border-color: #cbd5e1;
                    }
                """)
            if hasattr(self, 'btn_reset_yaml') and self.btn_reset_yaml:
                self.btn_reset_yaml.setIcon(PhArrowCounterClockwise.regular(size=12, color="#64748b"))
                _MkThemeEngine.apply_style_sheet(self.btn_reset_yaml, """
                    QPushButton {
                        border: 1px solid #e2e8f0;
                        background-color: #ffffff;
                        color: #64748b;
                        font-size: 11px;
                        font-weight: 500;
                        border-radius: 4px;
                        padding: 2px 8px;
                    }
                    QPushButton:hover {
                        background-color: #f8fafc;
                        color: #0f172a;
                        border-color: #cbd5e1;
                    }
                """)
            if hasattr(self, 'yaml_editor') and self.yaml_editor:
                self.yaml_editor.apply_theme(False)
            if hasattr(self, 'combo_model') and hasattr(self.combo_model, 'apply_theme'):
                self.combo_model.apply_theme(False)

            # 5. Dynamic Form Labels & Dividers
            for lbl in self.field_header_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 13px; font-weight: 600; color: #18181b;")
            for lbl in self.input_field_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 12px; font-weight: 500; color: #334155;")
            for ic in self.info_icons:
                ic.set_theme(False)
            for lbl in self.section_header_labels:
                _MkThemeEngine.apply_style_sheet(lbl, "font-size: 13px; font-weight: 700; color: #0f172a;")
            for line in self.section_lines:
                _MkThemeEngine.apply_style_sheet(line, "color: #e2e8f0; background-color: #e2e8f0; height: 1px; border: none;")

            # 6. Accordion Header Bar & Advanced Container
            if hasattr(self, "adv_header_bar") and self.adv_header_bar:
                self.adv_header_bar.apply_theme()

            if getattr(self, "_adv_built", False):
                _MkThemeEngine.apply_style_sheet(self.adv_container, """
                    QFrame#AdvContainer {
                        background-color: transparent;
                        border: none;
                    }
                    QFrame#AdvContainer QWidget {
                        background: transparent;
                    }
                """)
                if hasattr(self, "btn_reset_defaults") and self.btn_reset_defaults:
                    self.btn_reset_defaults.setIcon(PhArrowCounterClockwise.regular(size=12, color="#64748b"))
                    _MkThemeEngine.apply_style_sheet(self.btn_reset_defaults, """
                        QPushButton {
                            border: 1px solid #e2e8f0;
                            background-color: #ffffff;
                            color: #64748b;
                            font-size: 12px;
                            font-weight: 500;
                            border-radius: 6px;
                            padding: 3px 10px;
                        }
                        QPushButton:hover {
                            background-color: #f8fafc;
                            color: #0f172a;
                            border-color: #cbd5e1;
                        }
                    """)


            # 7. Device & Environment Card
            _MkThemeEngine.apply_style_sheet(self.card_device, """
                QFrame#DeviceCard {
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                }
                QFrame#DeviceCard QLabel {
                    background: transparent;
                    background-color: transparent;
                    border: none;
                }
            """)
            if hasattr(self, "pill_toggle") and self.pill_toggle:
                self.pill_toggle.set_dark_mode(False)
            if hasattr(self, "env_selector") and self.env_selector:
                self.env_selector.set_dark_mode(False)
            for cb in [getattr(self, "combo_device", None), getattr(self, "combo_imgsz", None), getattr(self, "combo_optimizer", None)]:
                if cb and hasattr(cb, "set_dark_mode"):
                    cb.set_dark_mode(False)
            _MkThemeEngine.apply_style_sheet(self.lbl_target_dev, "font-size: 12px; color: #64748b; background: transparent; border: none;")
            if hasattr(self, "cli_viewer") and self.cli_viewer:
                self.cli_viewer.apply_theme(False)
            if hasattr(self, "lbl_cli_title") and self.lbl_cli_title:
                _MkThemeEngine.apply_style_sheet(self.lbl_cli_title, "font-size: 12px; font-weight: 600; color: #64748b; background: transparent; border: none;")

            # 8. Footer & Action Buttons
            _MkThemeEngine.apply_style_sheet(self.footer_widget, """
                QFrame#DialogFooter {
                    border-top: 1px solid #f1f5f9;
                    background-color: transparent;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_cancel, """
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    color: #0f172a;
                    font-weight: 500;
                    font-size: 13px;
                    border-radius: 6px;
                    padding: 7px 18px;
                }
                QPushButton:hover {
                    background-color: #f8fafc;
                    border-color: #cbd5e1;
                }
            """)
            _MkThemeEngine.apply_style_sheet(self.btn_start, """
                QPushButton {
                    background-color: #09090b;
                    border: 1px solid #09090b;
                    color: #ffffff;
                    font-weight: 600;
                    font-size: 13px;
                    border-radius: 6px;
                    padding: 7px 22px;
                }
                QPushButton:hover {
                    background-color: #27272a;
                    border-color: #27272a;
                }
            """)

        apply_monkeyqt_theme(self)
