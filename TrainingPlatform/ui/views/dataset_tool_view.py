# -*- coding: utf-8 -*-
"""
MonkeyQt Refactored Dataset Tool View for AI Vision Platform
Provides 4 key dataset processing sub-tools:
1. TO_YOLO: JSON/XML 转 YOLO 格式与划分
2. YOLO_SPLIT: YOLO 纯标注与图片划分
3. UNET: JSON 转 U-Net 掩码
4. YOLO_TO_UNET: YOLO 分割转 U-Net 掩码
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFileDialog, QTextEdit, QMessageBox)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCursor

from monkeyqt import MkCard, MkButton, MkInput, MkComboBox, MkForm, MkAlert, ThemeEngine, MkQWidget
from monkeyqt_icons import PhScales, PhPlay
from TrainingPlatform.core.dataset_worker import DatasetWorker, RatioDialog

class DatasetToolView(MkQWidget):
    def __init__(self, default_mode="TO_YOLO", parent=None):
        super().__init__(parent=parent, role="transparent")
        self.ratios = (0.8, 0.2, 0.0)
        self.worker = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header Title
        self.title_lbl = QLabel("高级数据集转换与划分工具箱 Dataset Tool")
        main_layout.addWidget(self.title_lbl)

        # Card 1: Configuration Form Card
        cfg_card = MkCard(title="")
        c_layout = cfg_card.content_layout

        self.input_img_dir = MkInput()
        self.input_img_dir.setPlaceholderText("选择/填写包含原图的文件夹路径...")
        btn_browse_img = MkButton("选择目录")
        btn_browse_img.clicked.connect(lambda: self.browse_dir(self.input_img_dir))
        box_img = QHBoxLayout()
        box_img.addWidget(self.input_img_dir)
        box_img.addWidget(btn_browse_img)

        self.input_ann_dir = MkInput()
        self.input_ann_dir.setPlaceholderText("(可选) 为空则代表标签与图片在同一个目录中")
        btn_browse_ann = MkButton("选择目录")
        btn_browse_ann.clicked.connect(lambda: self.browse_dir(self.input_ann_dir))
        box_ann = QHBoxLayout()
        box_ann.addWidget(self.input_ann_dir)
        box_ann.addWidget(btn_browse_ann)

        self.input_out_dir = MkInput()
        self.input_out_dir.setPlaceholderText("选择/填写处理后数据集的输出保存目录...")
        btn_browse_out = MkButton("选择目录")
        btn_browse_out.clicked.connect(lambda: self.browse_dir(self.input_out_dir))
        box_out = QHBoxLayout()
        box_out.addWidget(self.input_out_dir)
        box_out.addWidget(btn_browse_out)

        self.combo_mode = MkComboBox()
        self.combo_mode.addItems([
            "TO_YOLO (JSON/XML 标注转 YOLO 格式与划分)",
            "YOLO_SPLIT (YOLO 纯标注与图片打乱划分)",
            "UNET (JSON 标注转 U-Net Mask 掩码)",
            "YOLO_TO_UNET (YOLO 分割/框转 U-Net Mask 掩码)"
        ])
        
        # Set default mode if specified
        mode_idx_map = {"TO_YOLO": 0, "YOLO_SPLIT": 1, "UNET": 2, "YOLO_TO_UNET": 3}
        if default_mode in mode_idx_map:
            self.combo_mode.setCurrentIndex(mode_idx_map[default_mode])

        self.btn_ratio = MkButton("划分比例: 80% Train | 20% Val")
        self.btn_ratio.setIcon(PhScales.regular())
        self.btn_ratio.clicked.connect(self.open_ratio_dialog)

        box_opts = QHBoxLayout()
        box_opts.addWidget(self.combo_mode, 1)
        box_opts.addWidget(self.btn_ratio)

        form = MkForm()
        form.add_row("图片目录:", box_img)
        form.add_row("标签目录:", box_ann)
        form.add_row("输出目录:", box_out)
        form.add_row("处理模式配置:", box_opts)
        c_layout.addWidget(form)

        main_layout.addWidget(cfg_card)

        # Start Processing Button
        self.btn_start = MkButton("开始执行转换与数据集划分")
        self.btn_start.setIcon(PhPlay.regular())
        self.btn_start.setStyleSheet("font-size: 16px; font-weight: bold; padding: 12px;")
        self.btn_start.clicked.connect(self.start_processing)
        main_layout.addWidget(self.btn_start)

        # Card 2: Log Console
        log_card = MkCard(title="数据处理实时控制台 (Log Console)")
        l_layout = log_card.content_layout

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #67c23a;
                font-family: Consolas, monospace;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }
        """)
        l_layout.addWidget(self.console)
        main_layout.addWidget(log_card, 1)

        self.apply_theme()
        ThemeEngine.instance().themeChanged.connect(self.apply_theme)

    def apply_theme(self, *args):
        is_dark = ThemeEngine.is_dark()
        if is_dark:
            self.title_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        else:
            self.title_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #0f172a;")

    def set_mode(self, mode_key):
        mode_idx_map = {"TO_YOLO": 0, "YOLO_SPLIT": 1, "UNET": 2, "YOLO_TO_UNET": 3}
        if mode_key in mode_idx_map:
            self.combo_mode.setCurrentIndex(mode_idx_map[mode_key])

    def browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            line_edit.setText(d)

    def open_ratio_dialog(self):
        dlg = RatioDialog(self.ratios, self)
        if dlg.exec():
            self.ratios = dlg.ratios
            t, v, te = [int(r * 100) for r in self.ratios]
            txt = f"比例: {t}% Train | {v}% Val"
            if te > 0:
                txt += f" | {te}% Test"
            self.btn_ratio.setText(txt)

    def append_log(self, msg):
        self.console.append(msg)
        self.console.moveCursor(QTextCursor.End)

    def start_processing(self):
        img_dir = self.input_img_dir.text().strip()
        ann_dir = self.input_ann_dir.text().strip()
        out_dir = self.input_out_dir.text().strip()

        if not img_dir or not out_dir:
            QMessageBox.warning(self, "未完成", "请先选择有效的图片目录和输出存放目录！")
            return

        mode_raw = self.combo_mode.currentText().split()[0]

        self.console.clear()
        self.btn_start.setEnabled(False)
        self.btn_start.setText("后台处理中，请稍候...")

        self.worker = DatasetWorker(mode_raw, img_dir, ann_dir, out_dir, self.ratios)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finish_signal.connect(self.on_process_finish)
        self.worker.start()

    def on_process_finish(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始执行转换与数据集划分")
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "错误", msg)
