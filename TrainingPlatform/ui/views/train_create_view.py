# -*- coding: utf-8 -*-
"""
New Training Task Configuration View for AI Vision Training Platform
Using MonkeyQt MkForm, MkInput, MkComboBox, MkSlider, MkButton
"""
from monkeyqt.themes.engine import ThemeEngine as _MkThemeEngine
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QScrollArea, QMessageBox)
from PySide6.QtCore import Signal, Qt

from monkeyqt import (MkCard, MkButton, MkInput, MkComboBox,
                      MkForm, MkSlider, MkSwitch, MkAlert, MkQWidget)
from monkeyqt_icons import PhPlay
from TrainingPlatform.ui.components.env_selector import RoundedComboBox

class TrainCreateView(MkQWidget):
    task_created = Signal(int)  # Emits new task_id when created & started

    def __init__(self, db_manager, task_manager, parent=None):
        super().__init__(parent=parent, role="transparent")
        self.db = db_manager
        self.task_mgr = task_manager

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title_lbl = QLabel("创建训练任务 New Training Task")
        _MkThemeEngine.apply_style_sheet(title_lbl, "font-size: 20px; font-weight: bold; color: #303133;")
        main_layout.addWidget(title_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        _MkThemeEngine.apply_style_sheet(scroll, "QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setSpacing(20)

        # 1. Base Config Card
        card_base = MkCard(title="1. 基础场景配置 (Base Configuration)")
        l_base = card_base.content_layout

        self.input_task_name = MkInput()
        self.input_task_name.setPlaceholderText("训练任务名称 (例: YOLO11_PCB_v1)")

        self.combo_project = MkComboBox()
        self.combo_project.currentIndexChanged.connect(self.on_project_changed)

        self.combo_dataset = MkComboBox()

        self.combo_model = MkComboBox()
        self.combo_model.addItems([
            "yolov8n.pt (YOLOv8 Nano)",
            "yolov8s.pt (YOLOv8 Small)",
            "yolo11n.pt (YOLO11 Nano)",
            "yolo11s.pt (YOLO11 Small)",
            "yolo26n.pt (YOLO26 Nano)",
            "yolo26s.pt (YOLO26 Small)"
        ])

        self.combo_device = RoundedComboBox()
        self.combo_device.addItems(["0 (GPU 0)", "CPU"])

        form_base = MkForm()
        form_base.add_row("任务名称:", self.input_task_name)
        form_base.add_row("所属项目:", self.combo_project)
        form_base.add_row("选择数据集:", self.combo_dataset)
        form_base.add_row("模型架构/权重:", self.combo_model)
        form_base.add_row("硬件设备 (Device):", self.combo_device)
        l_base.addWidget(form_base)

        c_layout.addWidget(card_base)

        # 2. Hyperparameters Card
        card_hyper = MkCard(title="2. 核心超参数配置 (Hyperparameters)")
        l_hyper = card_hyper.content_layout

        self.input_epochs = MkInput()
        self.input_epochs.setText("100")

        self.input_batch = MkInput()
        self.input_batch.setText("16")

        self.combo_imgsz = MkComboBox()
        self.combo_imgsz.addItems(["640", "320", "416", "512", "1024"])

        self.combo_optimizer = MkComboBox()
        self.combo_optimizer.addItems(["AdamW", "SGD", "Adam", "RMSProp"])

        self.input_lr0 = MkInput()
        self.input_lr0.setText("0.01")

        self.input_workers = MkInput()
        self.input_workers.setText("4")

        self.input_patience = MkInput()
        self.input_patience.setText("50")

        form_hyper = MkForm()
        form_hyper.add_row("训练轮数 (Epochs):", self.input_epochs)
        form_hyper.add_row("批次大小 (Batch Size):", self.input_batch)
        form_hyper.add_row("图像分辨率 (Img Size):", self.combo_imgsz)
        form_hyper.add_row("优化器 (Optimizer):", self.combo_optimizer)
        form_hyper.add_row("初始学习率 (lr0):", self.input_lr0)
        form_hyper.add_row("数据加载线程 (Workers):", self.input_workers)
        form_hyper.add_row("早停轮数 (Patience):", self.input_patience)
        l_hyper.addWidget(form_hyper)

        c_layout.addWidget(card_hyper)

        # 3. Data Augmentation Card
        card_aug = MkCard(title="3. 数据增强配置 (Data Augmentation)")
        l_aug = card_aug.content_layout

        self.slider_mosaic = MkSlider()
        self.slider_mosaic.setValue(100)  # 1.0

        self.slider_mixup = MkSlider()
        self.slider_mixup.setValue(0)    # 0.0

        self.slider_flip = MkSlider()
        self.slider_flip.setValue(50)    # 0.5

        form_aug = MkForm()
        form_aug.add_row("Mosaic 拼接增强强度:", self.slider_mosaic)
        form_aug.add_row("Mixup 混合增强强度:", self.slider_mixup)
        form_aug.add_row("Horizontal Flip 水平翻转:", self.slider_flip)
        l_aug.addWidget(form_aug)

        c_layout.addWidget(card_aug)

        # Start Button
        self.btn_start = MkButton("一键开启 YOLO 模型训练")
        self.btn_start.setIcon(PhPlay.regular())
        _MkThemeEngine.apply_style_sheet(self.btn_start, "font-size: 16px; font-weight: bold; padding: 12px;")
        self.btn_start.clicked.connect(self.start_training)
        c_layout.addWidget(self.btn_start)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.refresh_options()

    def refresh_options(self):
        projects = self.db.get_projects()
        self.combo_project.blockSignals(True)
        self.combo_project.clear()
        for p in projects:
            self.combo_project.addItem(f"[{p['id']}] {p['name']} ({p['task_type']})", p['id'])
        self.combo_project.blockSignals(False)

        self.on_project_changed()

    def on_project_changed(self):
        idx = self.combo_project.currentIndex()
        if idx < 0:
            self.combo_dataset.clear()
            return
        pid = self.combo_project.itemData(idx)
        datasets = self.db.get_datasets(project_id=pid)
        self.combo_dataset.clear()
        for d in datasets:
            self.combo_dataset.addItem(f"[{d['id']}] {d['name']} ({d['image_count']} 张图)", d['id'])

    def start_training(self):
        task_name = self.input_task_name.text().strip()
        if not task_name:
            QMessageBox.warning(self, "警告", "请输入训练任务名称！")
            return

        pid_idx = self.combo_project.currentIndex()
        did_idx = self.combo_dataset.currentIndex()
        if pid_idx < 0 or did_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择有效的训练项目与数据集！")
            return

        project_id = self.combo_project.itemData(pid_idx)
        dataset_id = self.combo_dataset.itemData(did_idx)
        model_name = self.combo_model.currentText().split()[0]
        device_raw = "0" if "0" in self.combo_device.currentText() else "cpu"

        try:
            epochs = int(self.input_epochs.text().strip())
            batch = int(self.input_batch.text().strip())
            imgsz = int(self.combo_imgsz.currentText())
            optimizer = self.combo_optimizer.currentText()
            lr0 = float(self.input_lr0.text().strip())
            workers = int(self.input_workers.text().strip())
            patience = int(self.input_patience.text().strip())
        except ValueError as e:
            QMessageBox.critical(self, "错误", f"参数格式不正确，请确保数值合法: {str(e)}")
            return

        config_dict = {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device_raw,
            "optimizer": optimizer,
            "lr0": lr0,
            "workers": workers,
            "patience": patience,
            "mosaic": self.slider_mosaic.value() / 100.0,
            "mixup": self.slider_mixup.value() / 100.0,
            "fliplr": self.slider_flip.value() / 100.0,
        }

        # Save to DB
        task_id = self.db.add_task(
            task_name=task_name,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            config_dict=config_dict
        )

        # Start via TaskManager
        success, msg = self.task_mgr.start_task(task_id)
        if success:
            QMessageBox.information(self, "已启动", f"训练任务 [{task_name}] 已成功添加到训练队列并开始执行！")
            self.task_created.emit(task_id)
        else:
            QMessageBox.critical(self, "启动失败", msg)
