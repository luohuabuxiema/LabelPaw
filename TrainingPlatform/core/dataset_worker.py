# -*- coding: utf-8 -*-
"""
Dataset Worker and Ratio Utility for Dataset Tooling
Extracted & Refactored for AI Vision Platform
"""
import os
import random
import shutil
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox
from monkeyqt import MkForm, MkButton, MkMessageBox, apply_monkeyqt_theme

class DatasetWorker(QThread):
    log_signal = Signal(str)
    finish_signal = Signal(bool, str)

    def __init__(self, mode, img_dir, ann_dir, out_dir, ratios):
        super().__init__()
        self.mode = mode
        self.img_dir = img_dir
        self.ann_dir = ann_dir if ann_dir else img_dir
        self.out_dir = out_dir
        self.train_r, self.val_r, self.test_r = ratios

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            self.log("[INFO] 开始数据集处理任务...")
            self.log(f"[INFO] 图片目录: {self.img_dir}")
            self.log(f"[INFO] 标签目录: {self.ann_dir}")
            self.log(f"[INFO] 输出目录: {self.out_dir}")
            self.log(f"[INFO] 划分比例: Train({self.train_r*100:.0f}%), Val({self.val_r*100:.0f}%), Test({self.test_r*100:.0f}%)")
            self.log("-" * 50)

            if self.mode == "UNET":
                self.process_unet()
            elif self.mode == "YOLO_SPLIT":
                self.process_yolo_split()
            elif self.mode == "TO_YOLO":
                self.process_convert_to_yolo()
            elif self.mode == "YOLO_TO_UNET":
                self.process_yolo_to_unet()

        except Exception as e:
            self.log(f"[ERROR] 运行错误: {str(e)}")
            self.finish_signal.emit(False, str(e))

    def get_valid_pairs(self, exts, label_exts, use_json_xml_mix=False):
        valid_pairs = []
        if not os.path.exists(self.img_dir):
            return valid_pairs
        for filename in os.listdir(self.img_dir):
            if filename.lower().endswith(exts):
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.img_dir, filename)

                if use_json_xml_mix:
                    json_path = os.path.join(self.ann_dir, f"{base_name}.json")
                    xml_path = os.path.join(self.ann_dir, f"{base_name}.xml")
                    if os.path.exists(json_path):
                        valid_pairs.append((img_path, json_path, 'json'))
                    elif os.path.exists(xml_path):
                        valid_pairs.append((img_path, xml_path, 'xml'))
                else:
                    for l_ext in label_exts:
                        l_path = os.path.join(self.ann_dir, f"{base_name}{l_ext}")
                        if os.path.exists(l_path):
                            valid_pairs.append((img_path, l_path))
                            break
        return valid_pairs

    def split_data(self, valid_pairs):
        random.shuffle(valid_pairs)
        total = len(valid_pairs)
        
        train_c = int(total * self.train_r)
        val_c = int(total * self.val_r)
        test_c = int(total * self.test_r)

        remainder = total - (train_c + val_c + test_c)
        if remainder > 0:
            if self.train_r >= max(self.val_r, self.test_r):
                train_c += remainder
            elif self.val_r >= max(self.train_r, self.test_r):
                val_c += remainder
            else:
                test_c += remainder

        train_p = valid_pairs[:train_c]
        val_p = valid_pairs[train_c:train_c + val_c]
        test_p = valid_pairs[train_c + val_c:train_c + val_c + test_c]
        return train_p, val_p, test_p

    def process_unet(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp', '.jpeg'), ('.json',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片与 JSON 标注！")
            return

        self.log(f"[INFO] 找到 {len(valid_pairs)} 组有效数据。正在从 JSON 提取类别...")
        class_mapping = {"background": 0}
        unique_classes = set()
        for _, j_path in valid_pairs:
            with open(j_path, 'r', encoding='utf-8') as f:
                for s in json.load(f).get('shapes', []):
                    if s.get('label'): unique_classes.add(s['label'])
        for c in sorted(list(unique_classes)):
            class_mapping[c] = len(class_mapping)
        self.log(f"[INFO] 类别映射: {class_mapping}")

        palette = [0, 0, 0]
        for i in range(1, 256):
            if i < len(class_mapping):
                palette.extend([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            else:
                palette.extend([0, 0, 0])

        train_p, val_p, test_p = self.split_data(valid_pairs)

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'masks', split_name), exist_ok=True)
            self.log(f"[INFO] 生成 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, json_p in pairs:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                with open(json_p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mask = np.zeros((data.get('imageHeight', 100), data.get('imageWidth', 100)), dtype=np.uint8)
                for shape in data.get('shapes', []):
                    cid = class_mapping.get(shape.get('label'))
                    if cid is None: continue
                    pts = np.array(shape.get('points', []), np.int32)
                    if shape.get('shape_type') == 'rectangle' and len(pts) == 2:
                        cv2.rectangle(mask, pts[0], pts[1], cid, -1)
                    elif len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], cid)
                    elif shape.get('shape_type') == 'point':
                        cv2.circle(mask, pts[0], 5, cid, -1)

                from PIL import Image
                pil_mask = Image.fromarray(mask, mode='P')
                pil_mask.putpalette(palette)
                mask_save_path = os.path.join(self.out_dir, 'masks', split_name, f"{base_name}.png")
                pil_mask.save(mask_save_path)

        self.log("\n[SUCCESS] U-Net 格式数据集生成完毕！")
        self.finish_signal.emit(True, "U-Net 数据集处理成功")

    def generate_yaml(self, class_map):
        is_pose = getattr(self, 'is_pose_dataset', False)
        filename = "data-pose.yaml" if is_pose else "data.yaml"
        yaml_path = os.path.join(self.out_dir, filename)
        abs_out_dir = os.path.abspath(self.out_dir).replace("\\", "/")

        train_path = f"{abs_out_dir}/images/train"
        val_path = f"{abs_out_dir}/images/val"
        test_path = f"{abs_out_dir}/images/test"

        nc = len(class_map)
        names_dict = {v: k for k, v in class_map.items()}
        yaml_content = []
        yaml_content.append(f"# Train and Val data path")
        yaml_content.append(f"train: {train_path}")
        yaml_content.append(f"val: {val_path}")
        if self.test_r > 0:
            yaml_content.append(f"test: {test_path}")

        yaml_content.append(f"\n# Number of classes")
        yaml_content.append(f"nc: {nc}")

        yaml_content.append(f"\n# Class names")
        yaml_content.append(f"names:")
        for i in range(nc):
            yaml_content.append(f"  {i}: {names_dict[i]}")

        if is_pose:
            yaml_content.append(f"\n# Keypoints")
            kpt_shape = getattr(self, 'kpt_shape', [0, 3])
            yaml_content.append(f"kpt_shape: {kpt_shape}")
            flip_idx = list(range(kpt_shape[0]))
            for name in class_map.keys():
                if "person" in name.lower() or "person (coco)" in name.lower():
                    if kpt_shape[0] == 17:
                        flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
                    break
            yaml_content.append(f"flip_idx: {flip_idx}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yaml_content))

        self.log(f"[INFO] 已自动生成配置文件: {yaml_path}")

    def process_yolo_split(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp', '.jpeg'), ('.txt',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片与 TXT 标注！")
            return

        class_map = {}
        classes_txt = os.path.join(self.ann_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip(): class_map[line.strip()] = i
        else:
            class_map = {"class_0": 0}

        self.log(f"[INFO] 找到 {len(valid_pairs)} 组有效数据。")
        train_p, val_p, test_p = self.split_data(valid_pairs)

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"[INFO] 拷贝 {split_name} 集 ({len(pairs)} 张)...")
            for img_p, txt_p in pairs:
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))
                shutil.copy(txt_p, os.path.join(self.out_dir, 'labels', split_name, os.path.basename(txt_p)))

        self.generate_yaml(class_map)
        self.log("\n[SUCCESS] YOLO 数据集划分完成！")
        self.finish_signal.emit(True, "YOLO 数据集划分成功")

    def process_convert_to_yolo(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp', '.jpeg'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片与 JSON/XML 标注！")
            return

        self.log(f"[INFO] 找到 {len(valid_pairs)} 组有效数据，正在提取类别...")
        class_map = {}
        unique_classes = set()
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    for s in json.load(f).get('shapes', []):
                        if s.get('label'): unique_classes.add(s['label'])
            elif l_type == 'xml':
                for obj in ET.parse(l_path).getroot().findall('object'):
                    if obj.find('name') is not None: unique_classes.add(obj.find('name').text)
        for c in sorted(list(unique_classes)):
            class_map[c] = len(class_map)
        self.log(f"[INFO] 类别映射: {class_map}")

        self.is_pose_dataset = False
        self.kpt_shape = [0, 3]

        train_p, val_p, test_p = self.split_data(valid_pairs)
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"[INFO] 处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]

                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))
                lines = []

                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            cid = class_map.get(s.get('label'))
                            if cid is None: continue
                            pts = s.get('points', [])
                            shape_type = s.get('shape_type', 'polygon')

                            if shape_type == 'rectangle' and len(pts) == 2:
                                cx, cy = (pts[0][0] + pts[1][0]) / 2 / w, (pts[0][1] + pts[1][1]) / 2 / h
                                bw, bh = abs(pts[1][0] - pts[0][0]) / w, abs(pts[1][1] - pts[0][1]) / h
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                            elif shape_type == 'obb' and pts:
                                obb_pts = pts[:4]
                                pts_normalized = []
                                for pt in obb_pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))
                            elif shape_type == 'polygon' and pts:
                                pts_normalized = []
                                for pt in pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))
                            elif shape_type == 'pose':
                                rect = s.get('rect')
                                kps = s.get('keypoints', [])
                                if not rect or not kps: continue
                                self.is_pose_dataset = True
                                self.kpt_shape = [len(kps), 3]
                                val0, val1, bw, bh = rect
                                kpts_x = [kp[0] for kp in kps if len(kp) >= 2]
                                if kpts_x:
                                    min_kp_x = min(kpts_x)
                                    max_kp_x = max(kpts_x)
                                    mid_kp_x = (min_kp_x + max_kp_x) / 2
                                    if val0 < mid_kp_x - (max_kp_x - min_kp_x) * 0.15:
                                        cx = val0 + bw / 2
                                        cy = val1 + bh / 2
                                    else:
                                        cx, cy = val0, val1
                                else:
                                    cx, cy = val0, val1

                                cx, cy = cx / w, cy / h
                                bw, bh = bw / w, bh / h
                                
                                kps_normalized = []
                                for kp in kps:
                                    kx, ky, vis = kp
                                    kps_normalized.extend([f"{kx/w:.6f}", f"{ky/h:.6f}", f"{vis}"])
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kps_normalized))
                            elif shape_type == 'point' and len(pts) == 1:
                                cx, cy = pts[0][0] / w, pts[0][1] / h
                                pw, ph = 0.02, 0.02
                                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                                cy = max(ph / 2, min(1.0 - ph / 2, cy))
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}")

                elif l_type == 'xml':
                    root = ET.parse(label_p).getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        cid = class_map.get(label)
                        if cid is None: continue

                        bndbox = obj.find('bndbox')
                        polygon = obj.find('polygon')
                        robndbox = obj.find('robndbox')

                        if bndbox is not None:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            cx, cy = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
                            bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        elif polygon is not None:
                            pts_normalized = []
                            for pt in polygon.findall('pt'):
                                px = float(pt.find('x').text) / w
                                py = float(pt.find('y').text) / h
                                pts_normalized.extend([f"{px:.6f}", f"{py:.6f}"])
                            if pts_normalized:
                                lines.append(f"{cid} " + " ".join(pts_normalized))
                        elif robndbox is not None:
                            cx = float(robndbox.find('cx').text) / w
                            cy = float(robndbox.find('cy').text) / h
                            bw = float(robndbox.find('w').text) / w
                            bh = float(robndbox.find('h').text) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]): f.write(f"{c}\n")

        self.generate_yaml(class_map)
        self.log("\n[SUCCESS] JSON/XML 转 YOLO 划分完成！")
        self.finish_signal.emit(True, "格式转换与划分成功")

    def process_yolo_to_unet(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp', '.jpeg'), ('.txt',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片与 TXT 标注！")
            return

        self.log(f"[INFO] 找到 {len(valid_pairs)} 组有效数据。")
        class_map = {}
        classes_txt = os.path.join(self.ann_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip(): class_map[i] = line.strip()
        else:
            max_class_id = 0
            for _, txt_p in valid_pairs:
                with open(txt_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            if cid > max_class_id:
                                max_class_id = cid
            for i in range(max_class_id + 1):
                class_map[i] = f"class_{i}"

        self.log(f"[INFO] 类别映射: {class_map}")

        palette = [0, 0, 0]
        for i in range(1, 256):
            if i <= len(class_map):
                palette.extend([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            else:
                palette.extend([0, 0, 0])

        train_p, val_p, test_p = self.split_data(valid_pairs)

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'masks', split_name), exist_ok=True)
            self.log(f"[INFO] 生成 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, txt_p in pairs:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]

                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))
                mask = np.zeros((h, w), dtype=np.uint8)

                with open(txt_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            unet_cid = cid + 1
                            if len(parts) == 5:
                                cx, cy, bw, bh = map(float, parts[1:])
                                xmin = int((cx - bw / 2) * w)
                                ymin = int((cy - bh / 2) * h)
                                xmax = int((cx + bw / 2) * w)
                                ymax = int((cy + bh / 2) * h)
                                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), unet_cid, -1)
                            else:
                                pts_normalized = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                                pts_pixel = pts_normalized * np.array([w, h])
                                pts_pixel = pts_pixel.astype(np.int32)
                                cv2.fillPoly(mask, [pts_pixel], unet_cid)

                from PIL import Image
                pil_mask = Image.fromarray(mask, mode='P')
                pil_mask.putpalette(palette)
                mask_save_path = os.path.join(self.out_dir, 'masks', split_name, f"{base_name}.png")
                pil_mask.save(mask_save_path)

        self.log("\n[SUCCESS] YOLO 分割转 U-Net 掩码生成完毕！")
        self.finish_signal.emit(True, "YOLO 分割转 U-Net 处理成功")


class RatioDialog(QDialog):
    def __init__(self, current_ratios, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置数据集划分比例")
        self.resize(380, 260)
        self.setStyleSheet("background-color: white;")

        self.ratios = current_ratios
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header_label = QLabel("请分配数据集百分比（总和必须为 100%）")
        header_label.setStyleSheet("color: #909399; font-size: 13px;")
        layout.addWidget(header_label)

        form = MkForm()
        self.train_box = self.create_spinbox(current_ratios[0] * 100)
        self.val_box = self.create_spinbox(current_ratios[1] * 100)
        self.test_box = self.create_spinbox(current_ratios[2] * 100)

        form.add_row("训练集 (Train):", self.train_box)
        form.add_row("验证集 (Val):", self.val_box)
        form.add_row("测试集 (Test):", self.test_box)
        layout.addWidget(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = MkButton("取消")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = MkButton("保存设置")
        self.btn_ok.clicked.connect(self.check_and_accept)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

        apply_monkeyqt_theme(self)

    def create_spinbox(self, val):
        box = QSpinBox()
        box.setRange(0, 100)
        box.setValue(int(val))
        box.setSuffix(" %")
        box.setStyleSheet("""
            QSpinBox {
                border: 1px solid #dcdfe6;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 13px;
                min-height: 28px;
            }
            QSpinBox:focus {
                border-color: #409eff;
            }
        """)
        return box

    def check_and_accept(self):
        t, v, te = self.train_box.value(), self.val_box.value(), self.test_box.value()
        if t + v + te != 100:
            MkMessageBox.warning(self, "错误提示", "三个划分比例之和必须精确等于 100%！", cancel_text=None)
            return
        self.ratios = (t / 100.0, v / 100.0, te / 100.0)
        self.accept()
