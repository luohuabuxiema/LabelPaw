import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox, QLabel, \
    QListWidgetItem, QDialog, QMenu, QAbstractItemView, QProgressBar, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, \
    QCheckBox, QListWidget, QFrame, QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QThread, Signal, QSize, QEvent, QSettings, QTimer
from PySide6.QtGui import QPainter, QIcon, QPixmap, QColor, QAction, QActionGroup, QPolygonF, QMovie
from main_dataset_tool import DatasetToolWindow
import cv2
import numpy as np
from PIL import Image
import torch
try:
    from ui.author_info import AuthorInfoDialog
except ImportError:
    pass
from ui.main_window import Ui_MainWindow, TemplateSelectorWidget, FormatSelectorWidget
from ui.template_dialog import SkeletonTemplateDialog
from ui.model_selector_dialog import ModelSelectorDialog
from ui.theme import DARK_THEME, LIGHT_THEME
from labelpaw.graphics.canvas import Canvas, CanvasMode
from labelpaw.models.sam_client import SAMClient, SAM_MODEL_MAP
from labelpaw.data.exporter import Exporter
from labelpaw.graphics.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape
from labelpaw.config.pose_template import TemplateManager
from utils.message import DialogOver
from labelpaw.models.yolo_predictor import YoloPredictorWorker


class SamBatchWorker(QThread):
    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(int, int)  # processed, total
    error = Signal(str)

    def __init__(self, processor, model, img_paths, prompts, current_format, class_list, canvas_mode, overwrite=False):
        super().__init__()
        self.processor = processor
        self.model = model
        self.img_paths = img_paths
        self.prompts = prompts
        self.current_format = current_format
        self.class_list = list(class_list)
        self.canvas_mode = canvas_mode
        self.is_cancelled = False
        self.overwrite = overwrite

    def run(self):
        total = len(self.img_paths)
        processed = 0

        try:
            for idx, img_path in enumerate(self.img_paths):
                if self.is_cancelled:
                    break

                filename = os.path.basename(img_path)
                self.progress.emit(idx, total, filename)

                # 1. 加载图像并获取宽高
                pil_img = Image.open(img_path).convert("RGB")
                w_img, h_img = pil_img.size

                # 2. 提取特征
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    state = self.processor.set_image(pil_img)

                # 3. 针对每个提示词逐一推理并整合结果
                shapes_data = []
                for prompt in self.prompts:
                    if self.is_cancelled:
                        break

                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_state = self.processor.set_text_prompt(prompt=prompt, state=state)

                        masks = out_state.get("masks", [])
                        scores = out_state.get("scores", [])
                        boxes = out_state.get("boxes", [])

                        if len(masks) > 0:
                            for i in range(len(masks)):
                                mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                                mask_np = np.squeeze(mask_np)

                                score_val = float(scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i])
                                box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]

                                if box.ndim > 1:
                                    box = box.squeeze()
                                x1, y1, x2, y2 = box
                                rect_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                                mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                poly_pts = []
                                rect_obb = []
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                                    poly_pts = approx.reshape(-1, 2).tolist()

                                    obb = cv2.minAreaRect(largest_contour)
                                    rect_obb = [float(obb[0][0]), float(obb[0][1]), float(obb[1][0]), float(obb[1][1]), float(obb[2])]

                                if poly_pts:
                                    shapes_data.append({
                                        "label": prompt,
                                        "type": "polygon",
                                        "points": poly_pts,
                                        "rect": rect_xywh,
                                        "obb": rect_obb,
                                        "score": score_val
                                    })

                if self.is_cancelled:
                    break

                # 4. 后台直接保存至文件
                if shapes_data:
                    base_name = os.path.splitext(img_path)[0]
                    export_shapes = []
                    
                    for s in shapes_data:
                        label = s["label"]
                        poly_pts = s["points"]
                        rect_xywh = s["rect"]
                        rect_obb = s["obb"]
                        
                        if self.canvas_mode == 1:  # CanvasMode.RECT
                            x, y, w, h = rect_xywh
                            export_shapes.append({
                                "label": label,
                                "type": "rectangle",
                                "points": [[x, y], [x + w, y + h]]
                            })
                        elif self.canvas_mode == 2:  # CanvasMode.POLY
                            export_shapes.append({
                                "label": label,
                                "type": "polygon",
                                "points": poly_pts
                            })
                        elif self.canvas_mode == 4:  # CanvasMode.RBOX
                            if rect_obb and len(rect_obb) == 5:
                                cx, cy, w, h, angle = rect_obb
                                rect_box = cv2.boxPoints(((cx, cy), (w, h), angle))
                                points = rect_box.tolist()
                                export_shapes.append({
                                    "label": label,
                                    "type": "obb",
                                    "points": points,
                                    "rect": [cx, cy, w, h],
                                    "angle": angle
                                })

                    # Load existing shapes if we are in merge mode (not self.overwrite)
                    all_shapes = []
                    if not self.overwrite:
                        if self.current_format == "json":
                            json_path = base_name + ".json"
                            if os.path.exists(json_path):
                                try:
                                    import json
                                    with open(json_path, "r", encoding="utf-8") as f:
                                        data_json = json.load(f)
                                    for s in data_json.get("shapes", []):
                                        all_shapes.append({
                                            "label": s.get("label", ""),
                                            "type": s.get("shape_type", "rectangle"),
                                            "points": s.get("points", []),
                                            "rect": s.get("rect", [0, 0, 0, 0]),
                                            "angle": s.get("angle", 0.0),
                                            "keypoints": s.get("keypoints", []),
                                            "kpt_shape": s.get("kpt_shape", []),
                                            "template_name": s.get("template_name", "")
                                        })
                                except Exception as e:
                                    print(f"加载已有 JSON 标注出错 {json_path}: {e}")
                        elif self.current_format == "yolo":
                            txt_path = base_name + ".txt"
                            if os.path.exists(txt_path):
                                try:
                                    with open(txt_path, "r", encoding="utf-8") as f:
                                        for line in f:
                                            parts = line.strip().split()
                                            if len(parts) >= 5:
                                                cls_id = int(parts[0])
                                                if cls_id < len(self.class_list):
                                                    label = self.class_list[cls_id]
                                                else:
                                                    label = f"class_{cls_id}"
                                                cx, cy, w, h = map(float, parts[1:5])
                                                abs_cx = cx * w_img
                                                abs_cy = cy * h_img
                                                abs_w = w * w_img
                                                abs_h = h * h_img
                                                x1 = abs_cx - abs_w / 2
                                                y1 = abs_cy - abs_h / 2
                                                all_shapes.append({
                                                    "label": label,
                                                    "type": "rectangle",
                                                    "points": [[x1, y1], [x1 + abs_w, y1 + abs_h]]
                                                })
                                except Exception as e:
                                    print(f"加载已有 YOLO 标注出错 {txt_path}: {e}")
                        elif self.current_format == "xml":
                            xml_path = base_name + ".xml"
                            if os.path.exists(xml_path):
                                try:
                                    import xml.etree.ElementTree as ET
                                    tree = ET.parse(xml_path)
                                    root = tree.getroot()
                                    for obj in root.findall("object"):
                                        label = obj.find("name").text
                                        bndbox = obj.find("bndbox")
                                        x1 = float(bndbox.find("xmin").text)
                                        y1 = float(bndbox.find("ymin").text)
                                        x2 = float(bndbox.find("xmax").text)
                                        y2 = float(bndbox.find("ymax").text)
                                        all_shapes.append({
                                            "label": label,
                                            "type": "rectangle",
                                            "points": [[x1, y1], [x2, y2]]
                                        })
                                except Exception as e:
                                    print(f"加载已有 XML 标注出错 {xml_path}: {e}")

                    # Add new prediction shapes to all_shapes
                    all_shapes.extend(export_shapes)

                    if all_shapes:
                        if self.current_format == "json":
                            out_path = base_name + ".json"
                            Exporter.save_json(out_path, img_path, w_img, h_img, all_shapes)
                        elif self.current_format == "yolo":
                            out_path = base_name + ".txt"
                            Exporter.save_yolo(out_path, w_img, h_img, all_shapes, self.class_list)
                        elif self.current_format == "xml":
                            out_path = base_name + ".xml"
                            Exporter.save_xml(out_path, img_path, w_img, h_img, all_shapes)
                    elif self.overwrite:
                        # 覆盖模式下且无新标注时清空文件
                        for ext in [".json", ".txt", ".xml"]:
                            out_path = base_name + ext
                            if os.path.exists(out_path):
                                try:
                                    os.remove(out_path)
                                except Exception as e:
                                    print(f"删除空标注文件失败 {out_path}: {e}")

                processed += 1

            self.finished.emit(processed, total)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
class YoloBatchPredictorWorker(QThread):
    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(int, int, list)  # processed, total, detected_classes
    error = Signal(str)

    def __init__(self, predictor, img_paths, current_format, class_list, template_manager=None, classes=None, overwrite=False):
        super().__init__()
        self.predictor = predictor
        self.img_paths = img_paths
        self.current_format = current_format
        self.class_list = list(class_list)
        self.template_manager = template_manager
        self.is_cancelled = False
        self.classes = classes
        self.overwrite = overwrite

    def run(self):
        import cv2
        import numpy as np
        from PIL import Image
        from labelpaw.data.exporter import Exporter
        from PySide6.QtCore import QRectF, QPointF
        from PySide6.QtGui import QPolygonF

        total = len(self.img_paths)
        processed = 0
        detected_classes = set()

        try:
            for idx, img_path in enumerate(self.img_paths):
                if self.is_cancelled:
                    break

                filename = os.path.basename(img_path)
                self.progress.emit(idx, total, filename)

                # 1. 获取图像尺寸
                try:
                    with Image.open(img_path) as pil_img:
                        w_img, h_img = pil_img.size
                except Exception as e:
                    print(f"无法读取图片尺寸 {img_path}: {e}")
                    processed += 1
                    continue

                base_name = os.path.splitext(img_path)[0]
                existing_shapes = []

                # 1.1 读取并保留已有的标注（以实现增量合并预测）
                if not self.overwrite:
                    if self.current_format == "json":
                        json_path = base_name + ".json"
                        if os.path.exists(json_path):
                            try:
                                import json
                                with open(json_path, "r", encoding="utf-8") as f:
                                    data_json = json.load(f)
                                for s in data_json.get("shapes", []):
                                    existing_shapes.append({
                                        "label": s.get("label", ""),
                                        "type": s.get("shape_type", "rectangle"),
                                        "points": s.get("points", []),
                                        "rect": s.get("rect", [0, 0, 0, 0]),
                                        "angle": s.get("angle", 0.0),
                                        "keypoints": s.get("keypoints", []),
                                        "kpt_shape": s.get("kpt_shape", []),
                                        "template_name": s.get("template_name", "")
                                    })
                            except Exception as e:
                                print(f"加载已有 JSON 标注出错 {json_path}: {e}")
                    elif self.current_format == "yolo":
                        txt_path = base_name + ".txt"
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, "r", encoding="utf-8") as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            cls_id = int(parts[0])
                                            if cls_id < len(self.class_list):
                                                label = self.class_list[cls_id]
                                            else:
                                                label = f"class_{cls_id}"
                                            cx, cy, w, h = map(float, parts[1:5])
                                            abs_cx = cx * w_img
                                            abs_cy = cy * h_img
                                            abs_w = w * w_img
                                            abs_h = h * h_img
                                            x1 = abs_cx - abs_w / 2
                                            y1 = abs_cy - abs_h / 2
                                            existing_shapes.append({
                                                "label": label,
                                                "type": "rectangle",
                                                "points": [[x1, y1], [x1 + abs_w, y1 + abs_h]]
                                            })
                            except Exception as e:
                                print(f"加载已有 YOLO 标注出错 {txt_path}: {e}")
                    elif self.current_format == "xml":
                        xml_path = base_name + ".xml"
                        if os.path.exists(xml_path):
                            try:
                                import xml.etree.ElementTree as ET
                                tree = ET.parse(xml_path)
                                root = tree.getroot()
                                for obj in root.findall("object"):
                                    label = obj.find("name").text
                                    bndbox = obj.find("bndbox")
                                    x1 = float(bndbox.find("xmin").text)
                                    y1 = float(bndbox.find("ymin").text)
                                    x2 = float(bndbox.find("xmax").text)
                                    y2 = float(bndbox.find("ymax").text)
                                    existing_shapes.append({
                                        "label": label,
                                        "type": "rectangle",
                                        "points": [[x1, y1], [x2, y2]]
                                    })
                            except Exception as e:
                                print(f"加载已有 XML 标注出错 {xml_path}: {e}")

                # 2. 运行 YOLO 同步预测
                shapes = self.predictor.predict_sync(img_path, classes=self.classes)
                
                # 3. 解析预测结果并与已有标注进行 IoU 重合度去重过滤
                def get_ext_rect(s):
                    stype = s.get("type", "rectangle")
                    if stype in ["rectangle", "polygon", "obb"] and s.get("points"):
                        pts = s["points"]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                    elif stype == "pose" and s.get("rect"):
                        cx, cy, w, h = s["rect"]
                        return QRectF(cx - w/2, cy - h/2, w, h)
                    return None

                def get_iou(r1, r2):
                    if not r1 or not r2:
                        return 0.0
                    intersection = r1.intersected(r2)
                    inter_area = max(0.0, intersection.width()) * max(0.0, intersection.height())
                    area1 = max(0.0, r1.width()) * max(0.0, r1.height())
                    area2 = max(0.0, r2.width()) * max(0.0, r2.height())
                    union_area = area1 + area2 - inter_area
                    if union_area > 0.0:
                        return inter_area / union_area
                    return 0.0

                export_shapes = list(existing_shapes)
                
                for s in shapes:
                    shape_type = s["type"]
                    label = s["label"]
                    data = s["data"]
                    detected_classes.add(label)

                    # 计算预测框的外接矩形
                    new_rect = None
                    if shape_type == "rect":
                        new_rect = data
                    elif shape_type in ["poly", "rbox"]:
                        new_rect = data.boundingRect()
                    elif shape_type == "pose":
                        new_rect = data["rect"]

                    # 仅在【相同类别】且【IoU > 0.8】时去重；不同类别即使重合度再高也要保留
                    is_duplicate = False
                    if new_rect:
                        for ext in existing_shapes:
                            if ext.get("label", "") == label:
                                ext_rect = get_ext_rect(ext)
                                if ext_rect and get_iou(new_rect, ext_rect) > 0.8:
                                    is_duplicate = True
                                    break

                    if is_duplicate:
                        continue

                    # 转换新预测的框为对应格式并加入队列
                    if shape_type == "rect":
                        x1, y1 = data.x(), data.y()
                        w, h = data.width(), data.height()
                        export_shapes.append({
                            "label": label,
                            "type": "rectangle",
                            "points": [[x1, y1], [x1 + w, y1 + h]]
                        })
                    elif shape_type == "poly":
                        points = [[data[i].x(), data[i].y()] for i in range(data.count())]
                        export_shapes.append({
                            "label": label,
                            "type": "polygon",
                            "points": points
                        })
                    elif shape_type == "rbox":
                        points = [[data[i].x(), data[i].y()] for i in range(data.count())]
                        pts = np.array(points, dtype=np.float32)
                        rect_obb = cv2.minAreaRect(pts)
                        cx, cy = float(rect_obb[0][0]), float(rect_obb[0][1])
                        w, h = float(rect_obb[1][0]), float(rect_obb[1][1])
                        angle = float(rect_obb[2])

                        export_shapes.append({
                            "label": label,
                            "type": "obb",
                            "points": points,
                            "rect": [cx, cy, w, h],
                            "angle": angle
                        })
                    elif shape_type == "pose":
                        rect = data["rect"]
                        cx, cy = rect.x() + rect.width()/2, rect.y() + rect.height()/2
                        w, h = rect.width(), rect.height()

                        kps = data["keypoints"]
                        keypoints = []
                        for kp in kps:
                            pos = kp["pos"]
                            keypoints.append([pos.x(), pos.y(), kp["vis"]])

                        # 构造和保存模板
                        template_name = None
                        if "skeleton" in s and self.template_manager:
                            template_name = f"YOLO_Auto_{len(keypoints)}"
                            yolo_skeleton = s["skeleton"]
                            kpt_names = s.get("kpt_names", [])
                            
                            template = {
                                "name": template_name,
                                "label": label,
                                "keypoints": [],
                                "connections": []
                            }
                            
                            for i in range(len(keypoints)):
                                name = kpt_names[i] if i < len(kpt_names) else f"kp_{i}"
                                template["keypoints"].append({"name": name, "color": "#00FF00", "default_pos": [0.5, 0.5]})
                                
                            for edge in yolo_skeleton:
                                if len(edge) == 2:
                                    p1, p2 = edge[0], edge[1]
                                    if p1 > 0 and p2 > 0:
                                        template["connections"].append([p1 - 1, p2 - 1])
                                    else:
                                        template["connections"].append([p1, p2])
                                        
                            self.template_manager.add_template(template)

                        if not template_name and self.template_manager:
                            for t in self.template_manager.templates:
                                if len(t.get("keypoints", [])) == len(keypoints):
                                    template_name = t["name"]
                                    break

                        if not template_name:
                            template_name = f"YOLO_Pose_{len(keypoints)}"

                        export_shape_dict = {
                            "label": label,
                            "type": "pose",
                            "points": [],
                            "rect": [cx, cy, w, h],
                            "angle": 0.0,
                            "keypoints": keypoints,
                            "kpt_shape": [len(keypoints), 3],
                            "template_name": template_name
                        }
                        export_shapes.append(export_shape_dict)

                # 4. 后台增量保存到对应的标注文件中
                if export_shapes:
                    if self.current_format == "json":
                        out_path = base_name + ".json"
                        Exporter.save_json(out_path, img_path, w_img, h_img, export_shapes)
                    elif self.current_format == "yolo":
                        out_path = base_name + ".txt"
                        Exporter.save_yolo(out_path, w_img, h_img, export_shapes, self.class_list)
                    elif self.current_format == "xml":
                        out_path = base_name + ".xml"
                        Exporter.save_xml(out_path, img_path, w_img, h_img, export_shapes)
                elif self.overwrite:
                    for ext in [".json", ".txt", ".xml"]:
                        out_path = base_name + ext
                        if os.path.exists(out_path):
                            try:
                                os.remove(out_path)
                            except Exception as e:
                                print(f"无法删除空标注文件 {out_path}: {e}")

                processed += 1

            self.finished.emit(processed, total, list(detected_classes))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class BatchProgressDialog(QDialog):
    def __init__(self, parent=None, is_dark_theme=False, title="SAM 3 批量标注中"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(420, 180)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.is_cancelled = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        
        # Header layout (Loading.gif + text)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        self.loading_label = QLabel()
        self.loading_label.setFixedSize(32, 32)
        self.movie = QMovie("ui/icon/Loading.gif")
        self.movie.setScaledSize(QSize(32, 32))
        self.loading_label.setMovie(self.movie)
        self.movie.start()
        
        self.status_label = QLabel("正在初始化批量标注任务...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        header_layout.addWidget(self.loading_label)
        header_layout.addWidget(self.status_label, 1)
        layout.addLayout(header_layout)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #475569;
                border-radius: 6px;
                text-align: center;
                height: 20px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #22C55E;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Cancel Button Layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("取消任务")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #F87171; }
        """)
        self.btn_cancel.clicked.connect(self.cancel_task)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        if is_dark_theme:
            self.setStyleSheet("""
                QDialog { background-color: #1E293B; color: #F8FAFC; }
                QLabel { color: #F8FAFC; }
                QProgressBar { border-color: #334155; background-color: #0F172A; color: #F8FAFC; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background-color: #F8FAFC; color: #0F172A; }
                QLabel { color: #0F172A; }
                QProgressBar { border-color: #E2E8F0; background-color: #F1F5F9; color: #0F172A; }
            """)

    def cancel_task(self):
        self.is_cancelled = True
        self.status_label.setText("正在取消任务，请稍候...")
        self.btn_cancel.setEnabled(False)


class YoloClassFilterDialog(QDialog):
    def __init__(self, class_names, selected_ids, is_dark_theme=False, parent=None):
        super().__init__(parent)
        self.class_names = class_names  # dict, e.g., {0: 'person', 1: 'bicycle'}
        self.initial_selected_ids = set(selected_ids)
        self.is_dark_theme = is_dark_theme
        self.selected_ids = []
        
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(360, 520)
        
        # Colors
        bg_color = "#0F172A" if is_dark_theme else "#FFFFFF"
        border_color = "#334155" if is_dark_theme else "#E2E8F0"
        text_color = "#F8FAFC" if is_dark_theme else "#0F172A"
        list_bg = "#0F172A" if is_dark_theme else "#FFFFFF"
        item_hover_bg = "#1E293B" if is_dark_theme else "#F1F5F9"
        primary_color = "#22C55E"
        btn_hover_bg = "#16A34A"
        
        # Main Container QFrame
        self.main_container = QFrame(self)
        self.main_container.setObjectName("MainContainer")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.main_container)
        
        # Layout inside Main Container
        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(10, 10, 10, 15)
        container_layout.setSpacing(10)
        
        # Custom Title Bar
        self.title_bar = QWidget()
        self.title_bar.setObjectName("TitleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 5, 5, 5)
        
        self.title_label = QLabel("YOLO 类别过滤")
        self.title_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {text_color};")
        title_layout.addWidget(self.title_label)
        
        title_layout.addStretch()
        
        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.reject)
        
        close_icon = QIcon("ui/icon/x.svg")
        if is_dark_theme:
            close_icon = self.set_icon_color(close_icon, QColor("#94A3B8"))
        else:
            close_icon = self.set_icon_color(close_icon, QColor("#64748B"))
        self.btn_close.setIcon(close_icon)
        self.btn_close.setIconSize(QSize(12, 12))
        
        self.btn_close.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background-color: transparent;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: rgba(239, 68, 68, 0.2);
            }}
        """)
        title_layout.addWidget(self.btn_close)
        container_layout.addWidget(self.title_bar)
        
        # Search Line Edit
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索类别名称...")
        
        search_icon = QIcon("ui/icon/magnifying-glass.svg")
        if is_dark_theme:
            search_icon = self.set_icon_color(search_icon, QColor("#94A3B8"))
        else:
            search_icon = self.set_icon_color(search_icon, QColor("#64748B"))
        self.search_edit.addAction(search_icon, QLineEdit.LeadingPosition)
        
        self.search_edit.textChanged.connect(self.on_search_changed)
        container_layout.addWidget(self.search_edit)
        
        # Select All Checkbox
        self.chk_all = QCheckBox("全选 / 反选")
        self.chk_all.setCursor(Qt.PointingHandCursor)
        self.chk_all.setTristate(True)
        self.chk_all.clicked.connect(self.on_chk_all_clicked)
        container_layout.addWidget(self.chk_all)
        
        # List Widget
        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self.on_item_changed)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        container_layout.addWidget(self.list_widget)
        
        # Populate List
        self._updating_list = True
        sorted_cids = sorted(self.class_names.keys())
        for cid in sorted_cids:
            cname = self.class_names[cid]
            item = QListWidgetItem(f"{cid}: {cname}")
            item.setData(Qt.UserRole, cid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            
            is_checked = (cid in self.initial_selected_ids)
            item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
            self.list_widget.addItem(item)
            
        self._updating_list = False
        self.update_chk_all_state()
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("重置全选")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_to_all)
        
        self.btn_confirm = QPushButton("确认")
        self.btn_confirm.setCursor(Qt.PointingHandCursor)
        self.btn_confirm.clicked.connect(self.accept_selection)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_confirm)
        container_layout.addLayout(btn_layout)
        
        # Set Stylesheet for inner elements
        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                color: {text_color};
            }}
            #MainContainer {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QLineEdit {{
                background-color: {list_bg};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 8px 12px;
                color: {text_color};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {primary_color};
            }}
            QCheckBox {{
                color: {text_color};
                font-weight: bold;
                font-size: 13px;
                spacing: 8px;
            }}
            QListWidget {{
                background-color: {list_bg};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-radius: 6px;
                color: {text_color};
            }}
            QListWidget::item:hover {{
                background-color: {item_hover_bg};
            }}
            QPushButton {{
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
                background-color: {list_bg};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {item_hover_bg};
            }}
        """)
        
        self.btn_confirm.setStyleSheet(f"""
            QPushButton {{
                background-color: {primary_color};
                color: white;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {btn_hover_bg};
            }}
        """)
        
        # Install global event filter to close on clicking outside
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if click is on the title bar or label
            child = self.childAt(event.position().toPoint())
            if child in [self.title_bar, self.title_label] or (child and child.parent() == self.title_bar and child != self.btn_close):
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, '_drag_pos'):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def event(self, event):
        if event.type() == QEvent.WindowDeactivate:
            self.reject()
            return True
        return super().event(event)
            
    def on_search_changed(self, text):
        text = text.lower().strip()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text not in item.text().lower())
        self.update_chk_all_state()
            
    def on_chk_all_clicked(self):
        # Determine target state: if any visible item is currently unchecked, we want to check all.
        # Otherwise (all checked), we want to uncheck all.
        any_unchecked = False
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden() and item.checkState() == Qt.Unchecked:
                any_unchecked = True
                break
                
        self._updating_list = True
        target_state = Qt.Checked if any_unchecked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(target_state)
        self._updating_list = False
        self.update_chk_all_state()
        
    def on_item_changed(self, item):
        if getattr(self, '_updating_list', False):
            return
        self.update_chk_all_state()
        
    def on_item_clicked(self, item):
        from PySide6.QtGui import QCursor
        pos = self.list_widget.viewport().mapFromGlobal(QCursor.pos())
        if pos.x() > 35:
            self._updating_list = True
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
            self._updating_list = False
            self.update_chk_all_state()
            
    def on_item_double_clicked(self, item):
        self._updating_list = True
        new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        item.setCheckState(new_state)
        self._updating_list = False
        self.update_chk_all_state()
        
    def update_chk_all_state(self):
        checked_count = 0
        total_count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                total_count += 1
                if item.checkState() == Qt.Checked:
                    checked_count += 1
                    
        if total_count == 0:
            self.chk_all.setCheckState(Qt.Unchecked)
        elif checked_count == total_count:
            self.chk_all.setCheckState(Qt.Checked)
        elif checked_count == 0:
            self.chk_all.setCheckState(Qt.Unchecked)
        else:
            self.chk_all.setCheckState(Qt.PartiallyChecked)
        
    def reset_to_all(self):
        self._updating_list = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)
        self._updating_list = False
        self.update_chk_all_state()
        
    def accept_selection(self):
        checked_ids = []
        total_count = self.list_widget.count()
        
        for i in range(total_count):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                cid = item.data(Qt.UserRole)
                checked_ids.append(cid)
                
        if len(checked_ids) == total_count or len(checked_ids) == 0:
            self.selected_ids = []
        else:
            self.selected_ids = checked_ids
            
        self.accept()

    def set_icon_color(self, icon, color):
        from PySide6.QtGui import QPainter, QIcon
        pixmap = icon.pixmap(100, 100)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        new_icon = QIcon()
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        return new_icon

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ActivationChange:
            if not self.isActiveWindow():
                self.reject()
        super().changeEvent(event)

    def cleanup_filter(self):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass

    def reject(self):
        self.cleanup_filter()
        super().reject()

    def accept(self):
        self.cleanup_filter()
        super().accept()

    def closeEvent(self, event):
        self.cleanup_filter()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            from PySide6.QtGui import QCursor
            if not self.geometry().contains(QCursor.pos()):
                self.reject()
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.template_manager = TemplateManager()

        self.is_dark_theme = False
        self.classListWidget.set_theme(self.is_dark_theme)
        self.setStyleSheet(LIGHT_THEME)

        self.scene = Canvas(self)
        self.view.setScene(self.scene)

        self.current_image_path = None
        self.current_dir = None
        self.class_list = []
        self.current_format = "json"
        self.yolo_filtered_class_ids = []
        self.auto_advance_after_smart = True

        self.modeLabel = QLabel("模式: 矩形标注")
        self.statusBar.addWidget(self.modeLabel)

        self.helpLabel = QLabel("状态: 正在初始化")
        self.statusBar.addWidget(self.helpLabel)

        self.sam_client = SAMClient(self)
        self.sam_client.inference_result.connect(self.scene.handle_sam_result)
        self.sam_client.text_result_ready.connect(self.handle_text_results)
        self.sam_client.model_status_changed.connect(self.update_model_status)
        self.scene.sam_client = self.sam_client

        # 撤销/重做时数据栈
        self.undo_stack = []
        self.redo_stack = []
        self.max_history_steps = 20  # 保留20步历史记录
        self.scene.state_changed.connect(self.push_state)  # 绑定画板信号
        
        self._init_pose_templates()

        self._connect_signals()
        self._set_mode(CanvasMode.RECT)
        
        # 初始化模型下拉菜单
        self._init_model_selector()
        
        # 默认加载 SAM 3
        self.on_model_selected("sam3")
        
        self.yolo_worker = None

        # 初始化 btnDeleteFiles 的图标颜色
        from PySide6.QtGui import QColor
        self.current_icon_color = QColor(15, 23, 42)
        self.btnDeleteFiles.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
        
        self.btnPredict.setIcon(QIcon("ui/icon/lightning-fill.svg"))
        
        # 预测按钮动画
        self.predict_movie = QMovie("ui/icon/Loading.gif")
        self.predict_movie.frameChanged.connect(self.update_predict_icon)
        
        # 初始化选择框状态和可用性
        self.update_selected_count()

        # 初始化 samPromptBtn 样式与交互 (Gemini 3 Style)
        self.update_prompt_btn_icon()
        self.samPromptBtn.setEnabled(False)
        self.samPromptBtn.show()
        
        # 绑定文字变化信号以实现动态状态切换
        self.samPromptInput.textChanged.connect(self.on_prompt_text_changed)
        
        # 安装事件过滤器以实现 focus 边框变化
        self.samPromptInput.installEventFilter(self)

        # 恢复上次关闭时的标注路径和图片
        self.restore_last_state()

    def update_prompt_btn_icon(self):
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon, QColor, QPainter
        active_color = QColor(255, 255, 255)
        if self.is_dark_theme:
            disabled_color = QColor(255, 255, 255, 60)
        else:
            disabled_color = QColor(100, 116, 139, 120)  # Slate gray for perfect visibility in light theme!
            
        icon = QIcon("ui/icon/arrow-up.svg")
        
        # Active Pixmap
        active_pixmap = icon.pixmap(100, 100)
        painter = QPainter(active_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(active_pixmap.rect(), active_color)
        painter.end()
        
        # Disabled Pixmap
        disabled_pixmap = icon.pixmap(100, 100)
        dpainter = QPainter(disabled_pixmap)
        dpainter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        dpainter.fillRect(disabled_pixmap.rect(), disabled_color)
        dpainter.end()
        
        new_icon = QIcon()
        new_icon.addPixmap(active_pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(active_pixmap, QIcon.Normal, QIcon.Off)
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.On)
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.Off)
        
        self.samPromptBtn.setIcon(new_icon)
        self.samPromptBtn.setIconSize(QSize(16, 16))

    def update_prompt_btn_state(self):
        supports_text = self.samPromptInput.isEnabled()
        has_text = bool(self.samPromptInput.text().strip())
        self.samPromptBtn.setEnabled(supports_text and has_text)

    def on_prompt_text_changed(self, text):
        self.update_prompt_btn_state()

    def eventFilter(self, watched, event):
        if watched == getattr(self, 'samPromptInput', None):
            if event.type() == QEvent.FocusIn:
                self.samTextGroup.setProperty("focused", "true")
                self.samTextGroup.style().unpolish(self.samTextGroup)
                self.samTextGroup.style().polish(self.samTextGroup)
            elif event.type() == QEvent.FocusOut:
                self.samTextGroup.setProperty("focused", "false")
                self.samTextGroup.style().unpolish(self.samTextGroup)
                self.samTextGroup.style().polish(self.samTextGroup)
        return super().eventFilter(watched, event)

    def update_predict_icon(self):
        self.btnPredict.setIcon(QIcon(self.predict_movie.currentPixmap()))

    def _init_pose_templates(self):
        templates = self.template_manager.get_template_names()
        self.templateWidget.update_templates(templates, main_window=self)
        if not self.scene.current_pose_template:
            self.scene.current_pose_template = self.template_manager.get_template("Person (COCO)")

    def _init_model_selector(self):
        self.btnModelSelector.clicked.connect(self._show_model_selector)

    def _show_model_selector(self):
        dialog = ModelSelectorDialog(current_model_key=getattr(self.sam_client, 'current_model_key', None), is_dark_theme=self.is_dark_theme, parent=self)
        dialog.model_selected.connect(self.on_model_selected)
        
        # Position popup near button
        pos = self.btnModelSelector.mapToGlobal(self.btnModelSelector.rect().bottomLeft())
        dialog.move(pos.x(), pos.y() + 5)
        dialog.exec()

    def on_model_selected(self, model_info_or_key):
        if isinstance(model_info_or_key, str):
            key = model_info_or_key
            if key in SAM_MODEL_MAP:
                model_info = SAM_MODEL_MAP[key]
            else:
                model_info = {"key": key, "display_name": key, "type": "sam2", "supports_text": False}
        else:
            model_info = model_info_or_key
            key = model_info["key"]
            
        display_name = model_info.get("display_name", key)
        
        self.btnModelSelector.setText(f" {display_name} ▾")
        self.btnModelSelector.setIcon(QIcon("ui/icon/s.svg"))
        
        if key in SAM_MODEL_MAP or model_info.get("type", "").startswith("sam"):
            self.btnPredict.hide()
            self.btnClassFilter.hide()
            self.yolo_filtered_class_ids = []
            self.current_yolo_predictor = None
            if key not in SAM_MODEL_MAP:
                SAM_MODEL_MAP[key] = model_info
            self.sam_client.load_model_by_key(key)
            # 如果当前在点标注模式（不支持SAM），自动切换回矩形标注
            if self.scene.mode == CanvasMode.POINT:
                self.actionRect.trigger()
        elif model_info.get("type", "").startswith("yolo"):
            self.sam_client.cleanup()
            self.sam_client.current_model_key = key
            from labelpaw.models.yolo_predictor import YoloPredictor
            path = model_info.get("path", "")
            if path and os.path.exists(path):
                try:
                    self.current_yolo_predictor = YoloPredictor(path)
                    
                    # 自动根据 YOLO 任务类型切换绘制模式
                    task = getattr(self.current_yolo_predictor, 'task', 'detect')
                    if task == 'detect':
                        self.actionRect.trigger()
                    elif task == 'segment':
                        self.actionPoly.trigger()
                    elif task == 'pose':
                        self.actionPoint.trigger()
                    elif task == 'obb':
                        self.actionRBox.trigger()
                        
                    self.update_model_status(True, f"已加载 YOLO 模型: {display_name}")
                    self.btnPredict.show()
                    self.btnClassFilter.show()
                    self.btnClassFilter.setText(" 全部类别 ▾")
                    self.yolo_filtered_class_ids = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.update_model_status(False, f"加载 YOLO 模型失败: {str(e)}")
                    self.btnPredict.hide()
                    self.btnClassFilter.hide()
                    self.yolo_filtered_class_ids = []
            else:
                self.update_model_status(False, f"未找到对应的 YOLO 模型文件")
                self.btnPredict.hide()
                self.btnClassFilter.hide()
                self.yolo_filtered_class_ids = []
        else:
            self.btnPredict.hide()
            self.btnClassFilter.hide()
            self.yolo_filtered_class_ids = []
            self.current_yolo_predictor = None
            self.sam_client.cleanup()
            self.sam_client.current_model_key = key
            self.update_model_status(True, f"已选择模型: {display_name}。非 SAM 模型暂不支持智能推理。")
        
        supports_text = model_info.get("supports_text", False)
        
        # 只有不在点标注模式下才启用，因为点标注即使是SAM3也不可用
        if self.scene.mode != CanvasMode.POINT:
            self.samPromptInput.setEnabled(supports_text)
            self.update_prompt_btn_state()
            if supports_text:
                self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog、cat)")
            else:
                self.samPromptInput.setPlaceholderText(f"{display_name} 不支持提示词")

    def edit_pose_template(self, name):
        dlg = SkeletonTemplateDialog(self, self.template_manager, self.is_dark_theme)
        dlg.load_template(name)
        dlg.name_edit.setText(name)
        if dlg.exec() == QDialog.Accepted:
            self._init_pose_templates()
            self.templateWidget._on_template_selected(name, f"{name} ▾")

    def delete_pose_template(self, name):
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "删除骨架模板", f"确定要删除骨架模板 '{name}' 吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.template_manager.delete_template(name)
            self._init_pose_templates()
            self.templateWidget._on_template_selected("Person (COCO)", "Person (COCO) ▾")

    def _connect_signals(self):
        self.templateWidget.edit_template.connect(self.edit_pose_template)
        self.templateWidget.delete_template.connect(self.delete_pose_template)
        self.btnAuthorInfo.clicked.connect(self.show_author_info)
        self.btnCollapse.clicked.connect(self.toggle_sidebar)
        self.btnThemeToggle.clicked.connect(self.toggle_theme)

        self.btnUndo.clicked.connect(self.undo)
        self.btnRedo.clicked.connect(self.redo)
        self.btnDelete.clicked.connect(self.delete_selected)
        self.btnSave.clicked.connect(lambda: self.save_annotation(self.current_format))
        self.btnKeyboard.clicked.connect(self.show_help_dialog)

        self.actionOpen.triggered.connect(self.open_dir)

        # 下拉菜单组件信号
        self.formatWidget.format_changed.connect(self.set_current_format)

        self.btnDatasetTool.clicked.connect(self.open_dataset_tool)

        # self.actionFormatJSON.triggered.connect(lambda: self.set_current_format("json"))
        # self.actionFormatYOLO.triggered.connect(lambda: self.set_current_format("yolo"))
        # self.actionFormatXML.triggered.connect(lambda: self.set_current_format("xml"))

        self.actionRect.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RECT))
        self.actionPoly.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POLY))
        self.actionPoint.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POINT))
        self.actionRBox.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RBOX))

        self.templateWidget.template_changed.connect(self.on_pose_template_changed)

        self.samSwitch.toggled.connect(self.on_sam_toggled)

        # 同步顶部 Draw / Smart 按钮与 SAM 开关状态
        self.btnDrawMode.toggled.connect(lambda checked: self.samSwitch.setChecked(False) if checked else None)
        self.btnSmartMode.toggled.connect(lambda checked: self.samSwitch.setChecked(True) if checked else None)

        self.btnPredict.clicked.connect(self.on_predict_clicked)
        self.btnClassFilter.clicked.connect(self.on_class_filter_clicked)

        self.samPromptBtn.clicked.connect(self.trigger_sam_prompt)
        self.samPromptInput.returnPressed.connect(self.trigger_sam_prompt)

        self.chkSelectAll.stateChanged.connect(self.on_select_all_toggled)
        self.btnDeleteFiles.clicked.connect(self.delete_checked_files)
        self.listFiles.itemChanged.connect(self.on_file_item_changed)
        self.listFiles.currentItemChanged.connect(self.on_file_selected)
        self.listFiles.customContextMenuRequested.connect(self.show_file_list_context_menu)
        self.scene.mouse_moved.connect(self.update_coordinate_label)
        self.scene.shape_drawn.connect(self.handle_new_shape)

        # Annotation tree dual-sync signals
        self.scene.state_changed.connect(self.update_annotation_tree)
        self.scene.selectionChanged.connect(self.sync_selection_to_tree)
        self.scene.canvas_item_hovered.connect(self.classListWidget.highlight_item_by_shape)

        self.scene.shape_double_clicked.connect(self.edit_shape_label)  # 双击修改

        # 类别管理组件信号
        self.classListWidget.class_added.connect(self._on_class_added_from_widget)
        self.classListWidget.class_renamed.connect(self._on_class_renamed_from_widget)
        self.classListWidget.color_changed.connect(self.on_class_color_changed)
        self.classListWidget.item_changed.connect(self.on_list_item_changed)
        self.classListWidget.shape_class_reassigned.connect(self.on_shape_class_reassigned)

    def on_predict_clicked(self):
        if not getattr(self, 'current_yolo_predictor', None):
            DialogOver(self, "YOLO 模型未加载或初始化失败！", "提示", "warning")
            return

        if self.yolo_worker and self.yolo_worker.isRunning():
            return

        # 获取当前勾选的所有图片路径
        checked_paths = []
        for i in range(self.listFiles.count()):
            item = self.listFiles.item(i)
            if item and item.checkState() == Qt.Checked:
                checked_paths.append(item.text())

        if checked_paths:
            # YOLO 批量预测逻辑
            self.statusBar.showMessage(f"正在进行 YOLO 批量预测 ({len(checked_paths)}张图片)...", 3000)
            self.batch_dialog = BatchProgressDialog(self, self.is_dark_theme, "YOLO 批量预测中")
            
            self.yolo_worker = YoloBatchPredictorWorker(
                predictor=self.current_yolo_predictor,
                img_paths=checked_paths,
                current_format=self.current_format,
                class_list=self.class_list,
                template_manager=self.template_manager,
                classes=self.yolo_filtered_class_ids,
                overwrite=self.btnOverwrite.isChecked()
            )

            # 连接进度更新
            def update_progress(current, total, filename):
                percent = int((current / total) * 100) if total > 0 else 0
                self.batch_dialog.progress_bar.setValue(percent)
                self.batch_dialog.status_label.setText(f"正在预测 ({current + 1}/{total}):\n{filename}")
                
            self.yolo_worker.progress.connect(update_progress)

            # 完成回调
            def on_batch_finished(processed, total, detected_classes):
                self.batch_dialog.movie.stop()
                self.batch_dialog.accept()
                
                # 同步新增的类别到历史面板
                added_classes = []
                for cls in detected_classes:
                    if cls not in self.class_list:
                        self.add_class_to_list(cls)
                        added_classes.append(cls)
                if added_classes:
                    self.save_classes()

                # 如果当前打开的图片刚好在批量预测的列表中，重新加载以即时渲染预测标注
                if self.current_image_path in checked_paths:
                    self.scene.clear_shapes()
                    self.load_annotations(self.current_image_path)
                    self.apply_class_colors_to_scene()
                    self.update_annotation_tree()
                    self.push_state()

                DialogOver(self, f"YOLO 批量预测完成！已成功处理 {processed}/{total} 张图片并自动保存。", "批量预测成功", "success")
                self.statusBar.showMessage(f"批量预测完成！处理了 {processed} 张图片", 5000)

            # 错误回调
            def on_batch_error(err_msg):
                self.batch_dialog.movie.stop()
                self.batch_dialog.reject()
                DialogOver(self, f"批量预测出错: {err_msg}", "预测失败", "danger")
                self.statusBar.showMessage("批量预测出错", 3000)

            self.yolo_worker.finished.connect(on_batch_finished)
            self.yolo_worker.error.connect(on_batch_error)
            
            # 取消按钮连接
            self.batch_dialog.btn_cancel.clicked.connect(lambda: setattr(self.yolo_worker, 'is_cancelled', True))

            self.yolo_worker.start()
            self.batch_dialog.exec()
        else:
            # 默认单张图片预测逻辑
            if not self.current_image_path:
                DialogOver(self, "请先打开一张图片！", "提示", "warning")
                return
                
            self.statusBar.showMessage("正在使用 YOLO 进行预测...", 3000)
            self.helpLabel.setText("正在使用 YOLO 进行预测...")
            self.helpLabel.setStyleSheet("color: orange;")
            self.original_predict_text = self.btnPredict.text()
            self.btnPredict.setText("预测中")
            
            self.predict_movie.start()
            self.btnPredict.setEnabled(False)
            
            self.yolo_worker = YoloPredictorWorker(self.current_yolo_predictor, self.current_image_path, classes=self.yolo_filtered_class_ids)
            self.yolo_worker.finished.connect(self.on_predict_finished)
            self.yolo_worker.error.connect(self.on_predict_error)
            self.yolo_worker.start()

    def on_predict_finished(self, shapes):
        self.predict_movie.stop()
        self.btnPredict.setText(self.original_predict_text)
        self.btnPredict.setIcon(QIcon("ui/icon/lightning-fill.svg"))
        self.btnPredict.setEnabled(True)
        
        if self.btnOverwrite.isChecked():
            self.scene.clear_shapes()
            self.update_annotation_tree()
            
        if not shapes:
            if self.btnOverwrite.isChecked():
                self.auto_save_annotation()
                self.push_state()
            DialogOver(self, "未找到任何预测结果", "提示", "info")
            self.statusBar.showMessage("预测完成，未找到任何结果", 3000)
            self._update_help_text(self.scene.mode)
            self._auto_save_and_advance_after_smart()
            return
            
        # Get existing shapes for deduplication
        existing_shapes = []
        if not self.btnOverwrite.isChecked():
            for item in self.scene.items():
                from labelpaw.graphics.shapes import BaseShape
                if isinstance(item, BaseShape) and not getattr(item, 'is_temp', False):
                    existing_shapes.append(item)
            
        # Add shapes
        class_counts = {}
        added_count = 0
        
        for s in shapes:
            shape_type = s["type"]
            label = s["label"]
            data = s["data"]
            
            # Deduplication logic (IoU > 0.8)
            new_rect = None
            if shape_type == "rect":
                new_rect = data
            elif shape_type in ["poly", "rbox"]:
                new_rect = data.boundingRect()
            elif shape_type == "pose":
                new_rect = data["rect"]
                
            is_duplicate = False
            if new_rect:
                for ext_shape in existing_shapes:
                    if getattr(ext_shape, 'label', '') != label:
                        continue
                    ext_rect = ext_shape.sceneBoundingRect()
                    
                    intersection = new_rect.intersected(ext_rect)
                    inter_area = max(0, intersection.width()) * max(0, intersection.height())
                    area1 = max(0, new_rect.width()) * max(0, new_rect.height())
                    area2 = max(0, ext_rect.width()) * max(0, ext_rect.height())
                    union_area = area1 + area2 - inter_area
                    
                    if union_area > 0 and (inter_area / union_area) > 0.8:
                        is_duplicate = True
                        break
                        
            if is_duplicate:
                continue
            
            added_count += 1
            class_counts[label] = class_counts.get(label, 0) + 1
            
            if label not in self.class_list:
                self.add_class_to_list(label)
                
            new_shape = None
            if shape_type == "rect":
                new_shape = RectShape(data, label)
            elif shape_type == "poly" or shape_type == "rbox":
                new_shape = PolyShape(data, label)
            elif shape_type == "pose":
                rect = data["rect"]
                kps = data["keypoints"]
                
                # 优先使用模型预测结果中自带的骨架连接逻辑 (YOLO 官方定义)
                if "skeleton" in s:
                    yolo_skeleton = s["skeleton"]
                    kpt_names = s.get("kpt_names", [])
                    
                    # 构造临时模板，直接遵循 YOLO 官网/模型的连接定义
                    template = {
                        "name": f"YOLO_Auto_{len(kps)}",
                        "label": label,
                        "keypoints": [],
                        "connections": []
                    }
                    
                    # 填充点名称
                    for i in range(len(kps)):
                        name = kpt_names[i] if i < len(kpt_names) else f"kp_{i}"
                        template["keypoints"].append({"name": name, "color": "#00FF00", "default_pos": [0.5, 0.5]})
                        
                    # 填充连接线 (YOLO 官网通常是 1-based 索引，需要转为 0-based)
                    for edge in yolo_skeleton:
                        if len(edge) == 2:
                            p1, p2 = edge[0], edge[1]
                            # 自动检测并转换 1-based 到 0-based
                            if p1 > 0 and p2 > 0:
                                template["connections"].append([p1 - 1, p2 - 1])
                            else:
                                template["connections"].append([p1, p2])
                                
                    # 保存动态构建的骨架模板到本地模板库，以便再次打开时完美连线
                    self.template_manager.add_template(template)
                else:
                    # 兜底方案：尝试从本地模板库匹配
                    template = self.scene.current_pose_template
                    if not template or len(template.get("keypoints", [])) != len(kps):
                        found_template = False
                        for t in self.template_manager.templates:
                            if len(t.get("keypoints", [])) == len(kps):
                                template = t
                                found_template = True
                                break
                        
                        if not found_template:
                            template = {"name": f"YOLO_Pose_{len(kps)}", "keypoints": [], "connections": []}
                            for i in range(len(kps)):
                                template["keypoints"].append({"name": f"kp_{i}", "color": "#00FF00", "default_pos": [0.5, 0.5]})
                        
                new_shape = PoseShape(rect, template, label)
                for i, kp in enumerate(kps):
                    if i < len(new_shape.kps):
                        local_pt = new_shape.mapFromScene(kp["pos"])
                        new_shape.kps[i].setPos(local_pt)
                        new_shape.kps[i].set_visibility(kp["vis"])
                
                new_shape.update_bounding_box()
                new_shape.update_lines()
                
            if new_shape:
                self.scene.addItem(new_shape)
                if hasattr(new_shape, 'update_label_text'):
                    new_shape.update_label_text(label)
                if hasattr(new_shape, 'update_label_position'):
                    new_shape.update_label_position(new_shape)
                if hasattr(new_shape, 'set_color'):
                    new_shape.set_color(self.classListWidget.get_class_color(label))
                    
        if added_count == 0:
            DialogOver(self, "暂无新的预测内容需要添加", "提示", "info")
            self.statusBar.showMessage("预测完成，暂无新内容", 3000)
            self._update_help_text(self.scene.mode)
            self._auto_save_and_advance_after_smart()
            return
        
        self.save_classes()
        self.push_state()
        self.auto_save_annotation()
        
        # Format message
        msg_parts = [f"{cls} ({count})" for cls, count in class_counts.items()]
        msg = "添加了" + str(added_count) + "个目标：" + ", ".join(msg_parts)
        DialogOver(self, msg + "，并进行标注", "预测成功", "success")
        self.statusBar.showMessage(msg, 5000)
        self._update_help_text(self.scene.mode)
        self._auto_save_and_advance_after_smart()

    def on_predict_error(self, err_msg):
        self.predict_movie.stop()
        self.btnPredict.setText(self.original_predict_text)
        self.btnPredict.setIcon(QIcon("ui/icon/lightning-fill.svg"))
        self.btnPredict.setEnabled(True)
        DialogOver(self, f"预测失败: {err_msg}", "错误", "danger")
        self.statusBar.showMessage("预测出错", 3000)
        self._update_help_text(self.scene.mode)

    def on_class_filter_clicked(self):
        if not getattr(self, 'current_yolo_predictor', None):
            DialogOver(self, "YOLO 模型未加载或初始化失败！", "提示", "warning")
            return
            
        model = self.current_yolo_predictor.model
        if not hasattr(model, 'names') or not model.names:
            DialogOver(self, "未能从当前 YOLO 模型中提取类别列表！", "提示", "warning")
            return
            
        dialog = YoloClassFilterDialog(
            class_names=model.names,
            selected_ids=self.yolo_filtered_class_ids,
            is_dark_theme=self.is_dark_theme,
            parent=self
        )
        
        # Position dialog right below the btnClassFilter button
        try:
            pos = self.btnClassFilter.mapToGlobal(self.btnClassFilter.rect().bottomLeft())
            # Align the dialog center-ish with the button (offsetting x by -60 pixels)
            dialog.move(pos.x() - 60, pos.y() + 5)
        except Exception as e:
            print(f"Failed to position class filter dialog: {e}")
            
        if dialog.exec() == QDialog.Accepted:
            self.yolo_filtered_class_ids = dialog.selected_ids
            # Update button text dynamically
            if not self.yolo_filtered_class_ids:
                self.btnClassFilter.setText(" 全部类别 ▾")
            elif len(self.yolo_filtered_class_ids) == 1:
                cid = self.yolo_filtered_class_ids[0]
                cname = model.names[cid]
                self.btnClassFilter.setText(f" {cname} ▾")
            else:
                self.btnClassFilter.setText(f" 已选 {len(self.yolo_filtered_class_ids)} 类 ▾")

    def _on_class_added_from_widget(self, cls_name):
        """从 ClassListWidget 添加新类别的回调"""
        if cls_name not in self.class_list:
            self.class_list.append(cls_name)
            self.save_classes()

    def _on_class_renamed_from_widget(self, old_name, new_name):
        """从 ClassListWidget 重命名类别的回调"""
        if old_name in self.class_list:
            idx = self.class_list.index(old_name)
            self.class_list[idx] = new_name
            # 遍历画板更新形状标签
            changed = False
            for shape in self.scene.items():
                if isinstance(shape, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                    if getattr(shape, 'label', '') == old_name:
                        shape.label = new_name
                        if hasattr(shape, 'update_label_text'):
                            shape.update_label_text(new_name)
                        changed = True
            self.save_classes()
            
            # 批量更新目录下的其他标注文件 (json, xml)
            if hasattr(self, 'current_dir') and self.current_dir:
                try:
                    import glob
                    import xml.etree.ElementTree as ET
                    import os, json
                    
                    # 更新 JSON
                    for json_file in glob.glob(os.path.join(self.current_dir, "*.json")):
                        if os.path.basename(json_file) == "class_colors.json":
                            continue
                        modified_file = False
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if 'shapes' in data:
                                for s in data['shapes']:
                                    if s.get('label') == old_name:
                                        s['label'] = new_name
                                        modified_file = True
                            if modified_file:
                                with open(json_file, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                            
                    # 更新 XML
                    for xml_file in glob.glob(os.path.join(self.current_dir, "*.xml")):
                        modified_file = False
                        try:
                            tree = ET.parse(xml_file)
                            root = tree.getroot()
                            for obj in root.findall('object'):
                                name_node = obj.find('name')
                                if name_node is not None and name_node.text == old_name:
                                    name_node.text = new_name
                                    modified_file = True
                            if modified_file:
                                tree.write(xml_file, encoding='utf-8', xml_declaration=True)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"批量更新文件失败: {e}")
            
            if changed:
                self.auto_save_annotation()
                self.push_state()
            DialogOver(self, f"已将所有的 '{old_name}' 批量变更为 '{new_name}'", "修改成功", "success")

    def on_class_color_changed(self, cls_name, color):
        """类别颜色变更时，同步更新画布上所有该类别的形状颜色"""
        from PySide6.QtGui import QColor as _QC
        if not isinstance(color, _QC):
            color = _QC(color)
        for item in self.scene.items():
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                if getattr(item, 'label', '') == cls_name and hasattr(item, 'set_color'):
                    item.set_color(color)

    def add_class_to_list(self, cls_name):
        """列表项支持双击编辑"""
        if cls_name not in self.class_list:
            self.class_list.append(cls_name)
            self.classListWidget.add_class(cls_name)

    def push_state(self):
        """把当前画布状态拍个快照，存进撤销堆栈"""
        if not self.current_image_path: return
        current_state = Exporter.extract_shapes(self.scene)

        # 如果拖了一下鼠标但什么都没变，就不存，节约内存
        if self.undo_stack:
            last_state = self.undo_stack[-1]
            if json.dumps(last_state, sort_keys=True) == json.dumps(current_state, sort_keys=True):
                return

        self.undo_stack.append(current_state)
        # 限制最大步数
        if len(self.undo_stack) > self.max_history_steps:
            self.undo_stack.pop(0)

        # 一旦有新操作，重做（前进）堆栈必须清空
        self.redo_stack.clear()
        self.update_undo_redo_buttons()

    def undo(self):
        """撤销：多边形绘制中撤销顶点，其他情况撤销整步操作"""
        from labelpaw.graphics.canvas import CanvasMode
        # 多边形绘制中：撤销最后一个顶点
        if self.scene.mode == CanvasMode.POLY and len(self.scene.poly_pts) > 0:
            removed_pt = self.scene.poly_pts.pop()
            if not hasattr(self, '_poly_redo_pts'):
                self._poly_redo_pts = []
            self._poly_redo_pts.append(removed_pt)
            self.scene.update_temp_poly()
            self.update_undo_redo_buttons()
            return

        # 常规撤销：恢复上一步画布状态
        if len(self.undo_stack) > 1:
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            previous_state = self.undo_stack[-1]
            self.restore_state(previous_state)
            self.update_undo_redo_buttons()

    def redo(self):
        """重做：多边形绘制中恢复顶点，其他情况前进一步"""
        from labelpaw.graphics.canvas import CanvasMode
        # 多边形绘制中：恢复被撤销的顶点
        if self.scene.mode == CanvasMode.POLY and hasattr(self, '_poly_redo_pts') and self._poly_redo_pts:
            pt = self._poly_redo_pts.pop()
            self.scene.poly_pts.append(pt)
            self.scene.update_temp_poly()
            self.update_undo_redo_buttons()
            return

        # 常规重做
        if self.redo_stack:
            next_state = self.redo_stack.pop()
            self.undo_stack.append(next_state)
            self.restore_state(next_state)
            self.update_undo_redo_buttons()
            
    def update_undo_redo_buttons(self):
        """更新撤销和重做按钮的可用状态（含多边形顶点撤销）"""
        from labelpaw.graphics.canvas import CanvasMode
        # 撤销可用：有历史状态 或 多边形绘制中有顶点可撤销
        can_undo = len(self.undo_stack) > 1
        if self.scene.mode == CanvasMode.POLY and len(self.scene.poly_pts) > 0:
            can_undo = True

        # 重做可用：有重做状态 或 多边形绘制中有被撤销的顶点
        can_redo = len(self.redo_stack) > 0
        if self.scene.mode == CanvasMode.POLY and hasattr(self, '_poly_redo_pts') and self._poly_redo_pts:
            can_redo = True

        self.btnUndo.setEnabled(can_undo)
        self.btnRedo.setEnabled(can_redo)
        
        # 强制刷新图标渲染
        from PySide6.QtGui import QIcon
        self.btnUndo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-left.svg"), self.current_icon_color))
        self.btnRedo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-right.svg"), self.current_icon_color))

    def restore_state(self, state):
        """根据快照数据，完全重建画板元素"""
        
        # 记录当前选中状态以便恢复
        selected_labels_or_poses = []
        for item in self.scene.selectedItems():
            if isinstance(item, PoseShape):
                # We can't perfectly match pose instances across re-creation easily, 
                # but we can try to match by rect center and label
                selected_labels_or_poses.append({"type": "pose", "center": item.pos()})
            elif hasattr(item, 'label'):
                selected_labels_or_poses.append({"type": "other", "label": item.label})

        self.scene.clear_shapes()
        for shape_data in state:
            label = shape_data.get("label", "")
            shape_type = shape_data.get("type", "")
            points = shape_data.get("points", [])

            # 同步类别列表
            if label and label not in self.class_list:
                # self.class_list.append(label)
                # self.listClasses.addItem(label)
                self.add_class_to_list(label)
                self.save_classes()

            shape = None
            if shape_type == "rectangle" and len(points) == 2:
                rect = QRectF(points[0][0], points[0][1], points[1][0] - points[0][0], points[1][1] - points[0][1])
                shape = RectShape(rect, label)
            elif shape_type == "polygon" and len(points) >= 3:
                qpoints = [QPointF(p[0], p[1]) for p in points]
                shape = PolyShape(QPolygonF(qpoints), label)
            elif shape_type == "point" and len(points) == 1:
                shape = PointShape(QPointF(points[0][0], points[0][1]), label)
            elif shape_type == "obb":
                rect_data = shape_data.get("rect")
                angle = shape_data.get("angle", 0)
                if rect_data and len(rect_data) == 4:
                    cx, cy, w, h = rect_data[0], rect_data[1], rect_data[2], rect_data[3]
                    shape = RotatedRectShape(cx, cy, w, h, angle, label)
            elif shape_type == "pose":
                rect_data = shape_data.get("rect")
                template_name = shape_data.get("template_name", "")
                kps_data = shape_data.get("keypoints", [])
                if rect_data and len(rect_data) == 4:
                    cx, cy, w, h = rect_data
                    angle = shape_data.get("angle", 0)
                    template = self.template_manager.get_template(template_name)
                    if not template:
                        template = {"name": template_name, "keypoints": [], "connections": []}
                        for i, kp in enumerate(kps_data):
                            template["keypoints"].append({"name": f"kp_{i}", "color": "#00FF00", "default_pos": [0.5, 0.5]})
                    
                    shape = PoseShape(QRectF(cx - w / 2, cy - h / 2, w, h), template, label)
                    shape.setRotation(angle)
                    for i, kp_data in enumerate(kps_data):
                        if i < len(shape.kps):
                            local_pt = shape.mapFromScene(QPointF(kp_data[0], kp_data[1]))
                            shape.kps[i].setPos(local_pt)
                            shape.kps[i].set_visibility(kp_data[2])
                    shape.update_lines()

            if shape:
                self.scene.addItem(shape)
                if hasattr(shape, 'update_label_text'):
                    shape.update_label_text(label)
                if hasattr(shape, 'update_label_position'):
                    shape.update_label_position(shape)
                # 应用类别颜色
                if hasattr(shape, 'set_color'):
                    shape.set_color(self.classListWidget.get_class_color(label))
                
                # 尝试恢复选中状态 (保持编辑模式)
                is_selected = False
                if shape_type == "pose":
                    for s in selected_labels_or_poses:
                        if s["type"] == "pose" and (s["center"] - shape.pos()).manhattanLength() < 5:
                            is_selected = True
                            break
                else:
                    for s in selected_labels_or_poses:
                        if s["type"] == "other" and s["label"] == label:
                            is_selected = True
                            break
                
                shape.setSelected(is_selected)
                
                if hasattr(shape, 'update_label_visibility'):
                    shape.update_label_visibility(shape, is_selected=is_selected, is_hovered=False)
                if hasattr(shape, '_update_handle_visibility'):
                    shape._update_handle_visibility()

        self.auto_save_annotation()
        self.update_annotation_tree()

    def open_dataset_tool(self):
        try:
            if not hasattr(self, 'dataset_window') or self.dataset_window is None:
                self.dataset_window = DatasetToolWindow()
            # 显示窗口
            self.dataset_window.show()
            # 把窗口强制拉到最前面
            self.dataset_window.raise_()
            self.dataset_window.activateWindow()

        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def trigger_sam_prompt(self):
        prompt = self.samPromptInput.text().strip()
        if not prompt:
            DialogOver(self, "请输入提示词进行提取！", "提示", "warning")
            return

        # 收集所有勾选的文件
        checked_paths = []
        for i in range(self.listFiles.count()):
            item = self.listFiles.item(i)
            if item and item.checkState() == Qt.Checked:
                checked_paths.append(item.text())

        if len(checked_paths) > 1:
            # 批量处理逻辑
            if self.sam_client.current_model_type != "sam3" or not self.sam_client.model or not self.sam_client.processor:
                DialogOver(self, "请先在上方选择并加载 SAM 3 模型以进行批量提示词处理！", "提示", "warning")
                return

            import re
            prompts = [p.strip() for p in re.split(r'[,，、\s]+', prompt) if p.strip()]
            if not prompts:
                DialogOver(self, "输入的提示词无效，请重新输入！", "提示", "warning")
                return

            self.batch_dialog = BatchProgressDialog(self, self.is_dark_theme)
            
            self.batch_worker = SamBatchWorker(
                processor=self.sam_client.processor,
                model=self.sam_client.model,
                img_paths=checked_paths,
                prompts=prompts,
                current_format=self.current_format,
                class_list=self.class_list,
                canvas_mode=self.scene.mode,
                overwrite=self.btnOverwrite.isChecked()
            )

            # 连接进度更新
            def update_progress(current, total, filename):
                percent = int((current / total) * 100) if total > 0 else 0
                self.batch_dialog.progress_bar.setValue(percent)
                self.batch_dialog.status_label.setText(f"正在标注 ({current + 1}/{total}):\n{filename}")
                
            self.batch_worker.progress.connect(update_progress)

            # 完成回调
            def on_finished(processed, total):
                self.batch_dialog.movie.stop()
                self.batch_dialog.accept()
                
                # 同步类别到历史面板
                for p in prompts:
                    if p not in self.class_list:
                        self.add_class_to_list(p)
                self.save_classes()
                
                # 如果当前打开的图片在被批量标注的列表中，重新加载以显示新标注
                if self.current_image_path in checked_paths:
                    self.scene.clear_shapes()
                    self.load_annotations(self.current_image_path)
                    self.apply_class_colors_to_scene()
                    self.update_annotation_tree()
                    self.push_state()
                
                DialogOver(self, f"批量标注完成！成功处理 {processed}/{total} 张图片。", "批量标注", "success")
                
            self.batch_worker.finished.connect(on_finished)

            # 错误回调
            def on_error(err_msg):
                self.batch_dialog.movie.stop()
                self.batch_dialog.reject()
                DialogOver(self, f"批量标注出错: {err_msg}", "错误", "danger")
                
            self.batch_worker.error.connect(on_error)

            # 取消按钮连接
            self.batch_dialog.btn_cancel.clicked.connect(lambda: setattr(self.batch_worker, 'is_cancelled', True))

            self.batch_worker.start()
            self.batch_dialog.exec()
        else:
            # 单张图片处理逻辑
            if self.scene.mode == CanvasMode.POINT:
                DialogOver(self, "点标注模式下无法使用 SAM 智能提取", "提示", "warning")
                return

            if not self.current_image_path:
                DialogOver(self, "请先打开一张图片！", "提示", "warning")
                return

            self.samSwitch.setChecked(True)
            self.helpLabel.setText(f"正在提取提示词: {prompt}...")
            self.helpLabel.setStyleSheet("color: orange;")
            self.sam_client.request_text_inference(prompt)

    def handle_text_results(self, results, prompt_text):
        if not results:
            self.helpLabel.setText(f"提取完成: 未发现关于 '{prompt_text}' 的目标")
            self.helpLabel.setStyleSheet("color: red;")
            if self.btnOverwrite.isChecked():
                self.scene.clear_shapes()
                self.update_annotation_tree()
                self.auto_save_annotation()
                self.push_state()
            self._auto_save_and_advance_after_smart()
            return

        self.helpLabel.setText(f"提取完成: 成功抓取 {len(results)} 个 '{prompt_text}' 目标")
        self.helpLabel.setStyleSheet("color: green;")

        if prompt_text not in self.class_list:
            self.add_class_to_list(prompt_text)
            self.save_classes()

        if self.btnOverwrite.isChecked():
            self.scene.clear_shapes()

        for res in results:
            if self.scene.mode == CanvasMode.RECT:
                x, y, w, h = res["rect"]
                shape = RectShape(QRectF(x, y, w, h), prompt_text)
            elif self.scene.mode == CanvasMode.RBOX:
                if "obb" in res and len(res["obb"]) == 5:
                    cx, cy, w, h, angle = res["obb"]
                    shape = RotatedRectShape(cx, cy, w, h, angle, prompt_text)
                else:
                    x, y, w, h = res["rect"]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    shape = RotatedRectShape(cx, cy, w, h, 0, prompt_text)
            else:
                qpts = [QPointF(p[0], p[1]) for p in res["poly_pts"]]
                shape = PolyShape(QPolygonF(qpts), prompt_text)

            self.scene.addItem(shape)
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(prompt_text)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            if hasattr(shape, 'update_label_visibility'):
                shape.update_label_visibility(shape, is_selected=False, is_hovered=False)
            if hasattr(shape, 'set_color'):
                shape.set_color(self.classListWidget.get_class_color(prompt_text))

        self.update_annotation_tree()
        self.auto_save_annotation()
        self.push_state()
        self._auto_save_and_advance_after_smart()

    def delete_selected(self):
        for item in self.scene.selectedItems():
            self.scene.removeItem(item)
        self.scene.state_changed.emit()

    def toggle_sidebar(self):
        if self.toolBar.toolButtonStyle() == Qt.ToolButtonTextBesideIcon:
            # 隐藏文字 — 收缩模式
            self.toolBar.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.toolBar.setFixedWidth(50)
            self.logoLabel.hide()
            
            # 收缩模式：缩小 Logo 图标
            logo_path = "ui/icon/logo.png"
            pix = QPixmap(logo_path).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logoIcon.setPixmap(pix)
            self.logoIcon.setFixedSize(24, 24)

            # 释放按钮宽度限制
            for btn in self._actionButtons:
                btn.setMinimumWidth(0)

            self.formatWidget.set_icon_only(True)
            self.btnDatasetTool.setText("")
            self.samIcon.hide() # 收缩时隐藏左侧图标
            # SAM 开关竖向显示
            self.samSwitch.setFixedSize(26, 50)
            self.samSwitch._vertical = True
            self.samSwitch.update()
        else:
            # 显示文字 — 展开模式
            self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.toolBar.setFixedWidth(190)
            
            # 展开模式：恢复 Logo 图标和文字
            self.logoLabel.show()
            self.logoLabel.setText("LabelPaw")
            logo_path = "ui/icon/logo.png"
            pix = QPixmap(logo_path).scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logoIcon.setPixmap(pix)
            self.logoIcon.setFixedSize(28, 28)

            # 恢复按钮宽度
            for btn in self._actionButtons:
                btn.setMinimumWidth(180)

            self.formatWidget.set_icon_only(False)
            self.btnDatasetTool.setText(" 数据集处理")
            self.samIcon.show() # 展开时显示左侧图标
            # SAM 开关横向显示
            self.samSwitch.setFixedSize(50, 26)
            self.samSwitch._vertical = False
            self.samSwitch.update()

    def set_icon_color(self, icon, color):
        from PySide6.QtGui import QPainter, QIcon, QPixmap, QColor
        pixmap = icon.pixmap(100, 100)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        
        # 核心逻辑：自己创建一个可以响应 enable/disable 的 QIcon
        # 我们把这个纯色图作为 Normal 状态
        new_icon = QIcon()
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        
        # 为了让禁用状态变灰，我们用 QPainter 绘制一个半透明的版本作为 Disabled 状态
        disabled_pixmap = icon.pixmap(100, 100)
        dpainter = QPainter(disabled_pixmap)
        dpainter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        
        # 深色主题下禁用颜色为半透明白，浅色主题下为半透明黑
        if color.lightness() > 128:
            disabled_color = QColor(255, 255, 255, 60)
        else:
            disabled_color = QColor(15, 23, 42, 60)
            
        dpainter.fillRect(disabled_pixmap.rect(), disabled_color)
        dpainter.end()
        
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.On)
        new_icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.Off)
        
        return new_icon

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME)
            self.btnThemeToggle.setText("☀")
            
            # 更新深色图标
            from PySide6.QtGui import QIcon, QColor
            self.current_icon_color = QColor(255, 255, 255)
            self.btnUndo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-left.svg"), self.current_icon_color))
            self.btnRedo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-right.svg"), self.current_icon_color))
            self.btnDelete.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
            self.btnDeleteFiles.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
            self.btnSave.setIcon(self.set_icon_color(QIcon("ui/icon/floppy-disk.svg"), self.current_icon_color))
            self.btnKeyboard.setIcon(self.set_icon_color(QIcon("ui/icon/keyboard.svg"), self.current_icon_color))
            
            # 更新侧边栏图标
            self.actionOpen.setIcon(self.set_icon_color(QIcon("ui/icon/folder.svg"), self.current_icon_color))
            self.actionRect.setIcon(self.set_icon_color(QIcon("ui/icon/rectangle.svg"), self.current_icon_color))
            self.actionPoly.setIcon(self.set_icon_color(QIcon("ui/icon/polygon.svg"), self.current_icon_color))
            self.actionPoint.setIcon(self.set_icon_color(QIcon("ui/icon/关键点.svg"), self.current_icon_color))
            self.actionRBox.setIcon(self.set_icon_color(QIcon("ui/icon/手机旋转1.svg"), self.current_icon_color))
            self.samIcon.setPixmap(self.set_icon_color(QIcon("ui/icon/魔法-copy.svg"), self.current_icon_color).pixmap(24, 24))
            self.btnDatasetTool.setIcon(self.set_icon_color(QIcon("ui/icon/wrench.svg"), self.current_icon_color))
            self.formatWidget.btn.setIcon(self.set_icon_color(QIcon("ui/icon/格式.svg"), self.current_icon_color))
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.btnThemeToggle.setText("🌙")
            
            # 更新浅色图标
            from PySide6.QtGui import QIcon, QColor
            self.current_icon_color = QColor(15, 23, 42) # 深灰色
            self.btnUndo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-left.svg"), self.current_icon_color))
            self.btnRedo.setIcon(self.set_icon_color(QIcon("ui/icon/arrow-u-up-right.svg"), self.current_icon_color))
            self.btnDelete.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
            self.btnDeleteFiles.setIcon(self.set_icon_color(QIcon("ui/icon/trash.svg"), self.current_icon_color))
            self.btnSave.setIcon(self.set_icon_color(QIcon("ui/icon/floppy-disk.svg"), self.current_icon_color))
            self.btnKeyboard.setIcon(self.set_icon_color(QIcon("ui/icon/keyboard.svg"), self.current_icon_color))
            
            # 更新侧边栏图标
            self.actionOpen.setIcon(self.set_icon_color(QIcon("ui/icon/folder.svg"), self.current_icon_color))
            self.actionRect.setIcon(self.set_icon_color(QIcon("ui/icon/rectangle.svg"), self.current_icon_color))
            self.actionPoly.setIcon(self.set_icon_color(QIcon("ui/icon/polygon.svg"), self.current_icon_color))
            self.actionPoint.setIcon(self.set_icon_color(QIcon("ui/icon/关键点.svg"), self.current_icon_color))
            self.actionRBox.setIcon(self.set_icon_color(QIcon("ui/icon/手机旋转1.svg"), self.current_icon_color))
            self.samIcon.setPixmap(self.set_icon_color(QIcon("ui/icon/魔法-copy.svg"), self.current_icon_color).pixmap(24, 24))
            self.btnDatasetTool.setIcon(self.set_icon_color(QIcon("ui/icon/wrench.svg"), self.current_icon_color))
            self.formatWidget.btn.setIcon(self.set_icon_color(QIcon("ui/icon/格式.svg"), self.current_icon_color))
        self.classListWidget.set_theme(self.is_dark_theme)
        self.update_prompt_btn_icon()

    def show_author_info(self):
        dialog = AuthorInfoDialog(self)
        dialog.exec()

    def show_help_dialog(self):
        help_text = """
        <h3>【快捷键大全】</h3>
        <ul>
            <li><b>A / 左方向键</b>：上一张图片</li>
            <li><b>D / 右方向键</b>：下一张图片</li>
            <li><b style="color:red;">Ctrl + S</b>：保存当前标注</li>
            <li><b style="color:blue;">Q</b>：开启/关闭 SAM 智能辅助</li>
            <li><b>R</b>：切换至 矩形标注</li>
            <li><b>P</b>：切换至 多边形标注</li>
            <li><b>T</b>：切换至 关键点标注</li>
            <li><b>O</b>：切换至 旋转框标注</li>
            <li><b>M</b>：触发 模型预测 (Smart 模式并加载模型时可用)</li>
            <li><b>Del / Backspace</b>：删除当前选中的标注框</li>
            <li><b>F1</b>：打开此帮助文档</li>
        </ul>
        <hr>
        <h3>【多边形绘制技巧】</h3>
        <ul>
            <li><b>左键点击</b>：添加顶点</li>
            <li><b>Ctrl + Z</b>：撤销上一个顶点</li>
            <li><b>双击 / Enter</b>：闭合多边形</li>

        </ul>
        <hr>
        <h3>【旋转框绘制快捷键】</h3>
        <ul>
            <li><b> Z / V</b>：每次向左/向右旋转 5°</li>
            <li><b>X / C</b>：每次向左/向右旋转 1°</li>
        </ul>
        <hr>
        <h3>【SAM 智能辅助】</h3>
        <ul>
            <li><b>鼠标点选</b>：开启开关后，鼠标悬停预览，点击直接确认生成高精度轮廓。</li>
            <li><b>提示词提取</b>：在右下角输入框输入目标名称（如: dog、cat），按回车即可一键全图抓取并打好框！左侧选中的是“矩形”还是“多边形”格式。</li>
        </ul>
        """
        QMessageBox.about(self, "LabelPaw 使用说明", help_text)

    def update_coordinate_label(self, x, y):
        self.coordLabel.setText(f"坐标: X: {x}, Y: {y}")

    def on_sam_toggled(self, checked):
        if checked and self.scene.mode == CanvasMode.POINT:
            yolo_pred = getattr(self, "current_yolo_predictor", None)
            is_yolo_pose = yolo_pred is not None and getattr(yolo_pred, "task", "") == "pose"
            
            if not is_yolo_pose:
                # 尝试寻找并加载默认的 YOLO pose 模型
                import sys
                if getattr(sys, 'frozen', False):
                    PROJECT_ROOT = os.path.dirname(sys.executable)
                else:
                    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
                LOCAL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
                HARDCODED_DEV_DIR = r"E:\11-AI\标注工具\weights"
                
                if os.path.exists(LOCAL_WEIGHTS_DIR):
                    model_base_dir = LOCAL_WEIGHTS_DIR
                elif os.path.exists(HARDCODED_DEV_DIR):
                    model_base_dir = HARDCODED_DEV_DIR
                else:
                    model_base_dir = LOCAL_WEIGHTS_DIR
                
                default_pose_path = None
                default_type = ""
                default_name = ""
                
                if os.path.exists(model_base_dir):
                    for item in os.listdir(model_base_dir):
                        item_path = os.path.join(model_base_dir, item)
                        if os.path.isdir(item_path) and item.startswith("yolo") and item.endswith("_weights"):
                            for f in os.listdir(item_path):
                                if f.endswith('-pose.pt') or f.endswith('-pose.onnx'):
                                    default_pose_path = os.path.join(item_path, f)
                                    default_name = f.replace('.pt', '').replace('.onnx', '')
                                    default_type = item.replace('_weights', '')
                                    break
                        if default_pose_path:
                            break
                            
                if default_pose_path and os.path.exists(default_pose_path):
                    self.statusBar.showMessage(f"正在为您自动切换至 {default_name} 模型...", 3000)
                    self.on_model_selected({
                        "key": default_name,
                        "display_name": default_name,
                        "type": default_type,
                        "path": default_pose_path
                    })
                else:
                    # 不再强行拦截，而是提醒用户切换模型，这样模型选择器才能显示出来
                    self.statusBar.showMessage("当前为点标注模式，请在上方选择 YOLO 姿态模型以使用智能辅助", 4000)
                    self.helpLabel.setText("请选择 YOLO 姿态模型")
                    self.helpLabel.setStyleSheet("color: orange;")

        self.scene.set_sam_enabled(checked)
        self._update_help_text(self.scene.mode)
        
        if checked:
            self.btnModelSelector.show()
            if getattr(self, 'current_yolo_predictor', None) is not None:
                self.btnPredict.show()
                self.btnClassFilter.show()
            if not self.btnSmartMode.isChecked():
                self.btnSmartMode.setChecked(True)
        else:
            self.btnModelSelector.hide()
            self.btnPredict.hide()
            self.btnClassFilter.hide()
            if not self.btnDrawMode.isChecked():
                self.btnDrawMode.setChecked(True)

        if self.scene.mode == CanvasMode.POINT:
            if checked:
                self.templateWidget.hide()
                self.sepTemplate.hide()
            else:
                self.templateWidget.show()
                self.sepTemplate.show()

    def on_pose_template_changed(self, text):
        if not text: return
        if text == "+ New Template...":
            dlg = SkeletonTemplateDialog(self, self.template_manager, self.is_dark_theme)
            if dlg.exec() == QDialog.Accepted:
                self._init_pose_templates()
                # Select the newly created template
                last_template = self.template_manager.get_template_names()[-1]
                self.templateWidget._on_template_selected(last_template, f"{last_template} ▾")
            else:
                self.templateWidget._on_template_selected("Person (COCO)", "Person (COCO) ▾")
        else:
            self.scene.current_pose_template = self.template_manager.get_template(text)
            self._update_help_text(self.scene.mode)
            self.formatWidget.set_yolo_enabled(True)
        # Ensure we are in point mode
        self._set_mode(CanvasMode.POINT)

    def _set_mode(self, mode):
        self.scene.set_mode(mode)
        # 切换模式时清空多边形顶点重做栈
        self._poly_redo_pts = []
        mode_name = CanvasMode.get_mode_name(mode)
        self.modeLabel.setText(f"模式: {mode_name}标注")
        self._update_help_text(mode)

        if mode == CanvasMode.RECT:
            self.actionRect.setChecked(True)
        elif mode == CanvasMode.POLY:
            self.actionPoly.setChecked(True)
        elif mode == CanvasMode.POINT:
            self.actionPoint.setChecked(True)
        elif mode == CanvasMode.RBOX:
            self.actionRBox.setChecked(True)

        if mode == CanvasMode.POINT:
            yolo_pred = getattr(self, 'current_yolo_predictor', None)
            is_yolo_pose = yolo_pred is not None and getattr(yolo_pred, 'task', '') == 'pose'

            if self.samSwitch.isChecked():
                self.templateWidget.hide()
                self.sepTemplate.hide()
            else:
                self.templateWidget.show()
                self.sepTemplate.show()

            self.samSwitch.setEnabled(True)
            
            # 移除自动开启逻辑，保持用户当前的手动/智能选择
            
            self.samPromptInput.setEnabled(False)
            self.samPromptBtn.setEnabled(False)
            self.samPromptInput.setPlaceholderText("点标注模式下文本提示不可用")
            
            if not self.scene.current_pose_template and not is_yolo_pose:
                self.formatWidget.set_yolo_enabled(False)
            else:
                self.formatWidget.set_yolo_enabled(True)
        else:
            self.templateWidget.hide()
            self.sepTemplate.hide()
            self.samSwitch.setEnabled(True)
            
            supports_text = self.sam_client.supports_text_prompt()
            self.samPromptInput.setEnabled(supports_text)
            self.update_prompt_btn_state()
            
            if supports_text:
                self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog、cat)")
            else:
                self.samPromptInput.setPlaceholderText("当前模型不支持提示词")
                
            self.formatWidget.set_yolo_enabled(True)

    def _update_help_text(self, mode):
        # 只有在智能模式开启，并且当前选择的是SAM模型时，才显示悬停预览提示
        is_sam = self.samSwitch.isChecked() and (getattr(self.sam_client, 'current_model_key', '') or '').startswith('sam')
        if mode == CanvasMode.RECT:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览外接矩形，左键点击直接确认生成矩形框")
            else:
                self.helpLabel.setText("操作: 拖动鼠标绘制常规矩形")
        elif mode == CanvasMode.POLY:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览轮廓节点，左键点击直接确认生成多边形")
            else:
                self.helpLabel.setText("操作: 点击添加顶点，双击闭合多边形")
        elif mode == CanvasMode.POINT:
            self.helpLabel.setText("操作: 点击放置骨架模板，选定后可通过手柄拖拽放大、旋转或微调关键点")
        elif mode == CanvasMode.RBOX:
            self.helpLabel.setText("操作: 拖动绘制旋转框，Z/X/C/V调整角度")

    def load_classes(self, dir_path):
        self.class_list.clear()
        self.classListWidget.load_classes(dir_path)
        self.class_list = self.classListWidget.get_class_list()

    def save_classes(self):
        if self.current_dir:
            self.classListWidget.set_working_dir(self.current_dir)
            self.classListWidget._class_list = list(self.class_list)
            self.classListWidget.save_classes()

    def apply_class_colors_to_scene(self):
        """加载标注后，根据类别颜色映射为所有形状设置颜色"""
        for item in self.scene.items():
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                label = getattr(item, 'label', '')
                if label and hasattr(item, 'set_color'):
                    color = self.classListWidget.get_class_color(label)
                    item.set_color(color)

    def handle_new_shape(self, shape):
        self.scene.addItem(shape)
        QApplication.processEvents()

        from labelpaw.graphics.shapes import PoseShape
        if isinstance(shape, PoseShape) and shape.template:
            # 骨架模板自动使用模板标签作为类别，跳过弹窗
            cls_name = shape.template.get("label", shape.template.get("name", "Unknown"))
            ok = True
        else:
            # 去除弹窗：直接获取当前选中的类别，没有则使用第一个或默认 "dog"
            selected_cls = self.classListWidget.get_selected_class()
            if not selected_cls:
                classes = self.classListWidget.get_class_list()
                selected_cls = classes[0] if classes else "dog"
            cls_name = selected_cls
            ok = True

        if ok and cls_name:
            cls_name = cls_name.strip()
            if cls_name not in self.class_list:
                # self.class_list.append(cls_name)
                # self.listClasses.addItem(cls_name)
                self.add_class_to_list(cls_name)
                self.save_classes()

            shape.label = cls_name
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(cls_name)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            
            from labelpaw.graphics.shapes import RectShape, PolyShape, RotatedRectShape
            is_rect_poly_obb = isinstance(shape, (RectShape, PolyShape, RotatedRectShape))
            
            if hasattr(shape, 'update_label_visibility'):
                shape.update_label_visibility(shape, is_selected=(not is_rect_poly_obb), is_hovered=False)
            # 应用类别颜色
            if hasattr(shape, 'set_color'):
                shape.set_color(self.classListWidget.get_class_color(cls_name))
                
            for item in self.scene.selectedItems():
                item.setSelected(False)
                
            if is_rect_poly_obb:
                shape.setSelected(False)
            else:
                shape.setSelected(True)
                
            self.push_state()
            self.update_annotation_tree()
            if self._is_smart_annotation_active():
                self._auto_save_and_advance_after_smart()
        else:
            self.scene.removeItem(shape)

    def edit_shape_label(self, shape):
        """二次修改已有标注框的类别 (根据用户要求，弹窗已去除，双击忽略或做其他逻辑)"""
        pass

    def on_list_item_changed(self, item):
        """ClassListWidget item_changed fallback (rename already handled by class_renamed signal)"""
        pass

    def on_shape_class_reassigned(self, shape, new_class_name):
        """当用户在右侧历史类别列表点击父类别时，将画布上已选中图形的类别修改为该类别"""
        old_label = getattr(shape, 'label', '')
        if old_label == new_class_name:
            return  # 类别未变化，无需操作

        shape.label = new_class_name
        if hasattr(shape, 'update_label_text'):
            shape.update_label_text(new_class_name)
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)
        # 应用新类别的颜色
        if hasattr(shape, 'set_color'):
            shape.set_color(self.classListWidget.get_class_color(new_class_name))

        # 更新标注树和保存
        self.update_annotation_tree()
        self.auto_save_annotation()
    def update_annotation_tree(self):
        shapes = []
        for item in self.scene.items():
            from labelpaw.graphics.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)) and not getattr(item, 'is_temp', False):
                shapes.append(item)
        self.classListWidget.update_annotations(shapes)

    def sync_selection_to_tree(self):
        from labelpaw.graphics.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape
        selected_shapes = [item for item in self.scene.selectedItems() if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)) and not getattr(item, 'is_temp', False)]
        if selected_shapes:
            self.classListWidget.select_item_by_shape(selected_shapes[0])
        else:
            self.classListWidget.clear_tree_selection()

    def on_select_all_toggled(self, state):
        self.listFiles.blockSignals(True)
        # 兼容 PySide6 整数和 CheckState 枚举比较
        is_unchecked = (state == 0 or state == Qt.Unchecked)
        if is_unchecked:
            check_state = Qt.Unchecked
        else:
            check_state = Qt.Checked
            # 强制设为 Checked 状态（以防万一）
            self.chkSelectAll.blockSignals(True)
            self.chkSelectAll.setCheckState(Qt.Checked)
            self.chkSelectAll.blockSignals(False)
        for i in range(self.listFiles.count()):
            item = self.listFiles.item(i)
            if item:
                item.setCheckState(check_state)
        self.listFiles.blockSignals(False)
        self.update_selected_count()

    def on_file_item_changed(self, item):
        self.update_selected_count()

    def update_selected_count(self):
        total = self.listFiles.count()
        checked = 0
        for i in range(total):
            item = self.listFiles.item(i)
            if item and item.checkState() == Qt.Checked:
                checked += 1
                
        self.labelSelectedCount.setText(f"(已选 {checked}/{total})")
        
        # 动态控制可用性
        has_items = total > 0
        self.chkSelectAll.setEnabled(has_items)
        self.btnDeleteFiles.setEnabled(has_items)
        
        self.chkSelectAll.blockSignals(True)
        if total == 0:
            self.chkSelectAll.setCheckState(Qt.Unchecked)
        elif checked == 0:
            self.chkSelectAll.setCheckState(Qt.Unchecked)
        elif checked == total:
            self.chkSelectAll.setCheckState(Qt.Checked)
        else:
            self.chkSelectAll.setCheckState(Qt.PartiallyChecked)
        self.chkSelectAll.setTristate(False)
        self.chkSelectAll.blockSignals(False)

    def open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录", self.current_dir or "")
        if dir_path:
            self.current_dir = dir_path
            self.listFiles.clear()
            self.load_classes(dir_path)
            self.listFiles.blockSignals(True)
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    item = QListWidgetItem(os.path.join(dir_path, f))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.listFiles.addItem(item)
            self.listFiles.blockSignals(False)
            self.update_selected_count()

            if self.listFiles.count() > 0:
                self.listFiles.setCurrentRow(0)

    # def update_model_status(self, success, msg):
    #     self.helpLabel.setText(msg)
    #     if success:
    #         self.helpLabel.setStyleSheet("color: green;")
    #     else:
    #         self.helpLabel.setStyleSheet("color: red;")

    def update_model_status(self, success, msg):
        self.helpLabel.setText(msg)
        if success:
            self.helpLabel.setStyleSheet("color: green;")
            # 模型加载成功后，检查用户是不是已经提前打开图片了
            if self.current_image_path:
                # self.helpLabel.setText("模型已就绪，正在自动分析当前图片特征...")
                # self.helpLabel.setStyleSheet("color: orange;")
                QApplication.processEvents()
                self.sam_client.set_image(self.current_image_path)
                # self.helpLabel.setText("分析完成，可以开始智能标注")
                # self.helpLabel.setStyleSheet("color: green;")
        else:
            self.helpLabel.setStyleSheet("color: red;")

    def show_file_list_context_menu(self, pos):
        """显示文件列表右键菜单"""
        selected_items = self.listFiles.selectedItems()
        if not selected_items:
            return

        menu = QMenu()
        # 应用主题样式
        if self.is_dark_theme:
            menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: #fff; border: 1px solid #444; }")
        
        count = len(selected_items)
        delete_action = menu.addAction(f"删除选中的 {count} 个文件")
        delete_action.setIcon(QIcon("ui/icon/trash.svg"))
        
        action = menu.exec(self.listFiles.mapToGlobal(pos))
        if action == delete_action:
            self.delete_selected_files(selected_items)

    def delete_selected_files(self, items):
        """批量删除选中的图片及其对应的标签文件"""
        count = len(items)

        # 记录需要删除的路径
        paths_to_delete = [item.text() for item in items]
        
        # 如果当前正在查看的文件被删除了，需要切换
        current_deleted = self.current_image_path in paths_to_delete
        
        # 暂时断开信号避免删除过程中触发切换和自动保存
        self.listFiles.currentItemChanged.disconnect(self.on_file_selected)

        if current_deleted:
            self.scene.clear_shapes()
            if self.scene.img_item:
                self.scene.removeItem(self.scene.img_item)
            self.scene.img_item = None
            self.current_image_path = None

        deleted_count = 0
        for item, img_path in zip(items, paths_to_delete):
            try:
                # 1. 删除图片文件
                if os.path.exists(img_path):
                    os.remove(img_path)
                
                # 2. 尝试删除所有可能的标签文件 (json, txt, xml)
                base_path = os.path.splitext(img_path)[0]
                for ext in [".json", ".txt", ".xml"]:
                    label_path = base_path + ext
                    if os.path.exists(label_path):
                        os.remove(label_path)
                
                # 从列表中移除
                row = self.listFiles.row(item)
                self.listFiles.takeItem(row)
                
                deleted_count += 1
            except Exception as e:
                print(f"删除文件 {img_path} 失败: {e}")

        # 恢复信号
        self.listFiles.currentItemChanged.connect(self.on_file_selected)
        self.update_selected_count()

        if current_deleted:
            self.statusBar.showMessage(f"已删除 {deleted_count} 个文件，当前预览已清除", 3000)
            # 如果列表还有文件，自动选中并加载第一项或当前项
            if self.listFiles.count() > 0:
                current_row = self.listFiles.currentRow()
                if current_row < 0:
                    current_row = 0
                item = self.listFiles.item(current_row)
                self.listFiles.setCurrentItem(item)
                # 手动触发一次加载
                self.on_file_selected(item, None)
        else:
            self.statusBar.showMessage(f"成功删除 {deleted_count} 个文件", 3000)

    def delete_checked_files(self):
        # Find all checked items
        checked_items = []
        for i in range(self.listFiles.count()):
            item = self.listFiles.item(i)
            if item and item.checkState() == Qt.Checked:
                checked_items.append(item)
                
        if not checked_items:
            DialogOver(self, "请先勾选要删除的文件！", "提示", "warning")
            return
            
        # Ask for confirmation
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, 
            "批量删除文件", 
            f"确定要删除勾选的 {len(checked_items)} 个图片及其标注文件吗？此操作不可逆！", 
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.delete_selected_files(checked_items)

    def on_file_selected(self, current, previous):
        if previous:
            self.auto_save_annotation()

        if current:
            path = current.text()
            self.current_image_path = path
            self.scene.load_image(path)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.load_annotations(path)
            self.apply_class_colors_to_scene()
            self.update_annotation_tree()

            self.undo_stack.clear()
            self.redo_stack.clear()
            self.push_state()
            
            # 清除旧图片的骨架预览
            if hasattr(self.scene, 'pose_preview_item') and self.scene.pose_preview_item:
                self.scene.removeItem(self.scene.pose_preview_item)
                self.scene.pose_preview_item = None

            # 只有SAM模型才需要后台分析图片特征
            is_sam = getattr(self.sam_client, 'current_model_key', '') and getattr(self.sam_client, 'current_model_key', '').startswith('sam')
            if is_sam:
                if self.sam_client.model:
                    QApplication.processEvents()
                    self.sam_client.set_image(path)
                else:
                    self.helpLabel.setText("等待后台加载模型，稍后将自动分析图片...")
                    self.helpLabel.setStyleSheet("color: orange;")
            else:
                # YOLO 或者没有模型时，恢复默认的标注提示词
                self._update_help_text(self.scene.mode)

    def _is_smart_annotation_active(self):
        model_key = getattr(self.sam_client, 'current_model_key', '') or ''
        return self.samSwitch.isChecked() and (model_key.startswith('sam') or model_key.startswith('yolo'))

    def _auto_save_and_advance_after_smart(self):
        if not self.auto_advance_after_smart:
            return
        if not self.current_image_path or self.listFiles.count() == 0:
            return
        QTimer.singleShot(250, self._advance_to_next_image_after_save)

    def _advance_to_next_image_after_save(self):
        self.auto_save_annotation()
        current_row = self.listFiles.currentRow()
        if current_row < 0:
            return
        if current_row < self.listFiles.count() - 1:
            self.listFiles.setCurrentRow(current_row + 1)
            self.statusBar.showMessage("已自动保存并切换到下一张图片", 3000)
        else:
            self.statusBar.showMessage("已自动保存，当前已是最后一张图片", 3000)

    def auto_save_annotation(self):
        if not self.current_image_path or not self.scene.img_item: return
        shapes_data = Exporter.extract_shapes(self.scene)
        base_name = os.path.splitext(self.current_image_path)[0]

        if not shapes_data:
            # 清理可能存在的空标注文件
            for ext in [".json", ".txt", ".xml"]:
                out_path = base_name + ext
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        print(f"自动删除空标注文件: {out_path}")
                    except Exception as e:
                        print(f"自动删除空标注文件失败: {out_path}, {e}")
            return

        img_rect = self.scene.img_item.pixmap().rect()

        try:
            if self.current_format == "json":
                out_path = base_name + ".json"
                Exporter.save_json(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
            elif self.current_format == "yolo":
                out_path = base_name + ".txt"
                Exporter.save_yolo(out_path, img_rect.width(), img_rect.height(), shapes_data, self.class_list)
            elif self.current_format == "xml":
                out_path = base_name + ".xml"
                Exporter.save_xml(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
        except Exception as e:
            print(f"自动保存失败: {str(e)}")

    def set_current_format(self, format_type):
        self.current_format = format_type
        self.formatWidget.set_format(format_type)

        # if format_type == "json":
        #     self.actionFormatJSON.setChecked(True)
        # elif format_type == "yolo":
        #     self.actionFormatYOLO.setChecked(True)
        # elif format_type == "xml":
        #     self.actionFormatXML.setChecked(True)

        if self.current_image_path:
            self.scene.clear_shapes()
            self.load_annotations(self.current_image_path)
        DialogOver(self, f"当前保存及读取格式变为 {format_type.upper()}", "格式切换", "info")

    def load_annotations(self, image_path):
        if not self.scene.img_item: return

        img_w = self.scene.img_item.pixmap().width()
        img_h = self.scene.img_item.pixmap().height()
        base_path = os.path.splitext(image_path)[0]

        if self.current_format == "json":
            self._load_json(base_path + ".json")
        elif self.current_format == "yolo":
            self._load_yolo(base_path + ".txt", img_w, img_h)
        elif self.current_format == "xml":
            self._load_xml(base_path + ".xml")

    def _add_shape_to_scene(self, shape, label):
        """往画板内添加加载出来的轮廓并同步历史类别"""
        if label not in self.class_list:
            # self.class_list.append(label)
            # self.listClasses.addItem(label)
            self.add_class_to_list(label)
            self.save_classes()
        self.scene.addItem(shape)

    def _load_json(self, json_path):
        if not os.path.exists(json_path): return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for shape_data in data.get("shapes", []):
                label = shape_data.get("label", "")
                points = shape_data.get("points", [])
                shape_type = shape_data.get("shape_type", "rectangle")

                if shape_type == "rectangle" and len(points) == 2:
                    rect = QRectF(points[0][0], points[0][1], points[1][0] - points[0][0], points[1][1] - points[0][1])
                    shape = RectShape(rect, label)
                elif shape_type == "polygon" and len(points) >= 3:
                    qpoints = [QPointF(p[0], p[1]) for p in points]
                    shape = PolyShape(QPolygonF(qpoints), label)
                elif shape_type == "point" and len(points) == 1:
                    shape = PointShape(QPointF(points[0][0], points[0][1]), label)
                # 旋转框 (OBB) 的解析分支
                elif shape_type == "obb":
                    rect_data = shape_data.get("rect")
                    angle = shape_data.get("angle", 0)
                    if rect_data and len(rect_data) == 4:
                        cx, cy, w, h = rect_data[0], rect_data[1], rect_data[2], rect_data[3]
                        shape = RotatedRectShape(cx, cy, w, h, angle, label)
                    else:
                        continue
                elif shape_type == "pose":
                    rect_data = shape_data.get("rect")
                    template_name = shape_data.get("template_name", "")
                    kps_data = shape_data.get("keypoints", [])
                    if rect_data and len(rect_data) == 4:
                        cx, cy, w, h = rect_data
                        angle = shape_data.get("angle", 0)
                        template = self.template_manager.get_template(template_name)
                        if not template:
                            # 降级处理，或者用空模板
                            template = {"name": template_name, "keypoints": [], "connections": []}
                            for i, kp in enumerate(kps_data):
                                template["keypoints"].append({"name": f"kp_{i}", "color": "#00FF00", "default_pos": [0.5, 0.5]})
                        
                        shape = PoseShape(QRectF(cx - w / 2, cy - h / 2, w, h), template, label)
                        shape.setRotation(angle)
                        # 设置读取的关键点位置和可见性
                        for i, kp_data in enumerate(kps_data):
                            if i < len(shape.kps):
                                local_pt = shape.mapFromScene(QPointF(kp_data[0], kp_data[1]))
                                shape.kps[i].setPos(local_pt)
                                shape.kps[i].set_visibility(kp_data[2])
                        shape.update_lines()
                    else:
                        continue
                else:
                    continue
                self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 JSON 标注失败: {e}")

    def _load_yolo(self, txt_path, img_w, img_h):
        if not os.path.exists(txt_path): return
        import math  # 局部导入数学库，用于逆向推导
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                label = self.class_list[class_id] if class_id < len(self.class_list) else str(class_id)
                shape = None

                # 1. YOLO BBox 格式 (常规矩形：5个参数)
                if len(parts) == 5:
                    cx, cy = float(parts[1]) * img_w, float(parts[2]) * img_h
                    w, h = float(parts[3]) * img_w, float(parts[4]) * img_h
                    shape = RectShape(QRectF(cx - w / 2, cy - h / 2, w, h), label)

                # YOLO OBB 旋转框格式 (9个参数：1 个类别 + 8 个坐标)
                elif len(parts) == 9:
                    x1, y1 = float(parts[1]) * img_w, float(parts[2]) * img_h
                    x2, y2 = float(parts[3]) * img_w, float(parts[4]) * img_h
                    x3, y3 = float(parts[5]) * img_w, float(parts[6]) * img_h
                    x4, y4 = float(parts[7]) * img_w, float(parts[8]) * img_h

                    # 利用四边形的顶点逆向推导出原生属性
                    cx = (x1 + x2 + x3 + x4) / 4.0
                    cy = (y1 + y2 + y3 + y4) / 4.0
                    w = math.hypot(x2 - x1, y2 - y1)
                    h = math.hypot(x4 - x1, y4 - y1)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                    # 重新生成带有完美手柄的 OBB 对象
                    shape = RotatedRectShape(cx, cy, w, h, angle, label)

                # YOLO Pose / Polygon 格式冲突处理
                elif len(parts) > 5:
                    # 尝试解析为 Pose
                    is_pose = False
                    kp_parts = parts[5:]
                    has_vis = (len(kp_parts) % 3 == 0)
                    kp_count = len(kp_parts) // 3 if has_vis else len(kp_parts) // 2
                    
                    template = self.scene.current_pose_template
                    if template and len(template.get("keypoints", [])) == kp_count:
                        is_pose = True
                    elif has_vis and len(parts) % 2 == 0: # 通常 Polygon 长度为奇数，如果有 visibility 导致长度为偶数且符合 3N，则肯定是 Pose
                        is_pose = True
                        
                    if is_pose:
                        cx, cy = float(parts[1]) * img_w, float(parts[2]) * img_h
                        w, h = float(parts[3]) * img_w, float(parts[4]) * img_h
                        if not template or len(template.get("keypoints", [])) != kp_count:
                            template = {"name": "YOLO_Import", "keypoints": [], "connections": []}
                            for i in range(kp_count):
                                template["keypoints"].append({"name": f"kp_{i}", "color": "#00FF00", "default_pos": [0.5, 0.5]})
                        
                        shape = PoseShape(QRectF(cx - w / 2, cy - h / 2, w, h), template, label)
                        idx = 0
                        for i in range(kp_count):
                            kx = float(kp_parts[idx]) * img_w
                            ky = float(kp_parts[idx+1]) * img_h
                            vis = int(float(kp_parts[idx+2])) if has_vis else 2
                            idx += 3 if has_vis else 2
                            if i < len(shape.kps):
                                local_pt = shape.mapFromScene(QPointF(kx, ky))
                                shape.kps[i].setPos(local_pt)
                                shape.kps[i].set_visibility(vis)
                        shape.update_lines()
                        self._add_shape_to_scene(shape, label)
                        continue

                    # 如果不是 Pose，且满足多边形条件
                    elif len(parts) % 2 == 1:
                        qpoints = [QPointF(float(parts[i]) * img_w, float(parts[i + 1]) * img_h) for i in
                                   range(1, len(parts), 2)]
                        shape = PolyShape(QPolygonF(qpoints), label)

                if shape is not None:
                    self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 YOLO 标注失败: {e}")

    def _load_xml(self, xml_path):
        import xml.etree.ElementTree as ET
        if not os.path.exists(xml_path): return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin, ymin = float(bndbox.find("xmin").text), float(bndbox.find("ymin").text)
                    xmax, ymax = float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)
                    shape = RectShape(QRectF(xmin, ymin, xmax - xmin, ymax - ymin), label)
                    self._add_shape_to_scene(shape, label)
        except Exception as e:
            print(f"加载 XML 标注失败: {e}")

    def save_annotation(self, format_type):
        if not self.current_image_path or not self.scene.img_item:
            QMessageBox.warning(self, "提示", "请先打开图片")
            DialogOver(self, "请先在左侧树形目录中打开图片", "操作错误", "warning")
            return
        shapes_data = Exporter.extract_shapes(self.scene)
        base_name = os.path.splitext(self.current_image_path)[0]

        if not shapes_data:
            deleted_any = False
            for ext in [".json", ".txt", ".xml"]:
                out_path = base_name + ext
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        deleted_any = True
                        print(f"手动删除空标注文件: {out_path}")
                    except Exception as e:
                        print(f"手动删除空标注文件失败: {out_path}, {e}")
            if deleted_any:
                DialogOver(self, "检测到当前无标注信息，已清除对应的空标注文件！", "提示", "warning")
            else:
                DialogOver(self, "当前界面没有标注信息，无需保存！", "提示", "warning")
            return

        img_rect = self.scene.img_item.pixmap().rect()

        try:
            if format_type == "json":
                out_path = base_name + ".json"
                Exporter.save_json(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)
            elif format_type == "yolo":
                out_path = base_name + ".txt"
                Exporter.save_yolo(out_path, img_rect.width(), img_rect.height(), shapes_data, self.class_list)
            elif format_type == "xml":
                out_path = base_name + ".xml"
                Exporter.save_xml(out_path, self.current_image_path, img_rect.width(), img_rect.height(), shapes_data)

            DialogOver(self, f"标注文件保存/更新成功！", "保存成功", "success")
            print(f"标注文件已保存到: {out_path}")
        except Exception as e:
            DialogOver(self, f"写入失败: {str(e)}", "保存出错", "danger")

    def closeEvent(self, event):
        self.auto_save_annotation()
        self.sam_client.cleanup()

        # 记录上次的状态
        settings = QSettings("luohuabuxiema", "LabelPaw")
        if self.current_dir:
            settings.setValue("last_dir", self.current_dir)
        else:
            settings.setValue("last_dir", "")
        if self.current_image_path:
            settings.setValue("last_image", self.current_image_path)
        else:
            settings.setValue("last_image", "")

        super().closeEvent(event)

    def restore_last_state(self):
        settings = QSettings("luohuabuxiema", "LabelPaw")
        last_dir = settings.value("last_dir", "")
        last_image = settings.value("last_image", "")

        if last_dir and os.path.exists(last_dir):
            self.current_dir = last_dir
            self.listFiles.clear()
            self.load_classes(last_dir)
            self.listFiles.blockSignals(True)
            target_row = -1
            for f in os.listdir(last_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(last_dir, f)
                    item = QListWidgetItem(full_path)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.listFiles.addItem(item)
                    if last_image and os.path.normpath(full_path) == os.path.normpath(last_image):
                        target_row = self.listFiles.count() - 1
            self.listFiles.blockSignals(False)
            self.update_selected_count()

            if self.listFiles.count() > 0:
                if target_row != -1:
                    self.listFiles.setCurrentRow(target_row)
                else:
                    self.listFiles.setCurrentRow(0)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # 撤销与重做快捷键（统一 Ctrl+Shift+Z 为重做）
        if key == Qt.Key_Z and modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
            self.redo()
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            self.undo()

        if key == Qt.Key_D or key == Qt.Key_Right:
            current_idx = self.listFiles.currentRow()
            if current_idx < self.listFiles.count() - 1:
                self.listFiles.setCurrentRow(current_idx + 1)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            current_idx = self.listFiles.currentRow()
            if current_idx > 0:
                self.listFiles.setCurrentRow(current_idx - 1)
        elif key == Qt.Key_Return or key == Qt.Key_Enter or key == Qt.Key_Escape:
            # 取消所有选中状态 (退出编辑模式)
            for item in self.scene.selectedItems():
                item.setSelected(False)
        elif key == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.save_annotation(self.current_format)
        elif key == Qt.Key_E:  # 快捷键 E 修改类别
            selected_items = self.scene.selectedItems()
            for item in selected_items:
                if hasattr(item, 'label'):
                    self.edit_shape_label(item)
                    break

        elif key == Qt.Key_Q:
            self.samSwitch.setChecked(not self.samSwitch.isChecked())
        elif key == Qt.Key_F1:
            self.show_help_dialog()
        elif key == Qt.Key_R:
            self.actionRect.trigger()
        elif key == Qt.Key_P:
            self.actionPoly.trigger()
        elif key == Qt.Key_M:
            # M for Model Prediction in Smart mode
            if self.samSwitch.isChecked() and getattr(self, 'current_yolo_predictor', None) is not None:
                self.btnPredict.click()
        elif key == Qt.Key_T:
            self.actionPoint.trigger()
        elif key == Qt.Key_O:
            self.actionRBox.trigger()

        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    # window.show()
    sys.exit(app.exec())
