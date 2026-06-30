# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import queue
from PySide6.QtCore import QObject, QThread, Signal
from PIL import Image

import os

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    pass

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============== SAM Model Map ==============
import sys

# 设置一个更加通用的模型权重基准目录
# 优先从当前项目根目录下的 weights 文件夹寻找，如果没有则回退到硬编码路径
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
HARDCODED_DEV_DIR = r"E:\11-AI\标注工具\weights"

if os.path.exists(LOCAL_WEIGHTS_DIR):
    MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR
elif os.path.exists(HARDCODED_DEV_DIR):
    MODEL_BASE_DIR = HARDCODED_DEV_DIR
else:
    MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR  # 默认指向项目下的 weights

def get_sam_model_map():
    sam_weights_dir = os.path.join(MODEL_BASE_DIR, "sam_weights")
    model_map = {}
    
    if not os.path.exists(sam_weights_dir):
        return model_map
        
    for f in os.listdir(sam_weights_dir):
        if not f.endswith(".pt"):
            continue
            
        path = os.path.join(sam_weights_dir, f)
        key = f.replace(".pt", "")
        size_mb = os.path.getsize(path) // (1024 * 1024)
        size_label = f"{size_mb} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
        
        # Determine type and config based on filename
        if "sam3" in f.lower():
            model_map[key] = {
                "display_name": "SAM 3" if f.lower() == "sam3.pt" else f"SAM 3 ({key})",
                "type": "sam3",
                "weight": path,
                "config": None,
                "supports_text": True,
                "size_label": size_label,
            }
        else:
            # SAM 2 logic
            config = None
            f_lower = f.lower()
            if "tiny" in f_lower or "_t.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in f_lower or "_s.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "base_plus" in f_lower or "_b+.pt" in f_lower or "base" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in f_lower or "_l.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_l.yaml"
                
            model_map[key] = {
                "display_name": key.replace("_", " ").title(),
                "type": "sam2",
                "weight": path,
                "config": config,
                "supports_text": False,
                "size_label": size_label,
            }
    return model_map

SAM_MODEL_MAP = get_sam_model_map()

# ============== SAM 3 Workers ==============
class Sam3ModelLoadWorker(QThread):
    loaded = Signal(object, object, bool, str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            if torch.cuda.is_available():
                model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint_path,
                    enable_inst_interactivity=True,
                    device="cpu",
                )
                model.half()
                model.to("cuda")
            else:
                model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint_path,
                    enable_inst_interactivity=True,
                    device="cpu",
                )
            processor = Sam3Processor(model)
            self.loaded.emit(model, processor, True, "模型加载成功")
        except Exception as e:
            self.loaded.emit(None, None, False, str(e))


class Sam3InferenceWorker(QThread):
    result_ready = Signal(list, list, list, float, bool)
    text_result_ready = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.inference_state = None
        self.task_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                task_type, data, is_click = self.task_queue.get(timeout=0.05)

                if not self.model or not self.inference_state:
                    continue

                if task_type == 'point':
                    x, y = data
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        masks, scores, _ = self.model.predict_inst(
                            inference_state=self.inference_state,
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=is_click
                        )

                    if len(scores) > 0:
                        best_idx = np.argmax(scores)
                        mask_np = masks[best_idx].cpu().numpy() if torch.is_tensor(masks) else masks[best_idx]
                        score_val = float(scores[best_idx].cpu() if torch.is_tensor(scores) else scores[best_idx])

                        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        poly_pts = []
                        rect_xywh = []
                        rect_obb = []
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                            poly_pts = approx.reshape(-1, 2).tolist()

                            x_r, y_r, w_r, h_r = cv2.boundingRect(largest_contour)
                            rect_xywh = [x_r, y_r, w_r, h_r]

                            obb = cv2.minAreaRect(largest_contour)
                            rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

                        self.result_ready.emit(poly_pts, rect_xywh, rect_obb, score_val, is_click)

                elif task_type == 'text':
                    prompts = data
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    
                    if not self.processor:
                        continue

                    for prompt_text in prompts:
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                            out_state = self.processor.set_text_prompt(prompt=prompt_text, state=self.inference_state)

                            masks = out_state.get("masks", [])
                            scores = out_state.get("scores", [])
                            boxes = out_state.get("boxes", [])

                            results = []
                            if len(masks) > 0:
                                for i in range(len(masks)):
                                    mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                                    mask_np = np.squeeze(mask_np)

                                    score_val = float(scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i])
                                    box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]

                                    if box.ndim > 1:
                                        box = box.squeeze()
                                    x1, y1, x2, y2 = box
                                    rect_xywh = [x1, y1, x2 - x1, y2 - y1]

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
                                        rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

                                    if poly_pts:
                                        results.append({
                                            "poly_pts": poly_pts,
                                            "rect": rect_xywh,
                                            "obb": rect_obb,
                                            "score": score_val
                                        })

                            self.text_result_ready.emit(results, prompt_text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"SAM3 推理错误: {e}")

    def request_inference(self, x, y, is_click=False):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(('point', (x, y), is_click))

    def request_text_inference(self, prompt_text):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(('text', prompt_text, True))

    def stop(self):
        self.running = False
        self.wait()


# ============== SAM 2.1 Workers ==============
class Sam2ModelLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, config_file, checkpoint_path):
        super().__init__()
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            model = build_sam2(self.config_file, ckpt_path=self.checkpoint_path, device="cuda")
            predictor = SAM2ImagePredictor(model)
            self.loaded.emit(predictor, True, "SAM 2.1 模型加载成功")
        except Exception as e:
            self.loaded.emit(None, False, str(e))


class Sam2InferenceWorker(QThread):
    """SAM 2.1 推理 Worker — 仅支持点击分割，不支持文本提示词"""
    result_ready = Signal(list, list, list, float, bool)

    def __init__(self):
        super().__init__()
        self.predictor = None
        self.image_set = False
        self.task_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                task_type, data, is_click = self.task_queue.get(timeout=0.05)

                if not self.predictor or not self.image_set:
                    continue

                if task_type == 'point':
                    x, y = data
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        masks, scores, _ = self.predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=is_click,
                        )

                    if len(scores) > 0:
                        best_idx = np.argmax(scores)
                        mask_np = masks[best_idx]
                        score_val = float(scores[best_idx])

                        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        poly_pts = []
                        rect_xywh = []
                        rect_obb = []
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                            poly_pts = approx.reshape(-1, 2).tolist()

                            x_r, y_r, w_r, h_r = cv2.boundingRect(largest_contour)
                            rect_xywh = [x_r, y_r, w_r, h_r]

                            obb = cv2.minAreaRect(largest_contour)
                            rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

                        self.result_ready.emit(poly_pts, rect_xywh, rect_obb, score_val, is_click)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"SAM2 推理错误: {e}")

    def request_inference(self, x, y, is_click=False):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(('point', (x, y), is_click))

    def stop(self):
        self.running = False
        self.wait()


# ============== Unified SAM Client ==============
class SAMClient(QObject):
    model_status_changed = Signal(bool, str)
    inference_result = Signal(list, list, list, float, bool)
    text_result_ready = Signal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.processor = None
        self.current_model_key = None   # 当前模型的 key
        self.current_model_type = None  # "sam3" / "sam2"

        # SAM3 worker
        self._sam3_worker = None
        # SAM2 worker
        self._sam2_worker = None
        # 当前活跃的推理 worker
        self._active_worker = None

        self.load_worker = None

    def load_model_by_key(self, model_key):
        """通过 key 从 SAM_MODEL_MAP 中选择并加载模型"""
        if model_key not in SAM_MODEL_MAP:
            self.model_status_changed.emit(False, f"未知模型: {model_key}")
            return
        
        # 如果已经是当前模型，不重复加载
        if model_key == self.current_model_key and self._active_worker is not None:
            self.model_status_changed.emit(True, f"{SAM_MODEL_MAP[model_key]['display_name']} 已就绪")
            return

        info = SAM_MODEL_MAP[model_key]
        model_type = info.get("type", "sam2")
        weight_path = info.get("weight")

        # 如果权重文件不存在，提示用户并不继续加载
        if not weight_path or not os.path.exists(weight_path):
            self.model_status_changed.emit(False, f"模型权重文件不存在或未下载")
            return

        # 先停止旧的推理 worker
        self._stop_active_worker()

        self.current_model_key = model_key
        self.current_model_type = model_type
        self.model_status_changed.emit(False, f"正在加载 {info.get('display_name', model_key)}，请稍候...")

        if model_type == "sam3":
            self.load_worker = Sam3ModelLoadWorker(weight_path)
            self.load_worker.loaded.connect(self._on_sam3_loaded)
            self.load_worker.start()
        elif model_type == "sam2":
            config = info.get("config")
            self.load_worker = Sam2ModelLoadWorker(config, weight_path)
            self.load_worker.loaded.connect(self._on_sam2_loaded)
            self.load_worker.start()

    # 兼容旧接口
    def load_model_async(self, checkpoint_path):
        """向后兼容：直接加载 SAM3 模型"""
        self.current_model_key = "sam3"
        self.current_model_type = "sam3"
        self._stop_active_worker()
        self.model_status_changed.emit(False, "正在后台加载模型，请稍候...")
        self.load_worker = Sam3ModelLoadWorker(checkpoint_path)
        self.load_worker.loaded.connect(self._on_sam3_loaded)
        self.load_worker.start()

    def _stop_active_worker(self):
        """停止当前的推理 worker"""
        if self._sam3_worker:
            self._sam3_worker.stop()
            self._sam3_worker = None
        if self._sam2_worker:
            self._sam2_worker.stop()
            self._sam2_worker = None
        self._active_worker = None

    def _on_sam3_loaded(self, model, processor, success, msg):
        if success:
            self.model = model
            self.processor = processor
            # 启动 SAM3 推理线程
            self._sam3_worker = Sam3InferenceWorker()
            self._sam3_worker.model = model
            self._sam3_worker.processor = processor
            self._sam3_worker.result_ready.connect(self.inference_result)
            self._sam3_worker.text_result_ready.connect(self.text_result_ready)
            self._sam3_worker.start()
            self._active_worker = self._sam3_worker
        self.model_status_changed.emit(success, msg)

    def _on_sam2_loaded(self, predictor, success, msg):
        if success:
            self.model = predictor
            self.processor = None  # SAM2 没有文本处理器
            # 启动 SAM2 推理线程
            self._sam2_worker = Sam2InferenceWorker()
            self._sam2_worker.predictor = predictor
            self._sam2_worker.result_ready.connect(self.inference_result)
            self._sam2_worker.start()
            self._active_worker = self._sam2_worker
        self.model_status_changed.emit(success, msg)

    def set_image(self, image_path):
        """设置当前图像（自动适配 SAM3 / SAM2）"""
        if self.current_model_type == "sam3":
            if not self.processor:
                return
            try:
                pil_img = Image.open(image_path).convert("RGB")
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    state = self.processor.set_image(pil_img)
                    if self._sam3_worker:
                        self._sam3_worker.inference_state = state
            except Exception as e:
                print(f"SAM3 图像特征提取失败: {e}")

        elif self.current_model_type == "sam2":
            if not self.model:
                return
            try:
                pil_img = Image.open(image_path).convert("RGB")
                img_np = np.array(pil_img)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    self.model.set_image(img_np)
                    if self._sam2_worker:
                        self._sam2_worker.image_set = True
            except Exception as e:
                print(f"SAM2 图像特征提取失败: {e}")

    def request_inference(self, x, y, is_click):
        if self._active_worker:
            self._active_worker.request_inference(x, y, is_click)

    def request_text_inference(self, prompt_text):
        """文本提示词推理（仅 SAM3 支持）"""
        if self.current_model_type == "sam3" and self._sam3_worker:
            import re
            prompts = [p.strip() for p in re.split(r'[,，、\s]+', prompt_text) if p.strip()]
            if prompts:
                self._sam3_worker.request_text_inference(prompts)

    def supports_text_prompt(self):
        """当前模型是否支持文本提示词"""
        if self.current_model_key and self.current_model_key in SAM_MODEL_MAP:
            return SAM_MODEL_MAP[self.current_model_key]["supports_text"]
        return False

    def cleanup(self):
        self._stop_active_worker()
        
        # 卸载模型以释放显存
        self.model = None
        self.processor = None
        self.predictor = None
        
        # 强制垃圾回收和显存清理
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
