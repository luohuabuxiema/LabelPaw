# -*- coding: utf-8 -*-
"""
YOLO Training Worker Thread using PySide6 QThread & Ultralytics API
"""
import sys
import os
import re
import json
import traceback
from PySide6.QtCore import QThread, Signal
from TrainingPlatform.core.worker_runtime import (
    WorkerProcess, get_workspace_root, normalize_python, terminate_process_tree,
)

class TrainWorker(QThread):
    log_signal = Signal(str)          # 实时日志文本
    progress_signal = Signal(dict)   # 包含 epoch, losses, mAP50 等指标字典
    finish_signal = Signal(bool, str) # 训练成功/失败标志, 消息

    def __init__(self, task_id, model_name, dataset_yaml, config_dict, save_dir, python_exe=None):
        super().__init__()
        self.task_id = task_id
        self.model_name = model_name
        self.dataset_yaml = dataset_yaml
        self.config_dict = config_dict
        self.save_dir = save_dir
        self.python_exe = normalize_python(python_exe)
        self._is_stopped = False
        self._process = None
        self._log_file_path = None
        self._pending_log_tag = None
        self._pending_log_line = None
        if self.save_dir:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                self._log_file_path = os.path.join(self.save_dir, "train.log")
            except Exception:
                pass

    def _flush_pending_log(self):
        if getattr(self, "_pending_log_line", None) and self._log_file_path:
            try:
                with open(self._log_file_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(f"{self._pending_log_line}\n")
            except Exception:
                pass
        self._pending_log_tag = None
        self._pending_log_line = None

    def _emit_log(self, msg: str):
        """Emit log line to GUI and append to task train.log."""
        self.log_signal.emit(msg)
        if self._log_file_path:
            m_epoch = re.match(r'^\s*(\d+/\d+)\s+', msg)
            is_val = ('Class' in msg and 'Images' in msg and 'Instances' in msg and 'Box(P' in msg)
            tag = m_epoch.group(1) if m_epoch else ("VAL_PROGRESS" if is_val else None)
            if tag:
                if self._pending_log_tag == tag:
                    self._pending_log_line = msg
                    return
                self._flush_pending_log()
                self._pending_log_tag = tag
                self._pending_log_line = msg
                return

            self._flush_pending_log()
            try:
                with open(self._log_file_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(f"{msg}\n")
            except Exception:
                pass

    def stop(self):
        self._is_stopped = True
        self._emit_log("[WARN] 收到用户停止指令，正在停止训练...")
        if self._process and self._process.poll() is None:
            try:
                terminate_process_tree(self._process)
            except Exception:
                pass

    def run(self):
        try:
            self._emit_log(f"[INFO] 开始启动 YOLO 训练任务 [Task ID: {self.task_id}]")
            self._emit_log(f"[INFO] 模型架构/预训练: {self.model_name}")
            self._emit_log(f"[INFO] 数据集配置: {self.dataset_yaml}")
            self._emit_log(f"[INFO] 存放目录: {self.save_dir}")
            if self.python_exe:
                self._emit_log(f"[INFO] 目标运行环境: {self.python_exe}")
            self._emit_log("-" * 60)

            if self._is_stopped:
                raise StopIteration("训练已被终止")
            # Every environment uses a child process; invalid paths must not silently
            # switch to training inside the GUI process.
            self._run_via_subprocess()

        except StopIteration as e:
            self._emit_log(f"[STOP] {str(e)}")
            self.finish_signal.emit(False, "训练已被终止")
        except Exception as e:
            err_msg = traceback.format_exc()
            self._emit_log(f"[ERROR] 训练异常失败:\n{err_msg}")
            self.finish_signal.emit(False, str(e))

    def _run_via_subprocess(self):
        """Execute training using the selected python executable in an isolated subprocess."""
        os.makedirs(self.save_dir, exist_ok=True)
        config_file = os.path.join(self.save_dir, "train_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=2)

        runner_script = os.path.join(os.path.dirname(__file__), "train_runner.py")
        workspace = get_workspace_root()
        runner_args = [
            "--task-id", str(self.task_id),
            "--model", str(self.model_name),
            "--data", str(self.dataset_yaml),
            "--save-dir", str(self.save_dir),
            "--config-file", str(config_file)
        ]

        self._emit_log(f"[INFO] 正在启动训练子进程...")
        child = WorkerProcess(
            self.python_exe, "--train-worker", runner_script, runner_args, cwd=workspace,
        )
        self._process = child.process
        if self._is_stopped:
            terminate_process_tree(self._process)

        success_state = {"success": False, "reason": None}
        try:
            self._consume_training_output(child, success_state)
        finally:
            child.close()

        success = success_state["success"]
        fail_reason = success_state["reason"]
        self._flush_pending_log()

        # Fallback verification: Check if best.pt or last.pt was generated if process exited with 0
        if not success and not fail_reason and self._process and self._process.returncode == 0:
            candidate_weights = [
                os.path.join(self.save_dir, "weights", "best.pt"),
                os.path.join(self.save_dir, "best.pt"),
                os.path.join(self.save_dir, "weights", "last.pt"),
                os.path.join(self.save_dir, "last.pt"),
            ]
            for w in candidate_weights:
                if os.path.exists(w) and os.path.getsize(w) > 0:
                    self._emit_log(f"[INFO] 验证检测到权重文件已成功生成 ({os.path.basename(w)})，确认训练完成")
                    success = True
                    break

        if self._is_stopped:
            self._emit_log("[WARN] 训练已被用户中途终止")
            self.finish_signal.emit(False, "训练已被用户中途终止")
        elif success and not fail_reason and self._process.returncode == 0:
            self.finish_signal.emit(True, "训练完成")
        else:
            err_msg = fail_reason or (
                "训练进程未返回完成标记" if self._process.returncode == 0
                else f"训练子进程退出码: {self._process.returncode}"
            )
            self._emit_log(f"[ERROR] 训练失败: {err_msg}")
            self.finish_signal.emit(False, err_msg)

    def _consume_training_output(self, child, state):
        for line in child.iter_lines():
            # tqdm dynamic in-place updates may deliver lines separated by \r
            chunks = line.split('\r') if '\r' in line else [line]
            last_display_chunk = None
            for chunk in chunks:
                raw_line = chunk.rstrip('\r\n')
                # Strip ANSI escape codes and whitespace for protocol checks
                clean_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_line).strip()
                if not clean_line and not raw_line:
                    continue

                if clean_line.startswith("__TRAIN_PROGRESS__"):
                    try:
                        data = json.loads(clean_line[len("__TRAIN_PROGRESS__"):].strip())
                        self.progress_signal.emit(data)
                    except Exception:
                        pass
                elif clean_line.startswith("__SAVE_DIR__"):
                    actual_save_dir = clean_line[len("__SAVE_DIR__"):].strip()
                    if actual_save_dir:
                        self.save_dir = actual_save_dir
                        try:
                            from TrainingPlatform.db.database import DatabaseManager
                            db = DatabaseManager()
                            db.update_task_status(self.task_id, "RUNNING", save_dir=actual_save_dir)
                        except Exception:
                            pass
                elif clean_line == "__TRAIN_SUCCESS__" or "__TRAIN_SUCCESS__" in clean_line:
                    state["success"] = True
                elif clean_line.startswith("__TRAIN_FAILED__"):
                    state["reason"] = clean_line[len("__TRAIN_FAILED__"):].strip()
                else:
                    if "[SUCCESS] [Runner] YOLO 模型训练成功完成" in clean_line:
                        state["success"] = True
                    # Strip ANSI escape codes for clean terminal logging
                    filtered_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_line).rstrip()
                    last_display_chunk = filtered_line

            if last_display_chunk is not None:
                self._emit_log(last_display_chunk)

    def _run_in_process(self):
        """Fallback in-process training."""
        # Ensure ultralytics is importable
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from ultralytics import YOLO

        model = YOLO(self.model_name)

        def on_train_epoch_end(trainer):
            if self._is_stopped:
                raise StopIteration("用户暂停/强行终止训练任务")

            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            metrics_data = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "progress": round((epoch / total_epochs) * 100, 2),
                "train_box_loss": 0.0,
                "train_cls_loss": 0.0,
                "train_dfl_loss": 0.0,
                "val_box_loss": 0.0,
                "val_cls_loss": 0.0,
                "val_dfl_loss": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0
            }

            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    loss_items = trainer.loss_items
                    if len(loss_items) >= 3:
                        metrics_data["train_box_loss"] = float(loss_items[0])
                        metrics_data["train_cls_loss"] = float(loss_items[1])
                        metrics_data["train_dfl_loss"] = float(loss_items[2])
                except Exception:
                    pass

            if hasattr(trainer, 'metrics') and trainer.metrics:
                try:
                    m = trainer.metrics
                    if hasattr(m, 'results_dict'):
                        rd = m.results_dict
                        metrics_data["precision"] = float(rd.get('metrics/precision(B)', 0.0))
                        metrics_data["recall"] = float(rd.get('metrics/recall(B)', 0.0))
                        metrics_data["map50"] = float(rd.get('metrics/mAP50(B)', 0.0))
                        metrics_data["map50_95"] = float(rd.get('metrics/mAP50-95(B)', 0.0))
                except Exception:
                    pass

            log_str = (f"[Epoch {epoch}/{total_epochs}] "
                       f"Box Loss: {metrics_data['train_box_loss']:.4f} | "
                       f"Cls Loss: {metrics_data['train_cls_loss']:.4f} | "
                       f"mAP50: {metrics_data['map50']:.4f}")
            self._emit_log(log_str)
            self.progress_signal.emit(metrics_data)

        train_args = {
            "data": self.dataset_yaml,
            "project": os.path.dirname(self.save_dir),
            "name": os.path.basename(self.save_dir),
            "exist_ok": True,
            "plots": True,
            "verbose": True
        }

        for k, v in self.config_dict.items():
            if v is not None and k not in ("project", "name", "exist_ok", "plots", "verbose", "data", "python_exe", "cli_command", "task_id", "task_name", "model_name", "cls_remap"):
                train_args[k] = v

        self._emit_log("[INFO] 正在投入模型训练，请稍候...")
        results = model.train(**train_args)

        if self._is_stopped:
            self._emit_log("[WARN] 训练已被用户中途终止")
            self.finish_signal.emit(False, "训练已被用户中途终止")
        else:
            self.finish_signal.emit(True, "训练完成")

