# -*- coding: utf-8 -*-
"""
Task Manager for AI Vision Training Platform
Manages active worker threads, task queue, and DB updates.
"""
import os
from PySide6.QtCore import QObject, Signal

class TaskManager(QObject):
    task_started = Signal(int)          # task_id
    task_log = Signal(int, str)         # task_id, log_text
    task_progress = Signal(int, dict)   # task_id, metrics_dict
    task_finished = Signal(int, bool, str) # task_id, success, message

    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.active_workers = {}  # task_id -> TrainWorker instance

    def start_task(self, task_id):
        if task_id in self.active_workers:
            return False, "该任务已经在运行中！"

        task = self.db.get_task_by_id(task_id)
        if not task:
            return False, "未找到指定的任务记录！"

        dataset = self.db.get_dataset_by_id(task["dataset_id"])
        if not dataset:
            return False, "未关联有效的数据集！"

        # Check dataset yaml path
        dataset_path = dataset["path"]
        if dataset_path.lower().endswith((".yaml", ".yml")) and os.path.exists(dataset_path):
            dataset_yaml = dataset_path
        else:
            dataset_yaml = os.path.join(dataset_path, "data.yaml")
            if not os.path.exists(dataset_yaml):
                # Check recursive data.yaml
                from TrainingPlatform.core.dataset_inspector import DatasetInspector
                info = DatasetInspector.inspect(dataset_path)
                dataset_yaml = info.get("yaml_path", "")

        if not os.path.exists(dataset_yaml):
            return False, f"未能在数据集目录中找到 data.yaml 文件: {dataset_path}"

        import json
        config_dict = json.loads(task["config_json"]) if task["config_json"] else {}
        
        # Save output runs dir under project root: runs/<task_type>/<task_name> (standard Ultralytics structure)
        from TrainingPlatform.core.env_manager import get_workspace_root
        workspace = get_workspace_root()

        from TrainingPlatform.core.checkpoint_metadata import infer_task
        task_type = infer_task(task["model_name"], config_dict)
        if task_type == "unknown":
            return False, "无法确定模型任务类型，请在训练配置中选择任务类型"
        task_name = task.get("task_name") or f"exp-{task_id}"

        save_dir = os.path.join(workspace, "runs", task_type, task_name)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        python_exe = config_dict.get("python_exe")
        from TrainingPlatform.core.train_worker import TrainWorker
        worker = TrainWorker(
            task_id=task_id,
            model_name=task["model_name"],
            dataset_yaml=dataset_yaml,
            config_dict=config_dict,
            save_dir=save_dir,
            python_exe=python_exe
        )

        # Wire signals
        worker.log_signal.connect(lambda msg: self._on_log(task_id, msg))
        worker.progress_signal.connect(lambda metrics: self._on_progress(task_id, metrics))
        worker.finish_signal.connect(lambda success, msg: self._on_finish(task_id, success, msg))

        self.active_workers[task_id] = worker
        self.db.update_task_status(task_id, "RUNNING", save_dir=save_dir)
        worker.start()

        self.task_started.emit(task_id)
        return True, "任务成功启动！"

    def stop_task(self, task_id):
        if task_id in self.active_workers:
            worker = self.active_workers[task_id]
            worker.stop()
            self.db.update_task_status(task_id, "STOPPED")
            return True, "已发送停止请求"
        else:
            self.db.update_task_status(task_id, "STOPPED")
            return True, "任务状态已更新为停止"

    def is_running(self, task_id):
        return task_id in self.active_workers and self.active_workers[task_id].isRunning()

    def _on_log(self, task_id, msg):
        self.task_log.emit(task_id, msg)

    def _on_progress(self, task_id, metrics):
        # Record metrics in DB
        self.db.record_metrics(task_id, metrics.get("epoch", 0), metrics)
        self.db.update_task_status(
            task_id, "RUNNING",
            current_epoch=metrics.get("epoch", 0),
            progress=metrics.get("progress", 0.0),
            best_map50=metrics.get("map50", 0.0),
            best_map50_95=metrics.get("map50_95", 0.0)
        )
        self.task_progress.emit(task_id, metrics)

    def _on_finish(self, task_id, success, msg):
        import csv
        status = "SUCCESS" if success else "FAILED"
        task = self.db.get_task_by_id(task_id)
        save_dir = task.get("save_dir", "") if task else ""

        best_map50 = task.get("best_map50", 0.0) if task else 0.0
        best_map50_95 = task.get("best_map50_95", 0.0) if task else 0.0

        # Parse final validation metrics (all-classes mean for best.pt from train.log or results.csv)
        if save_dir and os.path.exists(save_dir):
            try:
                from TrainingPlatform.core.metrics_parser import parse_final_experiment_metrics
                parsed_metrics = parse_final_experiment_metrics(save_dir)
                if parsed_metrics.get("map50", 0.0) > 0.0:
                    best_map50 = parsed_metrics["map50"]
                    best_map50_95 = parsed_metrics["map50_95"]
            except Exception as e:
                print(f"[TaskManager] Error parsing final metrics: {e}")

        self.db.update_task_status(task_id, status, best_map50=best_map50, best_map50_95=best_map50_95)

        # If success, add model record automatically
        if success and save_dir:
            best_pt = os.path.join(save_dir, "weights", "best.pt")
            last_pt = os.path.join(save_dir, "weights", "last.pt")
            if not os.path.exists(best_pt):
                best_pt = os.path.join(save_dir, "best.pt")
                last_pt = os.path.join(save_dir, "last.pt")

            if os.path.exists(best_pt) and task:
                model_name = f"{task['task_name']}_best"
                self.db.add_model(
                    name=model_name,
                    task_id=task_id,
                    project_id=task["project_id"],
                    model_arch=task["model_name"],
                    best_pt_path=best_pt,
                    last_pt_path=last_pt if os.path.exists(last_pt) else None,
                    metrics={"mAP50": round(best_map50, 4), "mAP50-95": round(best_map50_95, 4)}
                )

        if task_id in self.active_workers:
            del self.active_workers[task_id]

        self.task_finished.emit(task_id, success, msg)
