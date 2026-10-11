"""Plain-data checkpoint inspection shared by model import and training views.

The GUI calls the selected Python worker; loading a checkpoint never runs training,
downloads weights, or changes the source file.
"""
import csv
import inspect
import io
import json
import os
import sys
from functools import lru_cache

TASK_LABELS = {
    "detect": "Detect · 目标检测", "segment": "Segment · 实例分割",
    "pose": "Pose · 关键点", "classify": "Classify · 图像分类",
    "obb": "OBB · 旋转框", "semantic": "Semantic · 语义分割",
    "depth": "Depth · 深度估计", "unknown": "其他模型",
}


def config_dict(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return dict(value) if isinstance(value, dict) else {}


def infer_task(model_name="", args=None):
    task = config_dict(args).get("task")
    if task in TASK_LABELS:
        return task
    name = str(model_name).lower()
    for suffix, task in (("-seg", "segment"), ("-pose", "pose"),
                         ("-cls", "classify"), ("-obb", "obb")):
        if suffix in name:
            return task
    return "detect" if "yolo" in name else "unknown"


def metric_keys(task, scope="auto"):
    if task == "classify" and scope == "auto":
        return ("metrics/accuracy_top1", "metrics/accuracy_top5", None)
    suffix = {"segment": "M", "pose": "P", "obb": "B"}.get(task, "B")
    if scope != "auto":
        suffix = scope
    return tuple(f"metrics/{key}({suffix})" for key in ("precision", "recall", "mAP50-95"))


def normalize_history(value):
    if isinstance(value, str):
        rows = list(csv.DictReader(io.StringIO(value)))
        value = {key: [row[key] for row in rows] for key in rows[0]} if rows else {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, values in value.items():
        if not isinstance(values, (list, tuple)):
            continue
        try:
            result[str(key).strip()] = [float(v) for v in values]
        except (ValueError, TypeError):
            continue
    lengths = [len(values) for values in result.values() if values]
    count = len(result["epoch"]) if "epoch" in result else max(lengths, default=0)
    return {key: values for key, values in result.items() if len(values) == count}


def read_checkpoint(path):
    """Executed in the ML worker, returning JSON-compatible metadata only."""
    import torch
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    ckpt = torch.load(path, **kwargs)
    if not isinstance(ckpt, dict):
        raise ValueError("该文件不是包含模型信息的 Ultralytics checkpoint")
    model = ckpt.get("ema") or ckpt.get("model")
    args = config_dict(ckpt.get("train_args"))
    task = args.get("task")
    if task not in TASK_LABELS and model is not None:
        from ultralytics.nn.tasks import guess_model_task
        task = guess_model_task(model)
    history = normalize_history(ckpt.get("train_results"))
    if model is None and not history and not args:
        raise ValueError("模型缺少可识别的网络结构或训练信息")
    return {
        "path": os.path.abspath(path), "task": task or "unknown",
        "args": args, "history": history,
        "metrics": config_dict(ckpt.get("train_metrics")),
        "class_names": getattr(model, "names", {}),
        "parameter_count": sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None,
        "date": ckpt.get("date"), "version": ckpt.get("version"),
        "license": ckpt.get("license"), "git": ckpt.get("git"),
        "runtime_seconds": history.get("time", [None])[-1] if history.get("time") else None,
        "file_size": os.path.getsize(path),
    }


def load_checkpoint_metadata(path, python_exe=None):
    path = os.path.abspath(path)
    stat = os.stat(path)
    return _inspect_cached(path, stat.st_mtime_ns, stat.st_size, python_exe or sys.executable)


@lru_cache(maxsize=32)
def _inspect_cached(path, mtime, size, python_exe):
    from TrainingPlatform.core.worker_runtime import WorkerProcess, get_workspace_root
    runner = os.path.join(os.path.dirname(__file__), "model_inspect_runner.py")
    result = None
    diagnostics = []
    with WorkerProcess(python_exe, "--model-inspect", runner, ["--weights", path], cwd=get_workspace_root()) as child:
        for line in child.iter_lines():
            if line.startswith("__MODEL_METADATA__"):
                result = json.loads(line[len("__MODEL_METADATA__"):])
            else:
                diagnostics.append(line.strip())
        if child.process.returncode or result is None:
            raise RuntimeError("读取模型失败：" + "\n".join(diagnostics[-8:]))
    result["signature"] = [mtime, size]
    return result


def checkpoint_path(task, models=()):
    explicit = task.get("weights_path")
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    folder = task.get("save_dir") or ""
    for name in ("weights/best.pt", "best.pt", "weights/last.pt", "last.pt"):
        path = os.path.join(folder, name) if folder else ""
        if path and os.path.isfile(path):
            return path
    for model in models:
        if model.get("task_id") == task.get("id") and os.path.isfile(model["best_pt_path"]):
            return model["best_pt_path"]
    return ""


def experiment_data(task, models=(), python_exe=None):
    """Completed checkpoints own their metrics/config; live CSV owns live history."""
    path = checkpoint_path(task, models)
    metadata = {}
    error = None
    if path:
        try:
            stat = os.stat(path)
            saved = config_dict(task.get("_checkpoint_metadata"))
            if not saved:
                saved = next((config_dict(m.get("metadata_json")) for m in models
                              if os.path.normcase(os.path.abspath(m["best_pt_path"])) == os.path.normcase(os.path.abspath(path))), {})
            metadata = saved if saved.get("signature") == [stat.st_mtime_ns, stat.st_size] else load_checkpoint_metadata(path, python_exe)
        except Exception as exc:
            error = str(exc)
    args = config_dict(task.get("config_json"))
    args = {**args, **metadata.get("args", {})}
    history = metadata.get("history", {})
    folder = task.get("save_dir") or ""
    csv_path = os.path.join(folder, "results.csv")
    if folder and (not history or task.get("status") in ("RUNNING", "WAITING")) and os.path.isfile(csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            history = normalize_history({key: [row[key] for row in rows] for key in rows[0]}) if rows else {}
        except (OSError, ValueError):
            pass
    if not metadata.get("args") and folder:
        args_path = os.path.join(folder, "args.yaml")
        if os.path.isfile(args_path):
            try:
                import yaml
                with open(args_path, encoding="utf-8") as source:
                    args = {**args, **config_dict(yaml.safe_load(source))}
            except Exception:
                pass
    task_type = metadata.get("task") or infer_task(task.get("model_name"), args)
    return {**metadata, "path": path, "task": task_type, "args": args, "history": history,
            "metrics": metadata.get("metrics", {}), "error": error}


def model_experiment(model):
    """Adapt a model record for the same inspector without inventing a training task."""
    path = model["best_pt_path"]
    parent = os.path.dirname(path)
    folder = os.path.dirname(parent) if os.path.basename(parent).lower() == "weights" else ""
    return {"id": -model["id"], "task_name": model["name"], "model_name": os.path.basename(path),
            "weights_path": path, "save_dir": folder, "status": "IMPORTED", "config_json": "{}",
            "project_id": model.get("project_id"), "is_imported": True,
            "_checkpoint_metadata": config_dict(model.get("metadata_json"))}


def training_command(args):
    """Build a core reproduction command, preserving saved paths and effective batch."""
    keys = ("model", "data", "task", "project", "name", "epochs", "time", "batch", "imgsz", "device",
            "workers", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "seed", "patience", "close_mosaic", "save_period",
            "pretrained", "deterministic", "amp", "cos_lr", "rect", "single_cls", "fraction", "freeze",
            "multi_scale", "compile", "nbs", "box", "cls", "dfl", "pose", "kobj", "cls_remap",
            "mask_ratio", "overlap_mask", "dropout", "hsv_h", "hsv_s", "hsv_v", "degrees",
            "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic",
            "mixup", "cutmix", "copy_paste", "copy_paste_mode", "max_det", "iou", "val", "plots", "exist_ok")
    parts = ["yolo train"]
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = '"' + value.replace('"', '\\"') + '"'
        elif isinstance(value, bool):
            value = str(value).lower()
        parts.append(f"{key}={value}")
    return " ".join(parts) if len(parts) > 1 else "暂无训练配置"
