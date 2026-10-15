# -*- coding: utf-8 -*-
"""
Standalone YOLO Training Runner Script
Executed as a subprocess with a specific Python interpreter (Conda, venv, or built-in).
Ensures isolation, stability, and zero GUI freezing or crashing.
"""
import sys
import os
import json
import argparse
import traceback

if __package__:
    from .worker_runtime import prepare_runner_paths
else:
    from worker_runtime import prepare_runner_paths


def main(argv=None):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="YOLO Training Subprocess Runner")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args(argv)

    # Ignore harmless dependency warnings from requests/urllib3 in python environment
    import warnings
    warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="requests")

    # Ensure virtual environment packages (like ultralytics) take precedence over workspace root
    workspace = prepare_runner_paths(__file__)

    print(f"[Runner] 启动 YOLO 训练执行器 (Task ID: {args.task_id})", flush=True)
    print(f"[Runner] Python 解释器: {sys.executable}", flush=True)
    print(f"[Runner] 工作空间根目录: {workspace}", flush=True)

    # Read config JSON
    config_dict = {}
    if os.path.isfile(args.config_file):
        try:
            with open(args.config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except Exception as e:
            print(f"[WARN] [Runner] 读取配置失败: {e}", flush=True)

    # Import ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        version_str = getattr(ultralytics, '__version__', 'unknown')
        file_str = getattr(ultralytics, '__file__', 'unknown')
        print(f"[Runner] Ultralytics 已就绪 (v{version_str}): {file_str}", flush=True)
    except Exception as e:
        print(f"[ERROR] [Runner] 无法导入 ultralytics: {e}", flush=True)
        sys.exit(1)

    try:
        model = YOLO(args.model)

        def on_train_epoch_end(trainer):
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
                        metrics_data["precision"] = float(rd.get('metrics/precision(B)') or rd.get('metrics/precision(M)') or 0.0)
                        metrics_data["recall"] = float(rd.get('metrics/recall(B)') or rd.get('metrics/recall(M)') or 0.0)
                        metrics_data["map50"] = float(rd.get('metrics/mAP50(B)') or rd.get('metrics/mAP50(M)') or rd.get('metrics/mAP50(P)') or rd.get('metrics/mAP50') or 0.0)
                        metrics_data["map50_95"] = float(rd.get('metrics/mAP50-95(B)') or rd.get('metrics/mAP50-95(M)') or rd.get('metrics/mAP50-95(P)') or rd.get('metrics/mAP50-95') or 0.0)
                except Exception:
                    pass

            print(f"__TRAIN_PROGRESS__{json.dumps(metrics_data)}", flush=True)

        def on_fit_epoch_end(trainer):
            # Ultralytics populates validation metrics at fit epoch end
            try:
                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                m = getattr(trainer, 'metrics', None) or getattr(getattr(trainer, 'validator', None), 'metrics', None)
                if m and hasattr(m, 'results_dict'):
                    rd = m.results_dict
                    p = float(rd.get('metrics/precision(B)') or rd.get('metrics/precision(M)') or 0.0)
                    r = float(rd.get('metrics/recall(B)') or rd.get('metrics/recall(M)') or 0.0)
                    m50 = float(rd.get('metrics/mAP50(B)') or rd.get('metrics/mAP50(M)') or rd.get('metrics/mAP50(P)') or rd.get('metrics/mAP50') or 0.0)
                    m95 = float(rd.get('metrics/mAP50-95(B)') or rd.get('metrics/mAP50-95(M)') or rd.get('metrics/mAP50-95(P)') or rd.get('metrics/mAP50-95') or 0.0)
                    if m50 > 0.0 or m95 > 0.0:
                        metrics_data = {
                            "epoch": epoch,
                            "total_epochs": total_epochs,
                            "progress": round((epoch / total_epochs) * 100, 2),
                            "precision": p,
                            "recall": r,
                            "map50": m50,
                            "map50_95": m95
                        }
                        print(f"__TRAIN_PROGRESS__{json.dumps(metrics_data)}", flush=True)
            except Exception:
                pass

        def on_pretrain_routine_end(trainer):
            if hasattr(trainer, 'save_dir') and trainer.save_dir:
                print(f"__SAVE_DIR__{trainer.save_dir}", flush=True)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        # Resolve dataset yaml: if relative, check current dir, workspace or save_dir
        data_arg = args.data
        if not os.path.isabs(data_arg):
            if os.path.isfile(data_arg):
                data_arg = os.path.abspath(data_arg)
            elif os.path.isfile(os.path.join(workspace, data_arg)):
                data_arg = os.path.join(workspace, data_arg)
            elif os.path.isfile(os.path.join(args.save_dir, data_arg)):
                data_arg = os.path.join(args.save_dir, data_arg)

        train_args = {
            "data": data_arg,
            "project": os.path.dirname(args.save_dir),
            "name": os.path.basename(args.save_dir),
            "exist_ok": True,
            "plots": True,
            "verbose": True
        }

        for k, v in config_dict.items():
            if v is not None and k not in ("project", "name", "exist_ok", "plots", "verbose", "data", "python_exe", "cli_command", "task_id", "task_name", "model_name", "cls_remap"):
                train_args[k] = v

        device = train_args.get("device")
        multiple_devices = (
            isinstance(device, (list, tuple)) and len(device) > 1
        ) or (isinstance(device, str) and "," in device)
        if getattr(sys, "frozen", False) and multiple_devices:
            raise ValueError("内置运行环境支持 CPU 或单 GPU；多 GPU 分布式训练请选用外部 Python 环境")

        print("[Runner] 正在执行 model.train(...) ...", flush=True)
        results = model.train(**train_args)
        print("\n__TRAIN_SUCCESS__", flush=True)
        print("[SUCCESS] [Runner] YOLO 模型训练成功完成！", flush=True)
        sys.exit(0)

    except Exception as e:
        err = traceback.format_exc()
        print(f"[ERROR] [Runner] 训练异常:\n{err}", flush=True)
        print(f"\n__TRAIN_FAILED__{str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
