# -*- coding: utf-8 -*-
"""
Standalone YOLO Model Export Runner Script
Executed as an isolated subprocess to guarantee zero GUI freezing, zero crashes,
and complete protection against C++ runtime / OpenMP / segmentation faults (0xC0000005).
"""
import sys
import os
import json
import time
import argparse
import traceback
from contextlib import contextmanager

if __package__:
    from .worker_runtime import prepare_runner_paths
else:
    from worker_runtime import prepare_runner_paths


@contextmanager
def _torchscript_save_compat(export_format):
    """Torch 2.5's native Windows writer can reject existing non-ASCII paths."""
    if os.name != "nt" or export_format != "torchscript":
        yield
        return
    import torch
    original_save = torch.jit.ScriptModule.save

    def save(module, path, **kwargs):
        if isinstance(path, (str, os.PathLike)) and not os.fsdecode(path).isascii():
            # Python opens the Unicode path; Torch writes the serialized archive to it.
            with open(path, "wb") as output:
                return torch.jit.save(module, output, **kwargs)
        return original_save(module, path, **kwargs)

    torch.jit.ScriptModule.save = save
    try:
        yield
    finally:
        torch.jit.ScriptModule.save = original_save


def main(argv=None):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="YOLO Model Export Subprocess Runner")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON file containing export params")
    args = parser.parse_args(argv)

    # Prevent OpenMP runtime library conflict on Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Suppress harmless warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Ensure virtual environment packages take precedence
    workspace = prepare_runner_paths(__file__)

    print(f"[ExportRunner] 启动模型导出隔离进程", flush=True)
    print(f"[ExportRunner] Python 解释器: {sys.executable}", flush=True)
    print(f"[ExportRunner] 模型权重: {args.weights}", flush=True)

    if not os.path.isfile(args.weights):
        err_res = {"success": False, "error": f"模型权重文件不存在: {args.weights}"}
        print(f"__EXPORT_RESULT__:{json.dumps(err_res, ensure_ascii=False)}", flush=True)
        sys.exit(1)

    params = {}
    if os.path.isfile(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception as e:
            print(f"[ExportRunner] 读取配置失败: {e}", flush=True)

    t0 = time.time()
    try:
        from ultralytics import YOLO
        model = YOLO(args.weights)

        fmt = params.get("format", "onnx").lower()
        export_kwargs = {"format": fmt, "verbose": True}

        # Map supported arguments
        for k in ("imgsz", "batch", "half", "int8", "dynamic", "simplify", "opset", "nms", "workspace", "device"):
            if k in params and params[k] is not None:
                export_kwargs[k] = params[k]

        if fmt == "rknn" and "rknn_target" in params:
            export_kwargs["name"] = params["rknn_target"]

        if fmt == "hailo":
            export_kwargs["format"] = "onnx"
            export_kwargs["simplify"] = True
            export_kwargs["opset"] = 17

        try:
            print(f"[ExportRunner] 开始调用 YOLO.export 参数: {export_kwargs}", flush=True)
            with _torchscript_save_compat(export_kwargs["format"]):
                exported_path = model.export(**export_kwargs)
        except Exception as e_exp:
            if export_kwargs.get("simplify"):
                print(f"[ExportRunner] 算子精简优化 (simplify) 异常 ({e_exp})，正在自动切换为常规模式重试...", flush=True)
                export_kwargs["simplify"] = False
                params["simplify"] = False
                with _torchscript_save_compat(export_kwargs["format"]):
                    exported_path = model.export(**export_kwargs)
            else:
                raise e_exp

        exported_str = str(exported_path)
        duration = f"{time.time() - t0:.1f}s"

        # Calculate file or directory size
        size_bytes = 0
        if os.path.isfile(exported_str):
            size_bytes = os.path.getsize(exported_str)
        elif os.path.isdir(exported_str):
            for root, _, files in os.walk(exported_str):
                for f in files:
                    size_bytes += os.path.getsize(os.path.join(root, f))
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB" if size_bytes > 0 else "已生成"

        # Format clean summary
        summary_parts = [f"imgsz={params.get('imgsz', 640)}"]
        if params.get("int8"):
            summary_parts.append("quantize=8")
        elif params.get("half"):
            summary_parts.append("quantize=16")
        else:
            summary_parts.append("quantize=32")

        if "dynamic" in params:
            summary_parts.append(f"dynamic={str(params['dynamic']).lower()}")
        if "simplify" in params:
            summary_parts.append(f"simplify={str(params['simplify']).lower()}")
        if "batch" in params:
            summary_parts.append(f"batch={params['batch']}")

        res = {
            "success": True,
            "path": exported_str,
            "summary": ", ".join(summary_parts),
            "size": size_str,
            "duration": duration,
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "params": params
        }
        print(f"\n[ExportRunner] 导出成功: {exported_str} (耗时: {duration}, 大小: {size_str})", flush=True)
        print(f"__EXPORT_RESULT__:{json.dumps(res, ensure_ascii=False)}", flush=True)
        sys.exit(0)

    except Exception as e:
        err_str = str(e)
        trace_str = traceback.format_exc()
        print(f"[ExportRunner] 导出失败异常: {err_str}\n{trace_str}", flush=True)
        res = {
            "success": False,
            "error": err_str,
            "traceback": trace_str
        }
        print(f"__EXPORT_RESULT__:{json.dumps(res, ensure_ascii=False)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
