# -*- coding: utf-8 -*-
"""GPU/package probe shared by an external interpreter and the packaged EXE."""
import json
import sys

if __package__:
    from .worker_runtime import prepare_runner_paths
else:
    from worker_runtime import prepare_runner_paths


def main(argv=None):
    prepare_runner_paths(__file__)
    result = {
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "valid": True,
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "device_names": [],
        "ultralytics_installed": False,
        "ultralytics_version": None,
        "ultralytics_source": None,
        "error": None,
    }
    try:
        import torch
        result["torch_installed"] = True
        result["torch_version"] = str(torch.__version__)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count()) if result["cuda_available"] else 0
        result["cuda_version"] = str(getattr(torch.version, "cuda", "Unknown"))
        for index in range(result["device_count"]):
            try:
                result["device_names"].append(torch.cuda.get_device_name(index))
            except Exception:
                result["device_names"].append(f"GPU {index}")
        print(result["torch_version"], flush=True)
        print(result["cuda_available"], flush=True)
        print(result["device_count"], flush=True)
        print(f"CUDA: {result['cuda_version']}", flush=True)
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Torch 导入/检测异常: {exc}", flush=True)
    try:
        import ultralytics
        result["ultralytics_installed"] = True
        result["ultralytics_version"] = str(getattr(ultralytics, "__version__", "未知"))
        result["ultralytics_source"] = str(getattr(ultralytics, "__file__", ""))
    except Exception as exc:
        result["ultralytics_error"] = str(exc)
    result["valid"] = result["torch_installed"] and result["ultralytics_installed"]
    print("__ENV_PROBE_JSON__" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
