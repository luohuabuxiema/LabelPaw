# -*- coding: utf-8 -*-
"""
Python Environment Manager for AI Vision Platform
Provides dynamic discovery, validation, and probe execution for Python virtual environments
(Conda, venv, embedded, system Python, custom paths).
"""
import os
import sys
import re
import json
import shutil
from typing import List, Dict, Optional
from PySide6.QtCore import QObject, Signal, QThread
from TrainingPlatform.core.worker_runtime import (
    WorkerProcess, get_workspace_root, normalize_python,
)


def get_platform_data_dir() -> str:
    """Return workspace runs directory for persistent configs (portable)."""
    data_dir = os.path.join(get_workspace_root(), "runs")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def format_gpu_model_name(name: str) -> str:
    """
    Format full GPU device name into concise model name.
    Examples:
        'NVIDIA GeForce RTX 4070' -> 'RTX 4070'
        'NVIDIA GeForce RTX 4090 D' -> 'RTX 4090 D'
        'NVIDIA GeForce RTX 3060 Laptop GPU' -> 'RTX 3060'
        'NVIDIA RTX A4000' -> 'RTX A4000'
        'GeForce GTX 1660 Ti' -> 'GTX 1660 Ti'
    """
    if not name:
        return "GPU 0"
    s = name.strip()
    for prefix in ["NVIDIA GeForce ", "NVIDIA ", "GeForce "]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    for suffix in [" Laptop GPU", " GPU"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break
    return s


class EnvProbeWorker(QThread):
    """
    Background worker thread to probe a Python executable for PyTorch, CUDA, and Ultralytics
    without blocking the GUI event loop.
    """
    log_line = Signal(str)
    probe_finished = Signal(dict)

    def __init__(self, python_path: str, parent=None):
        super().__init__(parent)
        self.python_path = normalize_python(python_path)

    def run(self):
        workspace = get_workspace_root()
        self.log_line.emit(f">>> 正在测试 Python 环境: {self.python_path}")
        self.log_line.emit(f">>> 正在探测运行环境依赖 (PyTorch, CUDA, Ultralytics)...")

        if not os.path.isfile(self.python_path):
            self.log_line.emit(f"[ERROR] 指定的 Python 解释器不存在: {self.python_path}")
            self.probe_finished.emit({
                "python_path": self.python_path,
                "valid": False,
                "error": "文件不存在",
                "cuda_available": False,
            })
            return

        try:
            result_dict = None
            runner_script = os.path.join(os.path.dirname(__file__), "env_probe_runner.py")
            with WorkerProcess(self.python_path, "--env-probe", runner_script, [], cwd=workspace) as child:
                for line in child.iter_lines():
                    clean_line = line.rstrip()
                    if clean_line.startswith("__ENV_PROBE_JSON__"):
                        try:
                            result_dict = json.loads(clean_line[len("__ENV_PROBE_JSON__"):])
                        except (ValueError, TypeError):
                            pass
                    else:
                        filtered_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', clean_line).strip()
                        if filtered_line:
                            self.log_line.emit(filtered_line)
                returncode = child.process.returncode

            if not result_dict:
                result_dict = {
                    "python_path": self.python_path,
                    "valid": False,
                    "error": f"环境探测未返回有效结果 (退出码: {returncode})",
                    "cuda_available": False,
                    "torch_installed": False
                }
            elif returncode != 0:
                result_dict["valid"] = False
                result_dict["error"] = result_dict.get("error") or f"环境探测退出码: {returncode}"

            # Summary evaluation
            if result_dict.get("cuda_available"):
                dev_names = ", ".join(result_dict.get("device_names", [])) or "GPU 设备"
                self.log_line.emit(f"[OK] GPU 检测通过: {dev_names} (CUDA: {result_dict.get('cuda_version')})")
            elif result_dict.get("torch_installed"):
                self.log_line.emit(f"[WARN] 仅 CPU 可用: PyTorch {result_dict.get('torch_version')} (未启用 CUDA/GPU)")
            else:
                self.log_line.emit(f"[ERROR] 未检测到可用的 PyTorch: {result_dict.get('error', '未安装')}")

            if result_dict.get("ultralytics_installed"):
                src = result_dict.get("ultralytics_source", "")
                if workspace.lower() in src.lower():
                    self.log_line.emit(f"[INFO] Ultralytics 来源: 项目本地文件夹 ({src})")
                else:
                    self.log_line.emit(f"[INFO] Ultralytics 来源: 虚拟环境已安装包 ({src})")
            else:
                self.log_line.emit("[WARN] Ultralytics 未能正常加载")

            self.probe_finished.emit(result_dict)

        except Exception as e:
            self.log_line.emit(f"[ERROR] 运行测试探针异常: {str(e)}")
            self.probe_finished.emit({
                "python_path": self.python_path,
                "valid": False,
                "error": str(e),
                "cuda_available": False,
                "torch_installed": False
            })


class EnvironmentManager(QObject):
    """
    Singleton / service that discovers, manages, and probes Python environments.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EnvironmentManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_path = os.path.join(get_platform_data_dir(), "env_config.json")
        self.cached_envs: List[Dict] = []
        self.cached_probe_results: Dict[str, Dict] = {}
        self._current_worker: Optional[EnvProbeWorker] = None

    def load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"custom_envs": [], "selected_python": None}

    def save_config(self, cfg: dict):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_selected_python(self) -> str:
        cfg = self.load_config()
        stored = cfg.get("selected_python")
        sel = normalize_python(stored) if stored else None
        if sel and sel != stored:
            cfg["selected_python"] = sel
            self.save_config(cfg)
        if sel and os.path.isfile(sel):
            return sel
        envs = self.scan_environments()
        if envs:
            return envs[0]["path"]
        if not getattr(sys, 'frozen', False):
            return sys.executable
        import shutil
        sys_py = shutil.which("python")
        if sys_py and os.path.isfile(sys_py):
            return sys_py
        return ""

    def set_selected_python(self, python_path: str):
        cfg = self.load_config()
        cfg["selected_python"] = normalize_python(python_path)
        self.save_config(cfg)

    def add_custom_environment(self, python_path: str) -> bool:
        """Add a user-selected python.exe to custom envs and save."""
        python_path = normalize_python(python_path)
        if not os.path.isfile(python_path):
            return False

        cfg = self.load_config()
        customs = cfg.get("custom_envs", [])
        if python_path not in customs:
            customs.append(python_path)
            cfg["custom_envs"] = customs
            cfg["selected_python"] = python_path
            self.save_config(cfg)
            self.cached_envs = []
        return True

    def remove_custom_environment(self, python_path: str):
        cfg = self.load_config()
        customs = cfg.get("custom_envs", [])
        if python_path in customs:
            customs.remove(python_path)
            cfg["custom_envs"] = customs
            self.save_config(cfg)
            self.cached_envs = []

    def scan_environments(self, force_refresh: bool = False) -> List[Dict]:
        """
        Scan all available Python environments:
        1. Current runtime Python (sys.executable)
        2. Conda virtual environments (from ~/.conda/environments.txt & standard Conda paths)
        3. Built-in / Packaged Python (e.g. ./python or ./runtime in app directory)
        4. Workspace venv (.venv/Scripts/python.exe, venv/Scripts/python.exe)
        5. User custom added environments
        """
        if not force_refresh and getattr(self, "cached_envs", None):
            return self.cached_envs

        discovered = []
        seen_paths = set()

        def _add_env(name: str, py_path: str, category: str, is_default: bool = False):
            py_path = normalize_python(py_path)
            norm_path = os.path.normpath(py_path).lower()
            if norm_path in seen_paths:
                return
            if not os.path.isfile(py_path):
                return
            seen_paths.add(norm_path)
            discovered.append({
                "name": name,
                "path": py_path,
                "category": category,
                "is_current": (norm_path == os.path.normpath(sys.executable).lower()),
                "is_default": is_default
            })

        # 1. Current Runtime Python
        if getattr(sys, 'frozen', False):
            _add_env(
                "内置运行环境（随软件提供）",
                sys.executable,
                category="builtin",
                is_default=True
            )
        else:
            _add_env(
                f"当前开发环境 ({os.path.basename(os.path.dirname(sys.executable))})",
                sys.executable,
                category="current",
                is_default=True
            )

        # 2. Built-in / Packaged Python (for distributed builds)
        workspace = get_workspace_root()
        builtin_candidates = [
            os.path.join(workspace, "python", "python.exe"),
            os.path.join(workspace, "runtime", "python.exe"),
            os.path.join(workspace, "bin", "python.exe"),
        ]
        for b_py in builtin_candidates:
            if os.path.isfile(b_py):
                _add_env("内置独立环境 (Built-in)", b_py, category="builtin")

        # 3. Workspace venv
        venv_candidates = [
            os.path.join(workspace, ".venv", "Scripts", "python.exe"),
            os.path.join(workspace, "venv", "Scripts", "python.exe"),
            os.path.join(workspace, "env", "Scripts", "python.exe"),
            os.path.join(workspace, ".venv", "bin", "python"),
            os.path.join(workspace, "venv", "bin", "python"),
        ]
        for v_py in venv_candidates:
            if os.path.isfile(v_py):
                _add_env("项目虚拟环境 (venv)", v_py, category="venv")

        # 4. Conda Environments
        # 4a. Check ~/.conda/environments.txt
        conda_env_file = os.path.expanduser("~/.conda/environments.txt")
        if os.path.isfile(conda_env_file):
            try:
                with open(conda_env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        dir_path = line.strip()
                        if not dir_path or not os.path.isdir(dir_path):
                            continue
                        py_win = os.path.join(dir_path, "python.exe")
                        py_unix = os.path.join(dir_path, "bin", "python")
                        target_py = py_win if os.path.isfile(py_win) else py_unix
                        if os.path.isfile(target_py):
                            env_name = os.path.basename(dir_path)
                            _add_env(f"Conda: {env_name}", target_py, category="conda")
            except Exception:
                pass

        # 4b. Check common Conda directories if conda_env_file missed some
        common_conda_bases = [
            r"C:\ProgramData\anaconda3\envs",
            r"C:\ProgramData\miniconda3\envs",
            os.path.expanduser(r"~\anaconda3\envs"),
            os.path.expanduser(r"~\miniconda3\envs"),
            r"D:\1-Python\ProgramFiles\Miniconda\envs",
        ]
        for base_dir in common_conda_bases:
            if os.path.isdir(base_dir):
                try:
                    for sub in os.listdir(base_dir):
                        sub_dir = os.path.join(base_dir, sub)
                        if os.path.isdir(sub_dir):
                            py_win = os.path.join(sub_dir, "python.exe")
                            if os.path.isfile(py_win):
                                _add_env(f"Conda: {sub}", py_win, category="conda")
                except Exception:
                    pass

        # 5. User Custom Environments from config
        cfg = self.load_config()
        for cust_py in cfg.get("custom_envs", []):
            if os.path.isfile(cust_py):
                name = f"自定义: {os.path.basename(os.path.dirname(cust_py))}"
                _add_env(name, cust_py, category="custom")

        # 6. System PATH Python
        import shutil
        sys_py = shutil.which("python")
        if sys_py and os.path.isfile(sys_py):
            if not (getattr(sys, 'frozen', False) and os.path.abspath(sys_py).lower() == os.path.abspath(sys.executable).lower()):
                _add_env("系统环境 (System Python)", sys_py, category="system")

        if discovered and not any(e.get("is_default") for e in discovered):
            discovered[0]["is_default"] = True

        self.cached_envs = discovered
        return discovered

    def probe_environment_async(self, python_path: str, log_cb=None, finish_cb=None) -> EnvProbeWorker:
        """
        Run asynchronous environment probe in a worker thread.
        """
        worker = EnvProbeWorker(python_path)
        if log_cb:
            worker.log_line.connect(log_cb)

        def _on_finish(res):
            self.cached_probe_results[python_path] = res
            if finish_cb:
                finish_cb(res)

        worker.probe_finished.connect(_on_finish)
        self._current_worker = worker
        worker.start()
        return worker
