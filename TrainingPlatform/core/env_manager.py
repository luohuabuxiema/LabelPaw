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
import subprocess
from typing import List, Dict, Optional
from PySide6.QtCore import QObject, Signal, QThread


def get_workspace_root() -> str:
    """Return the root directory of LabelPaw workspace."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # cur_dir is TrainingPlatform/core -> workspace root is ../..
    return os.path.abspath(os.path.join(cur_dir, "..", ".."))


def get_platform_data_dir() -> str:
    """Return user data directory for persistent configs."""
    data_dir = os.path.join(os.path.expanduser("~"), ".ai_vision_platform")
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
        self.python_path = python_path

    def run(self):
        workspace = get_workspace_root()
        self.log_line.emit(f">>> 正在测试 Python 环境: {self.python_path}")
        self.log_line.emit(f">>> 正在探测独立虚拟环境依赖 (PyTorch, CUDA, Ultralytics)...")

        if not os.path.isfile(self.python_path):
            self.log_line.emit(f"[ERROR] 指定的 Python 解释器不存在: {self.python_path}")
            self.probe_finished.emit({
                "python_path": self.python_path,
                "valid": False,
                "error": "文件不存在",
                "cuda_available": False,
            })
            return

        probe_script = f"""
import sys
import json

# Clean up workspace and current dir from sys.path so environment's native packages are probed
workspace = r"{workspace}"
while workspace in sys.path:
    sys.path.remove(workspace)
while "" in sys.path:
    sys.path.remove("")
sys.path.append(workspace)

result = {{
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
    "error": None
}}

try:
    import torch
    result["torch_installed"] = True
    result["torch_version"] = str(torch.__version__)
    print(torch.__version__)
    
    cuda_avail = bool(torch.cuda.is_available())
    result["cuda_available"] = cuda_avail
    print(cuda_avail)
    
    dev_count = int(torch.cuda.device_count()) if cuda_avail else 0
    result["device_count"] = dev_count
    print(dev_count)
    
    cuda_ver = str(getattr(torch.version, "cuda", "Unknown"))
    result["cuda_version"] = cuda_ver
    print(f"CUDA: {{cuda_ver}}")
    
    if cuda_avail:
        for i in range(dev_count):
            try:
                name = torch.cuda.get_device_name(i)
                result["device_names"].append(name)
            except Exception:
                result["device_names"].append(f"GPU {{i}}")
except Exception as e:
    result["error"] = str(e)
    print(f"Torch 导入/检测异常: {{e}}")

try:
    import ultralytics
    result["ultralytics_installed"] = True
    result["ultralytics_version"] = str(getattr(ultralytics, "__version__", "未知"))
    result["ultralytics_source"] = str(getattr(ultralytics, "__file__", ""))
except Exception as e:
    result["ultralytics_error"] = str(e)

print("__ENV_PROBE_JSON__" + json.dumps(result))
"""

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Prepend workspace to PYTHONPATH
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{workspace}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else workspace

        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                [self.python_path, "-u", "-c", probe_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                cwd=workspace,
                startupinfo=startupinfo
            )

            result_dict = None
            for line in process.stdout:
                clean_line = line.rstrip()
                if clean_line.startswith("__ENV_PROBE_JSON__"):
                    json_str = clean_line[len("__ENV_PROBE_JSON__"):]
                    try:
                        result_dict = json.loads(json_str)
                    except Exception:
                        pass
                else:
                    # Strip ANSI escape sequences (e.g. \x1b[0m from colorama / tqdm)
                    filtered_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', clean_line).strip()
                    if filtered_line:
                        self.log_line.emit(filtered_line)

            process.wait()

            if not result_dict:
                result_dict = {
                    "python_path": self.python_path,
                    "valid": process.returncode == 0,
                    "error": f"进程退出码: {process.returncode}",
                    "cuda_available": False,
                    "torch_installed": False
                }

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
        sel = cfg.get("selected_python")
        if sel and os.path.isfile(sel):
            return sel
        return sys.executable

    def set_selected_python(self, python_path: str):
        cfg = self.load_config()
        cfg["selected_python"] = python_path
        self.save_config(cfg)

    def add_custom_environment(self, python_path: str) -> bool:
        """Add a user-selected python.exe to custom envs and save."""
        python_path = os.path.abspath(python_path)
        if not os.path.isfile(python_path):
            return False

        cfg = self.load_config()
        customs = cfg.get("custom_envs", [])
        if python_path not in customs:
            customs.append(python_path)
            cfg["custom_envs"] = customs
            cfg["selected_python"] = python_path
            self.save_config(cfg)
        return True

    def remove_custom_environment(self, python_path: str):
        cfg = self.load_config()
        customs = cfg.get("custom_envs", [])
        if python_path in customs:
            customs.remove(python_path)
            cfg["custom_envs"] = customs
            self.save_config(cfg)

    def scan_environments(self) -> List[Dict]:
        """
        Scan all available Python environments:
        1. Current runtime Python (sys.executable)
        2. Conda virtual environments (from ~/.conda/environments.txt & standard Conda paths)
        3. Built-in / Packaged Python (e.g. ./python or ./runtime in app directory)
        4. Workspace venv (.venv/Scripts/python.exe, venv/Scripts/python.exe)
        5. User custom added environments
        """
        discovered = []
        seen_paths = set()

        def _add_env(name: str, py_path: str, category: str, is_default: bool = False):
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
        _add_env(
            f"当前运行环境 ({os.path.basename(os.path.dirname(sys.executable))})",
            sys.executable,
            category="current",
            is_default=True
        )

        # 2. Built-in / Packaged Python (for distributed builds)
        workspace = get_workspace_root()
        builtin_candidates = [
            os.path.join(workspace, "python", "python.exe"),
            os.path.join(workspace, "runtime", "python.exe"),
            os.path.join(workspace, "_internal", "python.exe"),
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
