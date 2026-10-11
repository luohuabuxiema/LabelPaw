# -*- coding: utf-8 -*-
"""Launch and output transport for frozen EXE and external Python workers.

Keep this module free of Qt and ML imports: main.py imports it before GUI startup.
"""
import argparse
import codecs
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

WORKER_MODES = ("--train-worker", "--env-probe", "--export-worker", "--model-inspect")
_DLL_DIRECTORY_LOCK = threading.Lock()


def get_workspace_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def normalize_python(python_path=None):
    """Migrate only this app's old _internal/python.exe to its EXE worker."""
    candidate = os.path.abspath(python_path or sys.executable)
    if getattr(sys, "frozen", False):
        root = get_workspace_root()
        resource_root = getattr(sys, "_MEIPASS", os.path.join(root, "_internal"))
        builtin_paths = (
            sys.executable,
            os.path.join(root, "_internal", "python.exe"),
            os.path.join(resource_root, "python.exe"),
        )
        if os.path.normcase(candidate) in {
            os.path.normcase(os.path.abspath(path)) for path in builtin_paths
        }:
            return os.path.abspath(sys.executable)
    return candidate


def is_builtin_python(python_path=None):
    return bool(getattr(sys, "frozen", False)) and (
        os.path.normcase(normalize_python(python_path))
        == os.path.normcase(os.path.abspath(sys.executable))
    )


def prepare_runner_paths(script_file):
    """External environments prefer installed packages; frozen imports stay intact."""
    workspace = get_workspace_root() if getattr(sys, "frozen", False) else os.path.abspath(
        os.path.join(os.path.dirname(script_file), "..", "..")
    )
    if not getattr(sys, "frozen", False):
        normalized = os.path.normcase(workspace)
        sys.path[:] = [
            path for path in sys.path
            if path and os.path.normcase(os.path.abspath(path)) != normalized
        ]
        sys.path.append(workspace)
    return workspace


def dispatch_worker(argv=None):
    """Return None for GUI mode, otherwise execute one explicit worker and return its status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in WORKER_MODES:
        return None
    mode = argv.pop(0)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-output")
    options, worker_args = parser.parse_known_args(argv)
    originals = (sys.stdout, sys.stderr, sys.stdin, sys.argv)
    output = None
    input_stream = None
    try:
        if options.worker_output:
            output = open(options.worker_output, "a", encoding="utf-8", errors="replace", buffering=1)
            output.reconfigure(write_through=True)
            sys.stdout = sys.stderr = output
        elif sys.stdout is None or sys.stderr is None:
            raise RuntimeError("窗口模式子进程缺少 --worker-output 日志路径")
        if sys.stdin is None:
            input_stream = open(os.devnull, "r")
            sys.stdin = input_stream
        sys.argv = [sys.argv[0], *worker_args]
        # Static imports ensure PyInstaller includes each runner and its dependencies.
        if mode == "--train-worker":
            from TrainingPlatform.core.train_runner import main
        elif mode == "--env-probe":
            from TrainingPlatform.core.env_probe_runner import main
        elif mode == "--model-inspect":
            from TrainingPlatform.core.model_inspect_runner import main
        else:
            from TrainingPlatform.core.export_runner import main
        result = main(worker_args)
        return int(result or 0)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        if sys.stderr is not None:
            print(exc.code, file=sys.stderr, flush=True)
        return 1
    except BaseException as exc:
        if sys.stderr is not None:
            traceback.print_exc()
        if mode == "--train-worker" and sys.stdout is not None:
            print(f"__TRAIN_FAILED__{exc}", flush=True)
        return 1
    finally:
        sys.stdout, sys.stderr, sys.stdin, sys.argv = originals
        if output is not None:
            output.close()
        if input_stream is not None:
            input_stream.close()


def terminate_process_tree(process):
    """Stop a runner and its DataLoader/export children."""
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5, check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def _spawn_external(cmd, options):
    """Do not let the frozen app's DLL search directory override an external environment."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return subprocess.Popen(cmd, **options)
    import ctypes
    from ctypes import wintypes
    with _DLL_DIRECTORY_LOCK:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_directory = kernel32.GetDllDirectoryW
        get_directory.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        get_directory.restype = wintypes.DWORD
        set_directory = kernel32.SetDllDirectoryW
        set_directory.argtypes = [wintypes.LPCWSTR]
        set_directory.restype = wintypes.BOOL
        buffer = ctypes.create_unicode_buffer(get_directory(0, None) + 1)
        get_directory(len(buffer), buffer)
        previous = buffer.value or None
        if not set_directory(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return subprocess.Popen(cmd, **options)
        finally:
            set_directory(previous)


class WorkerProcess:
    """Pipe output for Python workers; tail a file for windowed EXE workers."""

    def __init__(self, python_path, mode, runner_script, args, *, cwd=None, env=None):
        if mode not in WORKER_MODES:
            raise ValueError(f"Unknown worker mode: {mode}")
        python_path = normalize_python(python_path)
        if not os.path.isfile(python_path):
            raise FileNotFoundError(f"运行环境不存在: {python_path}")
        self.output_path = None
        self.process = None
        child_env = os.environ.copy() if env is None else env.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUNBUFFERED"] = "1"
        options = dict(
            stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            cwd=cwd or get_workspace_root(), env=child_env,
        )
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        sink = None
        try:
            builtin = is_builtin_python(python_path)
            if builtin:
                fd, self.output_path = tempfile.mkstemp(prefix="labelpaw_worker_", suffix=".log")
                os.close(fd)
                sink = open(self.output_path, "ab", buffering=0)
                cmd = [python_path, mode, "--worker-output", self.output_path, *args]
                options["stdout"] = sink  # Also capture bootloader errors.
            else:
                if not os.path.isfile(runner_script):
                    raise FileNotFoundError(f"执行脚本不存在: {runner_script}")
                cmd = [python_path, "-u", os.path.abspath(runner_script), *args]
                options.update(stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            self.process = subprocess.Popen(cmd, **options) if builtin else _spawn_external(cmd, options)
        except BaseException:
            if sink is not None:
                sink.close()
            self.close()
            raise
        finally:
            if sink is not None:
                sink.close()

    def iter_lines(self):
        if self.output_path is None:
            yield from self.process.stdout
            self.process.wait()
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        skip_lf = False
        with open(self.output_path, "rb") as source:
            while True:
                chunk = source.read(65536)
                if chunk:
                    pending += decoder.decode(chunk)
                    if skip_lf and pending:
                        if pending.startswith("\n"):
                            pending = pending[1:]
                        skip_lf = False
                    # Both normal lines and tqdm's carriage-return progress are events.
                    while True:
                        end = min((pos for pos in (pending.find("\r"), pending.find("\n")) if pos >= 0), default=-1)
                        if end < 0:
                            break
                        separator = pending[end]
                        yield pending[:end] + "\n"
                        pending = pending[end + 1:]
                        if separator == "\r":
                            if pending.startswith("\n"):
                                pending = pending[1:]
                            elif not pending:
                                skip_lf = True
                    continue
                if self.process.poll() is not None:
                    # One final read covers bytes written just before process exit.
                    pending += decoder.decode(source.read(), final=True)
                    if skip_lf and pending.startswith("\n"):
                        pending = pending[1:]
                    if pending:
                        yield from pending.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
                    break
                time.sleep(0.05)
        self.process.wait()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                terminate_process_tree(self.process)
            if self.process.stdout is not None:
                self.process.stdout.close()
        if self.output_path:
            try:
                os.remove(self.output_path)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
