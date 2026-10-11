"""Build as a small --windowed EXE to exercise frozen transport and spawn support.

Exclude the three production runner modules when building this harness: lightweight
fixtures stand in for ML work so this check does not bundle a second copy of Torch.
"""
import json
import multiprocessing
import os
import sys
import types


def child_ping(queue):
    queue.put("spawn-ok")


def fixture_train(argv):
    if argv != ["中文 参数"]:
        raise AssertionError(f"Lost worker arguments: {argv}")
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_ping, args=(queue,))
    process.start()
    try:
        if queue.get(timeout=20) != "spawn-ok":
            raise AssertionError("DataLoader-style spawn failed")
        process.join(timeout=20)
        if process.exitcode != 0:
            raise AssertionError(f"Spawn worker exit: {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()
        queue.join_thread()
    print("中文训练日志", flush=True)
    print('__TRAIN_PROGRESS__{"epoch": 1}', flush=True)
    print("__TRAIN_SUCCESS__", flush=True)


def fixture_probe(argv):
    print('__ENV_PROBE_JSON__{"valid": true}', flush=True)


def fixture_export(argv):
    print('__EXPORT_RESULT__:{"success": true, "path": "中文模型.pt"}', flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    from TrainingPlatform.core.worker_runtime import WorkerProcess, dispatch_worker

    for module_name, entry in (
        ("train_runner", fixture_train), ("env_probe_runner", fixture_probe),
        ("export_runner", fixture_export),
    ):
        name = "TrainingPlatform.core." + module_name
        module = types.ModuleType(name)
        module.main = entry
        sys.modules[name] = module

    status = dispatch_worker()
    if status is not None:
        raise SystemExit(status)
    report_path = sys.argv[sys.argv.index("--report") + 1]
    report = {}
    try:
        legacy_python = os.path.join(os.path.dirname(sys.executable), "_internal", "python.exe")
        for mode in ("--train-worker", "--env-probe", "--export-worker"):
            args = ["中文 参数"] if mode == "--train-worker" else []
            with WorkerProcess(legacy_python, mode, "unused.py", args) as child:
                output = "".join(child.iter_lines())
                report[mode] = {"exit_code": child.process.returncode, "output": output}
                if child.process.returncode != 0:
                    raise AssertionError(f"{mode} failed: {output}")
        if "__TRAIN_SUCCESS__" not in report["--train-worker"]["output"]:
            raise AssertionError("Missing training completion")
        if "中文训练日志" not in report["--train-worker"]["output"]:
            raise AssertionError("Lost UTF-8 output")
    except BaseException as exc:
        report["error"] = str(exc)
        status = 1
    else:
        status = 0
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    raise SystemExit(status)
