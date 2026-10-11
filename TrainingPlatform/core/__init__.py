# -*- coding: utf-8 -*-
"""Core services, imported on demand so worker entry points do not load Qt."""
from importlib import import_module

_EXPORTS = {
    "DatabaseManager": "TrainingPlatform.db.database",
    "GPUMonitor": "TrainingPlatform.core.gpu_monitor",
    "DatasetInspector": "TrainingPlatform.core.dataset_inspector",
    "TrainWorker": "TrainingPlatform.core.train_worker",
    "TaskManager": "TrainingPlatform.core.task_manager",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
