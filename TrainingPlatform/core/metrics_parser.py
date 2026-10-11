# -*- coding: utf-8 -*-
"""
Metrics Parser Utility
Extracts accurate validation metrics (e.g. all-classes mean mAP50, mAP50-95, precision, recall)
from training logs (train.log) and results.csv.
Prioritizes the final validation evaluation of best.pt.
"""
import os
import re
import csv
from typing import Dict


def parse_final_experiment_metrics(save_dir: str, task_type=None) -> Dict[str, float]:
    """
    Extract the final model validation metrics (all classes mean) from train.log or results.csv.
    Returns a dict with:
        precision: float
        recall: float
        map50: float
        map50_95: float
    """
    res = {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
    }
    if not save_dir or not os.path.isdir(save_dir):
        return res
    if task_type is None:
        try:
            import yaml
            with open(os.path.join(save_dir, "args.yaml"), encoding="utf-8") as source:
                task_type = (yaml.safe_load(source) or {}).get("task")
        except Exception:
            pass

    # 1. Primary Source: train.log final validation block on best.pt
    # Ultralytics prints:
    # Validating ...weights/best.pt...
    #                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
    #                    all          4          4      0.584      0.987      0.829      0.559
    log_p = os.path.join(save_dir, "train.log")
    if os.path.isfile(log_p):
        try:
            with open(log_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            val_sections = re.findall(
                r'Validating[^\n]*(?:best\.pt|last\.pt)[\s\S]*?(?:__TRAIN_SUCCESS__|\[SUCCESS\]|\Z)',
                text
            )
            search_text = val_sections[-1] if val_sections else text

            # Look for the 'all' row which gives the mean across all classes
            # Pattern matches: all <images> <instances> <P> <R> <mAP50> <mAP50-95>
            pattern = r'^\s*all\s+\d+\s+\d+\s+([^\r\n]+)'
            matches = re.findall(pattern, search_text, re.MULTILINE)
            if matches:
                values = [float(value) for value in matches[-1].split()]
                secondary = task_type in ("segment", "pose") or (task_type is None and "Mask(" in search_text)
                offset = 4 if secondary and len(values) >= 8 else 0
                p, r, m50, m95 = values[offset:offset + 4]
                res["precision"] = round(float(p), 4)
                res["recall"] = round(float(r), 4)
                res["map50"] = round(float(m50), 4)
                res["map50_95"] = round(float(m95), 4)
                return res
        except Exception:
            pass

    # 2. Secondary Source: results.csv
    csv_p = os.path.join(save_dir, "results.csv")
    if os.path.isfile(csv_p):
        try:
            with open(csv_p, "r", encoding="utf-8", errors="ignore") as f:
                reader = list(csv.DictReader(f))
            if reader:
                last = {key.strip(): value for key, value in reader[-1].items() if key}
                suffix = {"segment": "M", "pose": "P"}.get(task_type, "B")
                if task_type is None and "metrics/mAP50(M)" in last:
                    suffix = "M"
                v50 = last.get(f"metrics/mAP50({suffix})") or last.get("metrics/mAP50")
                v95 = last.get(f"metrics/mAP50-95({suffix})") or last.get("metrics/mAP50-95")
                vp = last.get(f"metrics/precision({suffix})")
                vr = last.get(f"metrics/recall({suffix})")
                if v50:
                    res["map50"] = round(float(v50), 4)
                if v95:
                    res["map50_95"] = round(float(v95), 4)
                if vp:
                    res["precision"] = round(float(vp), 4)
                if vr:
                    res["recall"] = round(float(vr), 4)
        except Exception:
            pass

    return res
