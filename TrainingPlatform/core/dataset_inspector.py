# -*- coding: utf-8 -*-
"""
Dataset Inspection & Extraction Tool for AI Vision Training Platform
"""
import os
import zipfile
import shutil
import yaml
import cv2

class DatasetInspector:
    @staticmethod
    def extract_zip(zip_path, target_dir):
        """Extract a zip dataset archive to target directory."""
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        return target_dir

    @staticmethod
    def inspect(dataset_dir):
        """
        Inspect a YOLO dataset folder.
        Returns a dict with statistics and validation results.
        """
        res = {
            "valid": True,
            "error_msg": "",
            "dataset_dir": dataset_dir,
            "yaml_path": "",
            "image_count": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "class_count": 0,
            "classes": [],
            "empty_labels": 0,
            "corrupt_images": 0
        }

        if not os.path.exists(dataset_dir):
            res["valid"] = False
            res["error_msg"] = f"目录不存在: {dataset_dir}"
            return res

        # 1. Look for data.yaml or data-pose.yaml
        yaml_candidates = ["data.yaml", "data-pose.yaml", "dataset.yaml"]
        yaml_path = None
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if f.lower() in yaml_candidates:
                    yaml_path = os.path.join(root, f)
                    break
            if yaml_path:
                break

        if yaml_path and os.path.exists(yaml_path):
            res["yaml_path"] = yaml_path
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        res["class_count"] = content.get("nc", 0)
                        names = content.get("names", [])
                        if isinstance(names, dict):
                            res["classes"] = [names[k] for k in sorted(names.keys())]
                        elif isinstance(names, list):
                            res["classes"] = names
                        if not res["class_count"] and res["classes"]:
                            res["class_count"] = len(res["classes"])
            except Exception as e:
                res["error_msg"] += f"解析 data.yaml 警告: {str(e)}; "

        # 2. Count images in train/val/test
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for split in ['train', 'val', 'test']:
            # Search in dataset_dir/images/split or dataset_dir/split/images
            possible_paths = [
                os.path.join(dataset_dir, "images", split),
                os.path.join(dataset_dir, split, "images"),
                os.path.join(dataset_dir, split)
            ]
            count = 0
            for p in possible_paths:
                if os.path.exists(p) and os.path.isdir(p):
                    for fname in os.listdir(p):
                        if fname.lower().endswith(img_extensions):
                            count += 1
                    break
            
            if split == 'train':
                res["train_images"] = count
            elif split == 'val':
                res["val_images"] = count
            elif split == 'test':
                res["test_images"] = count

        res["image_count"] = res["train_images"] + res["val_images"] + res["test_images"]

        # If no split subfolders found, scan root folder for images
        if res["image_count"] == 0:
            all_imgs = [f for f in os.listdir(dataset_dir) if f.lower().endswith(img_extensions)]
            res["image_count"] = len(all_imgs)
            res["train_images"] = len(all_imgs)

        if res["image_count"] == 0:
            res["valid"] = False
            res["error_msg"] = "未在指定目录中找到有效的图像文件！"

        return res
