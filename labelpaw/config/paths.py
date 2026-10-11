# -*- coding: utf-8 -*-
"""
LabelPaw 资源与权重路径管理模块
统一管理应用程序根目录、打包运行环境 (PyInstaller) 以及 weights 目录定位与自动创建。
"""
import os
import sys


def get_app_root_dir() -> str:
    """
    获取应用程序根目录：
    - 打包运行模式 (PyInstaller): 返回 LabelPaw.exe 所在的根目录 (如 dist/LabelPaw)
    - 源码开发模式: 返回 LabelPaw 项目根目录
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    # 本文件在 labelpaw/config/paths.py，向上 3 级即为项目根目录
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_weights_base_dir() -> str:
    """
    智能获取模型权重基准目录 (weights 目录)。
    解析优先级：
    1. 环境变量 LABELPAW_WEIGHTS_DIR (方便服务器或多环境自定义)
    2. 软件根目录下的 weights 文件夹 (无论是打包还是源码运行，这是首选规范目录)
    3. 本地开发回退兼容目录 (仅在开发模式且目录真实存在时作为开发便利兜底)
    4. 自动创建并返回：若均不存在，自动在软件根目录下新建 weights 及其子目录，并写入说明文件。
    """
    # 1. 优先检查环境变量
    env_dir = os.environ.get("LABELPAW_WEIGHTS_DIR", "").strip()
    if env_dir and os.path.exists(env_dir):
        return env_dir

    app_root = get_app_root_dir()
    local_weights = os.path.join(app_root, "weights")

    # 2. 检查程序根目录下的 weights 文件夹 (推荐的标准存放位置)
    if os.path.exists(local_weights):
        sw = os.path.join(local_weights, "sam_weights")
        if os.path.exists(sw) and any(f.endswith(".pt") for f in os.listdir(sw)):
            return local_weights

    # 3. 开发者本地历史兼容 (仅在本地开发路径存在且有权重时兜底，方便本机直接运行打包 exe 测试)
    dev_compat_dir = r"E:\11-AI\标注工具\weights"
    if os.path.exists(dev_compat_dir):
        sw = os.path.join(dev_compat_dir, "sam_weights")
        if os.path.exists(sw) and any(f.endswith(".pt") for f in os.listdir(sw)):
            return dev_compat_dir

    if os.path.exists(local_weights):
        return local_weights

    # 4. 打包分发/给他人使用场景：若 weights 目录不存在，自动在根目录下创建
    try:
        os.makedirs(os.path.join(local_weights, "sam_weights"), exist_ok=True)
        readme_path = os.path.join(local_weights, "说明-模型权重存放位置.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(
                    "【LabelPaw 模型权重存放说明】\n\n"
                    "1. SAM 系列模型请存放在本目录下的 sam_weights 文件夹中：\n"
                    "   - 路径示例: weights/sam_weights/sam2.1_hiera_small.pt\n"
                    "   - 路径示例: weights/sam_weights/sam3.pt\n\n"
                    "2. YOLO 系列模型请存放在形如 yolo*_weights 的文件夹中：\n"
                    "   - 路径示例: weights/yolo11_weights/yolo11n-pose.pt\n"
                    "   - 路径示例: weights/yolo26_weights/yolo26n-seg.pt\n\n"
                    "支持的格式包含 .pt 与 .onnx。\n"
                )
    except Exception:
        pass

    return local_weights
