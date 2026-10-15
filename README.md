<div align="center">
  <p>
    <a href="https://github.com/luohuabuxiema/LabelPaw" target="_blank">
      <img alt="LabelPaw" width="200" src="assets/logo.png"></a>
  </p>
  <a href="README.md">English</a> | <a href="README_zh-CN.md">简体中文</a>
</div>


# LabelPaw - Intelligent Image Annotation & Model Training System (v3.0.0)
## Foreword
Due to the project's need for dataset annotation, I previously used tools like labelme and labelimg. Therefore, I decided to combine excellent computer vision models such as SAM2, SAM3, and YOLO pose estimation to develop a smarter and more efficient annotation tool. After multiple iterations, the system has welcomed its brand-new **v3.0.0 release** (featuring a one-click vision model training platform)!

Source Code Repository: [https://github.com/luohuabuxiema/LabelPaw](https://github.com/luohuabuxiema/LabelPaw)

## Changelog

- **2026-10-15**: Major release **v3.0.0**! Introduced the **One-Click Vision Model Training Platform (TrainingPlatform)** built with modern UI library MonkeyQt. Features one-click YOLO model training, local hardware probe & environment diagnostics, real-time academic metric curve tracking (Precision/Recall/mAP/Loss), hyperparameter tuning, and checkpoint resumption, creating a complete end-to-end loop for "Smart Annotation + Dataset Conversion + Model Training".
- **2026-05-28**: Added SAM 3 batch intelligent annotation, multi-prompt annotation support, and YOLO batch prediction annotation. Integrated new features including delete and multi-select checkbox buttons in the file list panel, along with other comprehensive UI optimizations.
- **2026-05-27**: Removed the class selection dialog and unified class modification and creation within the right-side category history panel. Optimized operations including deleting and hiding specified target boxes in the category tree view.
- **2026-05-15**: Added keypoint skeleton templates for human faces, hands, and pedestrians. Supports customizing keypoint templates and connectivity lines.
- **2026-05-14**: Added the SAM2.1 model for intelligent point-click annotation, and integrated the Ultralytics YOLO model. The YOLO model can be used for intelligent annotation of rectangles, segmentations, keypoints, and OBB (Oriented Bounding Boxes).
- **2026-05-13**: Perfected reciprocal format conversion for JSON/XML/YOLO, supported JSON to U-Net Mask generation, and one-click random dataset splitting.
- **2026-05-10**: Added support for Light and Dark theme modes, providing a more comfortable visual experience.
- **2026-04-12**: Built the PySide6-based foundational intelligent annotation interface (First Release).
- **2026-04-10**: Integrated the next-generation SAM3, supporting hover preview, single-point rapid contour extraction, and text prompt-based full-image target auto-segmentation.
- **2026-04-09**: Supported Rectangle (Rect), Polygon (Poly), and Point annotation, along with the original OBB rotated box control handle (supporting 360° seamless smooth rotation and slide-to-wall detection).
- **2026-04-08**: Supported native saving of JSON, YOLO (.txt), and XML (Pascal VOC).


## System Introduction

The system is built based on PySide6 and integrates the **SAM2**, **SAM3**, and **Ultralytics YOLO** vision models, which significantly enhances annotation efficiency:
- **Intelligent Point-Click & Prompt Segmentation**: When SAM intelligent annotation is enabled, it supports rapid target extraction in polygon, rectangle, and OBB modes.
- **Keypoint Skeleton Templates & Intelligent Annotation**: A brand-new keypoint module with built-in templates for pedestrians, hands, faces, etc. You can customize keypoint templates for quick annotation, and optionally use the YOLO model for intelligent keypoint detection and automatic connection.
- **One-Click Vision Model Training Platform (TrainingPlatform)**: Built with the MonkeyQt modern component library, supporting YOLO detection, segmentation, and pose training. Equipped with hardware environment probes, visual hyperparameter configuration, real-time academic metric curve tracking (Loss/Precision/Recall/mAP), and checkpoint resumption, seamlessly connecting data labeling to model training.

| Feature             | Interface Demonstration                                                     |
| ---------------- | ------------------------------------------------------------ |
| SAM 3 Batch Annotation     | ![SAM 3 Batch](assets/demo_sam3_batch.png) |
| YOLO Model Batch Prediction | ![YOLO Batch](assets/demo_yolo_batch.png) |
| Keypoint Annotation       | ![Keypoint](assets/demo_keypoint.png) |
| OBB Intelligent Annotation      | ![OBB](assets/demo_obb.png) |
| Rectangle Intelligent Annotation     | ![Rectangle](assets/demo_rect.png) |
| Keypoint Intelligent Annotation   | ![Keypoint Intelligent](assets/demo_keypoint_ai.png) |
| Hand Keypoint Template (Dark)   | ![Hand Keypoint Dark](assets/demo_template_hand_dark.png) |
| Built-in Keypoint Templates   | ![Built-in Keypoints](assets/demo_template_builtin.png) |
| Face Keypoint Template   | ![Face Keypoints](assets/demo_template_face.png) |
| Hand Keypoint Template (Light)   | ![Hand Keypoints Light](assets/demo_template_hand.png) |
| Custom Keypoint Template | ![Custom Keypoints](assets/demo_template_custom.png) |
| Dataset Processing Tool   | ![Dataset Tool](assets/demo_dataset_tool.png) |
| Training Dashboard & Hardware Monitor | ![Training Dashboard](assets/training_dashboard.png) |
| Project Metrics & Model Comparison | ![Project Metrics](assets/training_project_metrics.png) |
| Create Task & Hardware Probe | ![Create Training Task](assets/training_create_modal.png) |
| Experiment Curves & Academic Charts | ![Experiment Detail](assets/training_experiment_detail.png) |



## 🙊 Core Features

- **✨ AI Intelligent Assistance (SAM2/SAM3 Driven)**: Hover preview, single-point rapid contour extraction, and text prompt-based full-image target auto-segmentation.
- **🦴 Keypoint Skeleton Templates & Intelligent Annotation (YOLO Driven)**: Supports intelligent annotation for rectangles, segmentations, OBB, and keypoints. Keypoints feature built-in pedestrian (17 points), face (68 points), and hand (21 points) templates. Keypoint annotation supports custom skeleton templates.
- **🚀 One-Click Vision Model Training**: Built-in integrated training platform (TrainingPlatform) supporting rapid training for YOLO detection, segmentation, and pose models, featuring academic multi-curve comparison dashboards, real-time hardware monitoring, and task management.
- **📐 All-round Annotation Modes**: Rectangle (Rect), Polygon (Poly), Point, OBB rotated box, and Keypoint (Pose).
- **🔄 Ultimate OBB Interaction**: Rotated box control handle with 360° stepless smooth rotation and slide-to-wall detection.
- **💾 Multi-Format Conversion & Export**: Native support for saving JSON, YOLO (.txt), and XML (Pascal VOC), and one-click generation of U-Net Masks.
- **🗄️ Dataset Processing Workflow**: Supports train/val/test set random splitting by custom proportions.

---

## 🛠️ Deployment & Execution Environment

### 1. Basic Environment Dependencies

Python 3.10+ is recommended.

Create a virtual environment with the following command:

```bash
conda create -n py311 python==3.11.5
```
Activate the newly created virtual environment:

```bash
conda activate py311 
```


First, install the necessary PyTorch dependency:

Install `torch>=2.5.0` separately. PyTorch official website: [https://pytorch.org/](https://pytorch.org/get-started/previous-versions/?_gl=1*r08hqw*_up*MQ..*_ga*MTg1ODQzMTE5LjE3NzU4ODk5NDI.*_ga_469Y0W5V62*czE3NzU4ODk5NDEkajYwJGwwJGgw/)
![PyTorch Website](assets/env_pytorch_website.png)

**💡 PyTorch Installation Guidelines (Must-read for beginners)**

Please verify the following key points before installing PyTorch to avoid post-installation runtime errors:

**1. Confirm Graphics Card Support and CUDA Version (Extremely Important)**
* **Applicable OS**: This tutorial is based on the Windows environment.
* **How to Check**: Press `Win + R` keys, type `cmd` to open the command prompt, type `nvidia-smi`, and press Enter. In the top right corner of the displayed table, find the **CUDA Version**.
![nvidia-smi](assets/env_nvidia_smi.png)

* **Version Matching Requirement**: The PyTorch CUDA version you download (such as `cu118` or `cu121` in the command) **must be less than or equal to** the CUDA Version shown on your computer. If your computer does not have a dedicated NVIDIA GPU, or if you cannot find this info, please choose the **CPU version** command from the official website.


**2. Choose either Conda or Pip Command**

Install the specified version based on your computer. If the `conda` command gets stuck, you can try configuring domestic mirrors (Tsinghua/USTC) in the terminal first, and then remove `-c pytorch -c nvidia` at the end of the command (since `-c` forces the download from official foreign channels):

```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

The above command usually fails to install in most cases. Therefore, it is highly recommended to use the Alibaba Cloud wheels mirror. The `whl` packages for PyTorch GPU can be viewed at: [https://mirrors.aliyun.com/pytorch-wheels/](https://mirrors.aliyun.com/pytorch-wheels/)

The `cu` version at the end must correspond to your CUDA version (e.g., `cu118` for CUDA 11.8).

```bash
-f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
CUDA 11.8 Installation Command:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
CUDA 12.1 Installation Command:

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu121
```

**3. Verify Installation**
After the installation completes, check if PyTorch is running properly. In the terminal, type `python` and execute the following:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(f"CUDA: {torch.version.cuda}")
```
If it outputs `True`, congratulations, CUDA is successfully configured! If it outputs `False`, it means PyTorch was installed in CPU-only mode or your CUDA version is mismatched, which requires uninstallation and reinstallation.

---

After configuring PyTorch, run the following command in your virtual environment to install other required packages:

```bash
pip install -r requirements.txt
```
```python
pyside6~=6.4.2
numpy~=1.24.4
opencv-python~=4.11.0.86
pillow~=10.4.0
monkeyqt>=0.2.5
monkeyqt-icons>=2.1.2
einops~=0.8.2
pycocotools~=2.0.11
scipy~=1.15.3
tqdm~=4.67.1
iopath~=0.1.10
matplotlib~=3.10.8
timm~=1.0.26
ftfy~=6.3.1
psutil~=7.2.1
torchmetrics~=1.5.0
omegaconf~=2.3.0
numba~=0.64.0
huggingface-hub~=0.36.2
pandas~=2.3.3
scikit-learn~=1.8.0
setuptools==79.0.1
git+https://github.com/facebookresearch/sam3.git
git+https://github.com/facebookresearch/sam2.git
ultralytics==8.4.49
```

> 💡 **TrainingPlatform UI Dependencies (Essential for Upgrading from Older Environments)**:
> The one-click model training platform (TrainingPlatform) is built with the modern PySide6 UI component library **MonkeyQt**. If you are upgrading from an existing LabelPaw environment, you only need to install `monkeyqt` and `monkeyqt-icons`:
> ```bash
> pip install monkeyqt>=0.2.5 monkeyqt-icons>=2.1.2
> ```
> - **MonkeyQt Documentation**: [https://monkeyqt.lhbxm.top/](https://monkeyqt.lhbxm.top/)
> - **MonkeyQt Source Repository**: [https://github.com/luohuabuxiema/MonkeyQt](https://github.com/luohuabuxiema/MonkeyQt)

> Note: To use the AI intelligent annotation assistant, please make sure `sam3`, `sam2`, and `ultralytics` libraries along with their dependencies are correctly configured. If installing `sam3`, `sam2`, or `ultralytics` fails via pip, you can also install them using source code as detailed below.

#### Troubleshooting Common Installation Errors

**(1) Environment creation fails, reporting issues:**
![Conda Error](assets/err_conda_create.png)

* **Solution**: Go to `C:\Users\YourUsername` and delete the `.condarc` file.

**(2) Error `ModuleNotFoundError: No module named 'pkg_resources'`:**

![pkg_resources Error](assets/err_pkg_resources.png)

* **Solution**: Downgrade the `setuptools` version. Set it explicitly to `79.0.1`:

```bash
pip install setuptools==79.0.1
```
![setuptools Installation](assets/fix_setuptools.png)

Reference: [https://blog.csdn.net/u014451778/article/details/158469881](https://blog.csdn.net/u014451778/article/details/158469881/)

**(3) Error `ModuleNotFoundError: No module named 'triton'`:**
![Triton Error](assets/err_triton_missing.png)

* **Solution**: Download the offline installation wheels package for Windows and install separately.
* **Triton offline wheels download path**: [https://hf-mirror.com/madbuda/triton-windows-builds](https://hf-mirror.com/madbuda/triton-windows-builds/)

![Triton Downloads](assets/triton_wheel_download.png)

Reference: [https://blog.csdn.net/qq_42910179/article/details/155606159](https://blog.csdn.net/qq_42910179/article/details/155606159/)

### 2. Source Code Installation for SAM3, SAM2, and Ultralytics

To ensure that SAM2, SAM3, and Ultralytics (YOLO) work optimally, you can download their source code from the official repositories and place them in the `LabelPaw` root directory. Since official libraries are constantly updating, the source code method guarantees the best compatibility.

**Official Repositories**:

- **SAM2**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **SAM3**: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Ultralytics (YOLO)**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**Manual Steps**:

1. Click the green **Code** button and select **Download ZIP** on the respective GitHub page.
2. Unzip the downloaded zip file.
3. **Important**: The unzipped packages contain documents, tests, and many extra files. You only need the **core code directory** (the inner folder named exactly `sam2`, `sam3`, or `ultralytics`).
4. Copy these three core code folders (`sam2`, `sam3`, `ultralytics`) and paste them directly into the root directory of `LabelPaw`.

**Installation via command line (Optional, recommended for advanced users)**:
If you do not want to download and copy folders manually, you can use pip to install directly from the Git source:

```bash
# Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# Install SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# Install Ultralytics (YOLO)
pip install ultralytics
```

> ⚠️ **Notes for `git+` installation method**:
> 1. **Git required**: Your system must have [Git](https://git-scm.com/) installed and configured in your environment PATH, otherwise the command will fail.
> 2. **Network connectivity**: Due to GitHub instability in some regions, using `git+https://...` may encounter connection timeouts. We suggest users configure a command-line proxy or preferentially use the **manual ZIP download and unzip** method above, which is the most reliable.

### 3. Model Download & Path Configuration

**Model Download & Directory Structure**:

To enable intelligent annotation features, you need to download corresponding weight files (`.pt`) and organize them under a standard directory layout.

**1. Recommended Model Directory Structure**
Please set up your model folders under the project root exactly as follows:
```text
 weights/
      ├── sam_weights/          <-- Stores all Segment Anything (SAM) models (must be named exactly this way)
      │    ├── sam3.pt
      │    ├── sam2.1_hiera_tiny.pt
      │    └── ...
      ├── yolo26_weights/       <-- Stores YOLO26 models
      │    ├── yolo26n-pose.pt
      │    └── ...
      ├── yolov8_weights/       <-- You can also create other YOLO model directories
      │    ├── yolov8n.pt
      │    └── ...
      └── ...
```

**2. SAM Model Download & Setup**

- **SAM 3 Model (3.5 GB)**: Download `sam3.pt` from the official repository or HuggingFace. Store it under `\weights\sam_weights\sam3.pt`.
- **SAM 2.1 Models**:
  - SAM 2.1 Tiny: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
  - SAM 2.1 Small: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
  - SAM 2.1 Base: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
  - SAM 2.1 Large: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  - Place them directly under `\weights\sam_weights\` (keep default filenames).

**3. YOLO Model Download & Setup**
- **YOLO Models**: Download latest weights (YOLOv8, YOLO11, YOLOv26, etc.) from the official repository. For Pose Estimation, download pose-specific models (such as `yolo26n-pose.pt`).
- **Place them** in the corresponding folder, e.g., `\weights\yolo26_weights\`. *(Note: You can place any custom-trained YOLO model here and the software will scan and load it automatically!)*

**Modifying Model Base Path**:
For the system to detect your weights, you have two options:

> **Option 1**: Create a folder named `weights` directly in the project root directory, and place models inside following the structure above.

> **Option 2**: Create a `weights` folder elsewhere on your system. **Only** change one base path variable in the codebase:
> Open `main.py`, `labelpaw/models/sam_client.py`, and `ui/model_selector_dialog.py`, find the `HARDCODED_DEV_DIR` variable, and change it to your local absolute path: **`HARDCODED_DEV_DIR = r"YourAbsolutePath\weights"`**


*(Note: The system dynamically scans all folders matching `yolo*_weights` in the directory, so you just need to drop the weights inside the directory without any extra manual configuration!)*

**[Special Notice: Recommendations for Users Without a Dedicated GPU]**
If your computer lacks an independent NVIDIA GPU or has limited system performance, it is strongly recommended to prioritize lightweight YOLO series models (such as models with "n" or "s" suffixes). SAM series models, even the tiny variant, are computationally intensive and may run slowly or cause UI freezes in pure CPU environments, whereas lightweight YOLO models maintain smooth processing speeds on CPU.

### 4. Start the Application

Once everything is configured, run the desired entry point from the project root directory:

- **Main Application (Integrated Image Annotation & Model Training Entry)**:
```bash
python main.py
```
*(You can invoke the model training platform with one click from the left sidebar)*

- **Standalone One-Click Model Training Platform (TrainingPlatform)**:
```bash
python TrainingPlatform/main_app.py
```

---

## 📖 User Operation Guide

### 1. Intelligent Image Annotation Workflow (LabelPaw)

#### 📋 Basic Annotation Steps
1. **Open Directory**: Click "Open Directory" on the left sidebar to select your images folder.
2. **Select Format**: In the left dropdown menu, choose the export annotation format (supports `JSON`, `YOLO`, `XML`).
3. **Annotation Mode**: Select the target mode on the left toolbar (Rectangle, Polygon, Point, OBB Rotated Box, or Keypoint/Pose).
4. **Intelligent Annotation**: Enable **SAM Intelligent Assistant** (Shortcut: `Q`).
   - **SAM2 / SAM3 Models**: Supports real-time hover contour preview and single-click prompt segmentation; SAM3 supports text prompt queries (e.g., "car", "dog") for full-image automatic segmentation.
   - **YOLO Models**: Switch and execute detection, segmentation, OBB, and pose pre-inference from the top toolbar.
5. **Keypoint & Skeleton Templates**: In Keypoint mode, select preset templates (Human 17 keypoints, Face 68 keypoints, Hand 21 keypoints) or customize your own skeleton topologies.
6. **Dataset Processing**: Click "Dataset Tool" on the toolbar to perform bidirectional format conversions (JSON/XML/YOLO), U-Net binary mask generation, and train/val/test split by custom ratios.

#### ⌨️ Annotation Keyboard Shortcuts

| Shortcut | Action Description | Shortcut | Action Description |
| :--- | :--- | :--- | :--- |
| **A / ←** | Previous image | **D / →** | Next image |
| **Ctrl + S** | Save current image annotations | **Q** | Toggle SAM Intelligent Assistant On/Off |
| **R** | Switch to Rectangle mode (Rect) | **P** | Switch to Polygon mode (Poly) |
| **O** | Switch to Rotated Box mode (OBB) | **T** | Switch to Keypoint/Skeleton mode (Pose) |
| **M** | Trigger YOLO model pre-inference | **E** | Modify category label of selected shape |
| **Del / Backspace** | Delete selected shape or point | **Ctrl + Z** | Undo last operation (up to 20 steps) |
| **Ctrl + Y** | Redo operation (Ctrl+Shift+Z) | **Z / X / C / V** | Fine-tune OBB angle with precision |

---

### 2. One-Click Vision Model Training Tutorial (TrainingPlatform)

The TrainingPlatform is built with the modern UI library **MonkeyQt**, designed to empower users to train, monitor, evaluate, and export deep learning vision models through an intuitive graphical interface without writing code.

#### 🛠️ 2.1 Preparation Before Training

Before launching a training session, prepare the following essentials:

##### 1. Dataset Preparation & Directory Structure
After finishing annotation in LabelPaw, use the **Dataset Tool** on the toolbar to export and split your annotations into standard YOLO format:
```text
my_dataset/
├── images/
│   ├── train/          <-- Training images (.jpg, .png, etc.)
│   ├── val/            <-- Validation images
│   └── test/           <-- Test images (optional)
└── labels/
    ├── train/          <-- Corresponding training label files (.txt)
    ├── val/            <-- Corresponding validation label files (.txt)
    └── test/           <-- Corresponding test label files (optional)
```

##### 2. Prepare & Configure the `data.yaml` File
Create a `data.yaml` inside your dataset directory (the platform includes an integrated in-app code editor to inspect and edit YAML on the fly):

- **Standard Tasks (Object Detection / Instance Segmentation / Rotated OBB)**:
```yaml
# Train and Val image paths (supports absolute paths or relative paths)
train: D:/datasets/my_dataset/images/train
val: D:/datasets/my_dataset/images/val
test: # D:/datasets/my_dataset/images/test (optional)

# Number of classes
nc: 2

# Class names mapping
names:
  0: dog
  1: cat
```

- **Keypoint / Pose Estimation Task (Pose)**:
```yaml
train: D:/datasets/my_pose_dataset/images/train
val: D:/datasets/my_pose_dataset/images/val

nc: 1
names:
  0: person

# Keypoint shape: [number_of_keypoints, dimensions(x, y, visible)]
kpt_shape: [17, 3]
# Symmetry flip index mapping for data augmentation
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
```

##### 3. Base Pretrained Model Weights
- Supports YOLOv8, YOLO11, YOLO26 and other popular architectures.
- Choose standard official weights (e.g., `yolov8n.pt`, `yolo11s.pt`, `yolo26n-seg.pt`, `yolo26n-pose.pt`). If not pre-downloaded, the platform will automatically fetch them online upon training start, or you can browse for local weights.

##### 4. Hardware Environment Requirements
- **Dedicated NVIDIA GPU (Recommended)**: An NVIDIA GPU with CUDA-enabled PyTorch installed. The platform comes with an **automatic hardware probe** that identifies available GPUs, VRAM capacity (e.g., RTX 4090 24GB), and CUDA status upon launch.
- **CPU Users**: Training on CPU is fully supported. For CPU execution, select lightweight models (e.g., `yolov8n`) with small batch sizes.

---

#### 🚀 2.2 Step-by-Step Training Guide

##### Step 1: Launch TrainingPlatform
- **Integrated Entry**: In LabelPaw (`python main.py`), click the **"Model Training"** icon on the left sidebar.
- **Standalone Entry**: Directly run `python TrainingPlatform/main_app.py` in your terminal.

##### Step 2: Create or Select a Project
1. Navigate to the **"Project Management"** view from the left menu.
2. Click **"New Project"** in the top-right corner, enter a Project Name (e.g., `Road_Defect_Inspection`), select your task category (Detect, Segment, Pose, or OBB), and confirm.
3. Click into your newly created project details page.

##### Step 3: Configure Training Hyperparameters
Click **"New Training Task"** to open the all-in-one configuration dialog:

1. **Base Configuration (Model & Dataset)**:
   - **Base Model**: Choose a preset model or browse for custom local `.pt` weights.
   - **Dataset YAML**: Click "Browse" to select your `data.yaml`. The integrated editor on the right will preview the contents and allow immediate online modifications.
2. **Core Hyperparameters**:
   - **Epochs**: Default is `100` (typically set between `100` and `300`).
   - **Batch Size**: Enter `auto` for automatic memory allocation, or manually set fixed values like `8`, `16`, `32`.
   - **Image Size (ImgSz)**: Default is `640` (can be set to `1280` for high-resolution small object detection).
   - **Task Name**: Name your experiment run (e.g., `exp1_baseline`).
3. **Advanced Hyperparameters & Strategies (Collapsible Section)**:
   - **Optimizer & Learning Rate**: Select `auto`, `SGD`, `Adam`, or `AdamW`; customize `lr0` (default 0.01), cosine LR schedule (`cos_lr`), and `warmup_epochs`.
   - **Acceleration & Strategies**:
     - **Automatic Mixed Precision (AMP)**: Enabled by default to accelerate training and reduce VRAM usage;
     - **Patience**: Early stopping threshold (default 100);
     - **Close Mosaic**: Disables Mosaic augmentation in the final N epochs to ensure convergence (default 10);
     - **Resume Training**: Check "Resume" and select `last.pt` if a previous run was interrupted.
   - **Augmentations**: Fine-tune Mosaic, Mixup, HSV jitter, Rotation degrees, and horizontal/vertical flip ratios.
4. **Hardware Device Selection**:
   - The probe displays detected GPUs (e.g., `0: NVIDIA GeForce RTX 4090`). Select target GPU devices and set the number of dataloader `workers` (recommended: 4 or 8).

##### Step 4: Start Training & Real-Time Tracking
1. Click **"Start Training"** in the bottom right corner.
2. The task starts in the background and switches to the **"Real-Time Monitor"** dashboard:
   - **Academic Dynamic Metrics Curves**: Displays real-time training and validation curves:
     - **Loss Curves**: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, and validation loss;
     - **Academic Evaluation**: `metrics/precision`, `metrics/recall`, `metrics/mAP50`, `metrics/mAP50-95`;
     - Supports curve smoothing filters and multi-experiment comparisons.
   - **Formatted Terminal Output**: Streams real-time progress bars, VRAM consumption, estimated completion time, and learning rate.
   - **Hardware Probe Cards**: Displays real-time GPU VRAM usage percentage, GPU load, and CPU utilization.

##### Step 5: Evaluation & Model Export
1. Upon completion, academic evaluation charts are aggregated automatically:
   - Confusion Matrix (`confusion_matrix.png`)
   - PR Curves (`PR_curve.png`) and F1-Confidence Curves
   - Validation ground-truth vs. model prediction comparisons (`val_batch_pred.jpg`)
2. **Model Weights Archive**:
   - `best.pt`: Best performing weights based on validation mAP;
   - `last.pt`: Final checkpoint weights (for checkpoint resumption).
3. **Production Model Export**:
   - Under the experiment detail view, click **"Export Model"** to convert `best.pt` to **ONNX**, **TensorRT Engine**, **TorchScript**, etc., for deployment on edge devices and production environments.

---

#### 🔄 2.3 Closed-Loop Annotation & Training Workflow

In LabelPaw, annotation and model training form a complete continuous loop:
```text
┌─────────────────┐      Export Dataset (YOLO format)   ┌──────────────────────┐
│  LabelPaw       │ ──────────────────────────────────> │  TrainingPlatform    │
│  Smart Labeling │                                     │  Model Training      │
└─────────────────┘                                     └──────────────────────┘
        ▲                                                          │
        │           Import generated best.pt weights               │
        └──────────────────────────────────────────────────────────┘
```
1. Annotate a small batch of initial images (e.g., 100~200 images) in LabelPaw and export the dataset;
2. Train your custom model with one click in TrainingPlatform to produce a task-specific `best.pt`;
3. Place `best.pt` into LabelPaw's `weights/` directory;
4. In LabelPaw, select this custom model and enable **Smart Assistant** to run automated pre-inference and batch annotations on subsequent large-scale unlabeled datasets, boosting annotation efficiency by over 10x!

---

#### 💡 2.4 FAQ & Training Tips

- **Q1: Encountering `CUDA Out of Memory (OOM)` errors upon starting training?**
  - **Solutions**:
    1. Lower the `Batch Size` (e.g., from 32 to 16, 8, or 4);
    2. Lower the `ImgSz` (e.g., from 1280 to 640 or 480);
    3. Ensure `AMP (Automatic Mixed Precision)` is enabled in Advanced Settings;
    4. Reduce the number of dataloader `workers` (set to 2 or 4).
- **Q2: Can I train without an independent GPU?**
  - **Solutions**: Yes. Select `CPU` in the training device setting, choose a lightweight architecture like `yolov8n`, set `batch` to 4, and start with fewer epochs for validation.
- **Q3: How do I resume an interrupted training session?**
  - **Solutions**: In the "New Training Task" dialog, check **"Resume"** under advanced settings, point the base model to the `last.pt` checkpoint of the interrupted run, and click Start Training to continue seamlessly.

---

## 🤝 Secondary Development Welcome

The application features a modular design with high cohesion and low coupling. UI and model inference are separated cleanly:
- `main.py`: Main control window and event router.
- `labelpaw/`: Core image annotation packages including the draw canvas (`graphics/canvas.py`), dataset formatting exporter (`data/exporter.py`), SAM intelligent model inference (`models/sam_client.py`), and YOLO pose/detection inference (`models/yolo_predictor.py`).
- `TrainingPlatform/`: One-click model training platform package built on MonkeyQt, including training task orchestration (`core/`), modern views and academic charts (`ui/`), etc.
- `ui/`: Custom widgets and theme styles.

We welcome developers to Fork the repository and submit PRs!

## License

This project is licensed under the GPL-3.0 License. If you utilize this code in commercial or non-commercial projects, please comply with this license and open-source your derivative modifications. If this project helps you, please star our repository!

## Citation

If you use this software in your research, please cite as follows:

```bibtex
@misc{LabelPaw,
  year = {2026},
  author = {luohuabuxiema},
  publisher = {Github},
  journal = {Github repository},
  title = {LabelPaw: Intelligent image annotation system},
  howpublished = {\url{https://github.com/luohuabuxiema/LabelPaw}}
}
```

**Acknowledgments and References**:

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion et al.},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}

@software{ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```