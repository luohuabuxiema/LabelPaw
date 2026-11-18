# LabelPaw Local Environment and Feature Change Description

> This document records the actual configuration adjustments and feature changes made to the local LabelPaw project compared with the official repository version.
> It does not replace the official README and does not modify the official annotation or training workflow descriptions.
>
> Record date: 2026-11-17

## 1. Scope and Basis

- Official project: [`luohuabuxiema/LabelPaw`](https://github.com/luohuabuxiema/LabelPaw)
- Official Chinese documentation: [`README_zh-CN.md`](https://github.com/luohuabuxiema/LabelPaw/blob/main/README_zh-CN.md)
- This document is based on the current local project files, the actual installation process, and the verified runtime results.
- The added features do not change LabelPaw's existing image opening, annotation saving, dataset conversion, or training platform workflow.

## 2. Local Environment Configuration Changes

### 2.1 Verified Runtime Environment

The following environment has been successfully verified:

- Operating system: Linux
- Conda environment: `yolo`
- Python: `3.11`
- PyTorch: `2.5.0`
- TorchVision: `0.20.0`
- TorchAudio: `2.5.0`
- Conda CUDA runtime: `pytorch-cuda 11.8`

Create and activate the environment:

```bash
conda create -n yolo python=3.11
conda activate yolo
```

The PyTorch installation command actually used was:

```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 2.2 Actual Changes to `requirements.txt`

The following dependency versions in the current `requirements.txt` differ from the original dependency descriptions in the official README and were adjusted according to the dependency conflicts encountered in practice:

| Dependency | Original README | Current version | Reason |
|---|---|---|---|
| `numpy` | `numpy~=1.24.4` | `numpy~=1.26.4` | `sam3` requires `numpy>=1.26,<2`; the original version conflicted with `sam3` |
| `ftfy` | `ftfy~=6.3.1` | `ftfy==6.1.1` | `sam3 0.1.0` requires exactly `ftfy==6.1.1`; the original version conflicted with `sam3` |

The current file also declares the following installation items:

```text
git+https://github.com/facebookresearch/sam3.git
git+https://github.com/facebookresearch/sam2.git
ultralytics==8.4.49
```

Therefore, under normal circumstances, running:

```bash
pip install -r requirements.txt
```

installs SAM3, SAM2, and Ultralytics. The same installation commands do not need to be executed again.

### 2.3 Installation Methods for SAM2, SAM3, and Ultralytics

The current `requirements.txt` already contains the installation entries for SAM2, SAM3, and Ultralytics. The source-code download, source-directory copying, and `pip install git+...` methods described in the official README are backup methods. They should not be repeatedly combined with the corresponding entries in `requirements.txt`.

If the local-source method is selected, one source should be used consistently:

- Install through `requirements.txt`; or
- Use local source directories; or
- Install from the Git repositories with `pip install git+...`.

Do not keep multiple unverified versions at the same time without checking the actual import source. Otherwise, the version actually imported by Python may differ from the expected version.

### 2.4 MonkeyQt Compatibility Handling

During the actual installation process, the PyPI `monkeyqt` package could complete dependency installation, but the project was missing some components/APIs required by the current TrainingPlatform. The following types of import errors occurred:

- `MkMessageBox` could not be imported from `monkeyqt`;
- `MkSkeleton` could not be imported from `monkeyqt`;
- `monkeyqt.themes.engine` could not be imported;
- TrainingPlatform components such as `MkChartDashboard` could not be imported.

The successful solution was not to modify the LabelPaw training platform code. Instead, the MonkeyQt source was installed and the old package directory that shadowed the source installation was renamed:

```bash
conda activate yolo

git clone https://github.com/luohuabuxiema/MonkeyQt.git /tmp/MonkeyQt-source
pip install -e /tmp/MonkeyQt-source
```

Then rename the original package directory in the current Conda environment. The path used on this machine is:

```bash
mv /home/cjr/miniconda3/envs/yolo/lib/python3.11/site-packages/monkeyqt \
   /home/cjr/miniconda3/envs/yolo/lib/python3.11/site-packages/monkeyqt-old
```

> The `site-packages` path above belongs to the current machine. Other machines must first determine their own Conda Python path and must not copy this path directly.
>
> The current editable installation points to `/tmp/MonkeyQt-source`. Do not delete this directory, or the project may fail to import MonkeyQt again. If the source directory is moved, run `pip install -e <new-directory>` in the new location and verify the installation again.

Verify MonkeyQt and the training platform:

```bash
python -c "import monkeyqt; print(monkeyqt.__file__); from monkeyqt.themes.engine import ThemeEngine; from monkeyqt import MkMessageBox, MkSkeleton, MkSkeletonBlock, MkChartDashboard; print('MonkeyQt API 正常')"

cd ~/下载/LabelPaw
python -c "from TrainingPlatform.ui.main_window import MainWindow; print('训练平台导入正常')"
python main.py
```

After successful verification, the LabelPaw graphical interface can start normally.

## 3. Locally Added and Modified Features

The following features do not modify the trained model structure. They add functionality to the LabelPaw annotation interface and the TrainingPlatform inference result display.

### 3.1 Switching Numeric Classes with Number Keys

Files involved:

- `main.py`
- `ui/annotation_tree_widget.py`

The current implementation registers window-level `QShortcut` shortcuts for `0` through `9` and matches class names exactly:

- Press `0`: select the class whose name is `"0"`;
- Press `1`: select the class whose name is `"1"`;
- …
- Press `9`: select the class whose name is `"9"`.

This is not “select the Nth class in the list.” It matches the class name to the key value. For example:

```text
Class list: 0, 1, 2, 3
Press 2: select class 2
```

The implementation that has been tested supports single digits from `0` to `9`. If no class has the corresponding numeric name, the key does not map to another class.

To avoid interfering with text input, the shortcut does not switch classes when a text-editing field has focus.

### 3.2 Right-Click Category Deletion

Files involved:

- `ui/annotation_tree_widget.py`
- `main.py`

Usage:

1. Right-click the target category in the category list on the right;
2. Select **“删除类别”** (Delete Category);
3. Confirm the deletion in the confirmation dialog.

After a category is deleted, the current implementation synchronizes the following operations:

- Delete annotation objects belonging to that category from the current canvas;
- Remove the category from the current category list;
- Remove the corresponding category objects from JSON annotation files in the current dataset directory;
- Remove the corresponding objects from XML annotation files;
- Remove annotation lines for that category from YOLO `.txt` files;
- Decrease the category IDs after the deleted category by one in YOLO files;
- Update the class files, category tree, and current canvas;
- Perform automatic saving and record the state change.

The deletion dialog explicitly states that annotations for this category in the current dataset will be deleted and that subsequent YOLO category IDs will be adjusted.

> This operation modifies the current dataset files. Back up the images and annotation files before applying it to an important dataset.

### 3.3 Consistent Colors in the TrainingPlatform Inference Page

File involved:

```text
TrainingPlatform/ui/views/experiment_detail_view.py
```

The original display used two different color rules:

- Detection boxes in the image were colored by Ultralytics `res.plot()` according to class ID;
- The dots in the “识别目标” (Detected Objects) list on the right were colored according to the order in which detections appeared.

When the detection order did not match the class IDs, the box color in the image could differ from the color shown in the legend.

The current implementation:

1. Reads the class ID of each detection box;
2. Uses the Ultralytics class-color function to obtain the color for that class;
3. Passes the same class color to the corresponding detection-result dot on the right;
4. Ensures that the image detection boxes and the right-side legend use the same class-color mapping.

This change only affects the interface display. It does not change:

- Model weights;
- Detection-box coordinates;
- Class names;
- Confidence scores;
- The training workflow;
- The inference results themselves.

After changing the code, restart LabelPaw, open the “推理预测” (Inference) page of the training task again, and upload an image to see the change.

## 4. Current Startup Method

```bash
conda activate yolo
cd ~/下载/LabelPaw
python main.py
```

Do not start the project in the `base` environment without activating `yolo` first. The currently verified PySide6, PyTorch, MonkeyQt, and project dependencies are installed in the `yolo` environment.

## 5. Current Verification Scope

The following items have been confirmed or completed in practice:

- `requirements.txt` was installed successfully in the current `yolo` environment;
- After the MonkeyQt source compatibility handling, the LabelPaw main interface starts normally;
- TrainingPlatform can be imported;
- Numeric label shortcut switching has been used successfully;
- Right-click category deletion has been used successfully;
- Detection-box colors and right-side legend colors in the inference page are unified by class;
- The changes recorded in this document do not require modifications to the official project's original training logic.

This document does not present other operating systems, Python versions, GPUs, MonkeyQt installation paths, or model task types that were not verified on the current machine as universal guarantees.

## 6. Scope Not Modified in This Change

The following are outside the scope of the local changes documented here:

- The basic annotation modes already provided by the official LabelPaw project;
- The SAM2, SAM3, and Ultralytics model algorithms;
- YOLO training algorithms and training hyperparameter logic;
- Dataset annotation rules themselves;
- Robot-side deployment and TensorRT deployment workflows;
- The original annotation and model-training chapters in the official README.
