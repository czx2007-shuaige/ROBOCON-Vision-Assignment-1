# Assignment3 — YOLO11n 比赛方块检测

使用 Ultralytics YOLO11n 训练一个**单类目标检测**模型，从 RealSense D435i 彩色图中检测红蓝双色比赛方块（类别 `block`），并输出边界框。

本版本为**完全重做**：使用新相机采集数据，并由**专业标注平台 CVAT** 人工标注，彻底解决上一版“框偏小 / 跨相机域偏移导致检测不准”的问题。

---

## 1. 环境依赖

```bash
pip install ultralytics opencv-python numpy pyrealsense2
```

- 训练 / 推理需 CUDA（本机 RTX 5070，`device=0`）
- 实时相机测试需连接 RealSense D435i（彩色流即可，不依赖深度）

---

## 2. 数据采集（`src/capture_blocks.py`）

用实际用于测试的相机拍图（**训练与测试必须同相机，否则有域偏移**）：

```bash
python3 src/capture_blocks.py
```

- 空格 / `s` 保存一组（彩色 png + 深度 npy + 内参 json），`q` 退出
- 本次使用**新购入的 D435i**（非旧设备），共采集 184 组

---

## 3. 数据标注（CVAT，专业标注平台）

不使用自动/脚本标注，改用 **[CVAT](https://cvat.ai)** 专业标注工具人工拖框：

1. CVAT 建任务，标签仅 `block` 一个
2. 上传全部 `*_color.png`，逐张用 **Rectangle / Shape** 拖框框住**整个方块**（红面 + 蓝面都进框，框线贴边缘）
3. 导出格式选 **YOLO 1.1**，得到每张图的 `.txt` 标注

标注文件（YOLO 格式 `class_id cx cy w h`，归一化）最终与图片同名存放。
> 注意：CVAT 导出目录为 `obj_train_data/`，需将其中的 `*_color.txt` 复制到图片同级目录。

---

## 4. 数据集组织（`src/split_dataset.py`）

```bash
python3 src/split_dataset.py
```

- 读取 `captures/` 下有效标注（排除 10 张 CVAT 漏标的空标注），共 **174 张**有效样本
- 按 **8:1:1** 随机切分：`train 139 / val 17 / test 18`
- 生成 `dataset_detect/`（`images/`、`labels/` 各含 train/val/test 子目录）+ `data.yaml`（`nc=1, names=['block']`）

---

## 5. 训练（`src/train_detect.py`）

```bash
python3 src/train_detect.py
```

- 预训练权重：`yolo11n.pt`
- 配置：`epochs=200, imgsz=640, batch=16, patience=50`
- 实际 **104 epoch 早停**（best @ epoch 54），权重：`runs/detect/block_yolo11n/weights/best.pt`

### 精度指标

| 集合 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|-----------|--------|---------|--------------|
| val (17张) | 0.997 | 1.000 | 0.995 | 0.995 |
| test (18张, 独立) | — | 漏检 0 | IoU(推理/标注) 中位 **0.961**，最小 **0.906** | — |

test 集以“推理框 vs 人工标注框”的 IoU 验证，中位 0.961、零漏检，表明模型框的位置与大小都贴合人工标注（无偏小 / 无偏移）。

---

## 6. 推理

### 单张 / 目录图片测试（`src/predict_detect.py`）

```bash
python3 src/predict_detect.py --source <图片或目录> --weights weights/best.pt
```

### 实时相机测试（`src/camera_test_yolo.py`）

```bash
python3 src/camera_test_yolo.py            # 默认 RealSense D435i
python3 src/camera_test_yolo.py --cam 0    # 普通 USB / 笔记本摄像头
```

实时画面中每个被检出的方块显示绿框 + `block 0.xx`（类别与置信度）；`q` 退出，`空格` 保存当前帧。

---

## 7. 结果示例

检测结果示例见 `results/result.jpg`（test 集中 IoU 最高的一帧，绿框紧贴方块边界）。

---

## 8. 目录结构

```
Assignment3_YOLO/
├── README.md
├── src/
│   ├── capture_blocks.py   # 数据采集（D435i）
│   ├── label_detect.py     # 备选：本地标注脚本（本次用 CVAT，未使用）
│   ├── split_dataset.py    # 8:1:1 切分 + data.yaml
│   ├── train_detect.py     # 训练
│   ├── predict_detect.py   # 图片推理
│   ├── camera_test_yolo.py # 实时相机检测
│   ├── gen_and_eval.py     # test 集 IoU 验证 + 生成 result.jpg
│   └── block_pose.py       # 任务一：基于深度+YOLO掩码估计方块位姿
├── dataset_detect/         # 174 张训练数据（images/labels + data.yaml）
├── weights/
│   └── best.pt             # 训练得到的最佳权重
└── results/
    └── result.jpg          # 检测效果示例
```

---

## 9. 备注

- **为什么重做**：上一版用旧 D435i 采集 + 自动/脚本标注，存在（a）标注框偏小（仅框住方块约 28%）；（b）换另一台相机测试出现域偏移导致框不准、色彩偏差。本次改用**新相机同域采集 + CVAT 专业人工标注**，从根本上修正。
- **位姿任务**：`block_pose.py` 利用本检测结果的掩码结合深度图估计方块 3D 位姿；若用于队内赛抓取，需对新相机**重新标定内参**（生成对应 calib 文件）。
