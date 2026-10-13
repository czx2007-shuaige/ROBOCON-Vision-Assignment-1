#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment3: 训练方块检测模型（YOLO11n，detect）"""
from ultralytics import YOLO

DATA = "dataset_detect/data.yaml"

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")          # 预训练权重作为起点
    model.train(
        data=DATA,
        epochs=200,                     # 训练轮数（用户要求更多次、更精准）
        imgsz=640,                      # 输入尺寸
        batch=16,                       # 8GB 显存足够
        device=0,                       # RTX 5070
        name="block_yolo11n",
        patience=50,                    # 早停（放宽，配合 200 epoch）
    )
    print("\n训练完成，最佳权重: runs/detect/block_yolo11n/weights/best.pt")
