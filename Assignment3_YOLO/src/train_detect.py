#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment3: 训练方块检测模型（YOLO11n，detect）"""
from ultralytics import YOLO

DATA = "dataset_detect/data.yaml"

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")          # 预训练权重作为起点
    model.train(
        data=DATA,
        epochs=80,                      # 训练轮数
        imgsz=640,                      # 输入尺寸
        batch=16,                       # 8GB 显存足够
        device=0,                       # RTX 5070
        name="block_yolo11n",
        patience=30,                    # 早停
    )
    print("\n训练完成，最佳权重: runs/detect/block_yolo11n/weights/best.pt")
