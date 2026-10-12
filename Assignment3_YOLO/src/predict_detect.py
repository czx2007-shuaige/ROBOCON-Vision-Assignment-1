#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment3: 用训练好的模型对测试集图片推理，保存带检测框的结果图"""
import sys
import glob
from ultralytics import YOLO

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else \
    "runs/detect/block_yolo11n/weights/best.pt"

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    test_imgs = sorted(glob.glob("dataset_detect/test/images/*.png"))
    if not test_imgs:
        print("❌ 没有测试图片")
        sys.exit(1)
    results = model.predict(
        source=test_imgs,
        conf=0.25,
        save=True,                       # 自动保存带框结果图
        project="results_detect",
        name="pred",
        exist_ok=True,
    )
    print("\n===== 推理结果 =====")
    for r in results:
        name = r.path.split("/")[-1].split("\\")[-1]
        n = len(r.boxes) if r.boxes is not None else 0
        if n:
            for b in r.boxes:
                cls = model.names[int(b.cls)]
                print(f"  {name}: 类别={cls}  置信度={float(b.conf):.2f}  "
                      f"框={b.xywh[0].tolist()}")
        else:
            print(f"  {name}: 未检出目标")
    print(f"\n结果图已保存 -> results_detect/pred/")
