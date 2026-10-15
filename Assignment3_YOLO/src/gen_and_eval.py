#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 test 集验证训练结果（推理框 vs CVAT 标注框 IoU + 漏检），并生成 result.jpg。"""
import glob, os, numpy as np, cv2, sys
sys.path.insert(0, ".")
from ultralytics import YOLO

m = YOLO("runs/detect/block_yolo11n/weights/best.pt")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - iw * ih + 1e-9)


ious, miss, best = [], [], None
for imgp in sorted(glob.glob("dataset_detect/images/test/*.png")):
    stem = os.path.basename(imgp).replace(".png", "")
    labp = f"dataset_detect/labels/test/{stem}.txt"
    img = cv2.imread(imgp)
    H, W = img.shape[:2]
    gts = []
    for line in open(labp):
        p = line.split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = map(float, p[:5])
        gts.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                    int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
    res = m.predict(img, conf=0.25, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        miss.append(stem)
        continue
    xyxy = res.boxes.xyxy.cpu().numpy().reshape(-1, 4)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    bi = int(np.argmax(area))
    pred = tuple(map(int, xyxy[bi]))
    gt = gts[int(np.argmax([(g[2] - g[0]) * (g[3] - g[1]) for g in gts]))]
    i = iou(pred, gt)
    ious.append(i)
    if best is None or i > best[0]:
        best = (i, stem, img.copy(), pred)

print(f"test 共 {len(ious) + len(miss)} 张: 漏检 {len(miss)} -> {miss or '无'}")
if ious:
    print(f"IoU(推理框/标注框): 中位 {np.median(ious):.3f}  最小 {min(ious):.3f}")

os.makedirs("results_detect", exist_ok=True)
if best:
    i, stem, img, pred = best
    cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), (0, 255, 0), 3)
    cv2.putText(img, "block", (pred[0], max(pred[1] - 12, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite("results_detect/result.jpg", img)
    print(f"✅ result.jpg 来自 {stem} (IoU={i:.3f})")
