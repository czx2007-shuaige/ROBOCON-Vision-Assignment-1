#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 captures/ 里 CVAT 导出的标注切分成 YOLO 数据集（8:1:1），排除空标注。"""
import os, glob, shutil, random

CAP = "captures"
OUT = "dataset_detect"
random.seed(42)

# 收集有有效标注的 (png, txt) 对
pairs = []
for t in sorted(glob.glob(f"{CAP}/*_color.txt")):
    if not open(t).read().strip():
        continue  # 空标注（漏标）→ 排除
    png = t.replace("_color.txt", "_color.png")
    if os.path.exists(png):
        pairs.append((png, t))

random.shuffle(pairs)
n = len(pairs)
ntr, nval = int(n * 0.8), int(n * 0.1)
splits = {
    "train": pairs[:ntr],
    "val":   pairs[ntr:ntr + nval],
    "test":  pairs[ntr + nval:],
}

shutil.rmtree(OUT, ignore_errors=True)
for sp, items in splits.items():
    os.makedirs(f"{OUT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{sp}", exist_ok=True)
    for png, txt in items:
        shutil.copy(png, f"{OUT}/images/{sp}/")
        shutil.copy(txt, f"{OUT}/labels/{sp}/")

with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("nc: 1\n")
    f.write("names: ['block']\n")

print(f"有效标注图: {n}  (已排除空标注)")
print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
print(f"✅ 数据集已生成: {OUT}/")
