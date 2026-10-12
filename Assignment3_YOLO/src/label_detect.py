#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半自动矩形框标注（YOLO detect 格式）

先用【深度高度法/颜色法】自动生成候选框，人工确认或拖框修正，比纯手动画快很多。
操作: 空格=确认 鼠标拖框=重新框  r=清除  s=跳过  q=退出(可断点续标)
输出: dataset_detect/{train,val,test}/{images,labels} + data.yaml
"""
import os, sys, glob, json, random, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_pose as bp

CAP_DIR, OUT_DIR, IMG_EXT = "captures", "dataset_detect", "_color.png"


def auto_box(g, K, D, maps):
    """高度法优先，颜色法回退，返回 ([x1,y1,x2,y2], 方法名) 或 (None, None)。

    关键：方块半红半蓝、交处有阴影带，掩码常断成多块。
    必须【合并所有候选轮廓取并集包围框】，否则只会框住最大的一块
    （红面或蓝面），漏掉另一半——这正是第一版标注框偏小 2.68× 的根因。
    """
    color, depth, meta = g
    ds = meta["depth_scale"]
    d16 = depth.astype(np.uint16)

    def union_bbox(mask):
        # 与主循环一致：先 9×9×2 膨胀桥接，再合并所有轮廓取并集框
        md = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), 2)
        cs, _ = cv2.findContours(md, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) >= 400]
        if not cs:
            return None
        x1 = min(cv2.boundingRect(c)[0] for c in cs)
        y1 = min(cv2.boundingRect(c)[1] for c in cs)
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in cs)
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in cs)
        return [x1, y1, x2, y2]

    # 1) 深度高度法
    try:
        gh = bp.height_ground_and_mask(d16, ds, K, D, maps=maps)
        if gh:
            (_, _), mask = gh
            b = union_bbox(mask)
            if b:
                return b, "height"
    except Exception:
        pass
    # 2) 颜色法（红∪蓝）
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in bp.HSV_PRESETS["3"][1]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    b = union_bbox(m)
    if b:
        return b, "color"
    return None, None


def annotate(img, init, idx, total, name):
    """单张标注交互，返回 box 或 'SKIP' / 'QUIT'"""
    box = list(init) if init else None
    state = {"drawing": False, "p0": None}

    def on_mouse(e, x, y, f, p):
        nonlocal box
        if e == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["p0"] = (x, y)
            box = [x, y, x, y]
        elif e == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x0, y0 = state["p0"]
            box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        elif e == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            x0, y0 = state["p0"]
            box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]

    win = "标注 (空格=确认 拖框=重画 r=清除 s=跳过 q=退出)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = img.copy()
        if box:
            cv2.rectangle(disp, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(disp, f"[{idx}/{total}] {name}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(disp, "ENTER=ok  drag=redraw  r=clear  s=skip  q=quit",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):                       # 回车 / 空格
            if box and (box[2] - box[0]) > 4 and (box[3] - box[1]) > 4:
                cv2.destroyWindow(win)
                return box
        elif k == ord('r'):
            box = None
        elif k == ord('s'):
            cv2.destroyWindow(win)
            return "SKIP"
        elif k == ord('q'):
            cv2.destroyWindow(win)
            return "QUIT"


def yolo_line(cls_id, box, w, h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="block", help="类别名")
    ap.add_argument("--cap", default=CAP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--auto-all", action="store_true",
                    help="一键模式：全部采用自动生成的框，不逐张人工确认（快速出数据集）")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.cap, "*" + IMG_EXT)))
    if not imgs:
        print(f"❌ 没找到 {args.cap}/*{IMG_EXT}")
        sys.exit(1)

    # 标定内参（去畸变用）
    calib = None
    for sn, p in bp.CALIB_FILES.items():
        if os.path.exists(p):
            calib = p
            break
    K = D = maps = None
    first = cv2.imread(imgs[0])
    if first is not None:
        H, W = first.shape[:2]
        K = np.array([[605.0, 0, W / 2], [0, 605.0, H / 2], [0, 0, 1.0]])
        if calib:
            c = np.load(calib)
            K = c["K"]
            D = c["D"].reshape(-1, 1) if c["D"].ndim == 1 else c["D"]
        maps = bp.build_undistort_maps(K, D, W, H) if D is not None else None

    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, s, "labels"), exist_ok=True)

    # 已标注（断点续标）
    done = set()
    for s in ("train", "val", "test"):
        for p in glob.glob(os.path.join(args.out, s, "labels", "*.txt")):
            done.add(os.path.basename(p).replace(".txt", ".png"))
    todo = [p for p in imgs if os.path.basename(p) not in done]
    print(f"共 {len(imgs)} 张，已标注 {len(done)}，待标注 {len(todo)}，类别 '{args.cls}'")

    boxes = {}
    nauto = 0
    for i, path in enumerate(todo, 1):
        stem = os.path.basename(path).replace(IMG_EXT, "")
        depth_p = path.replace(IMG_EXT, "_depth.npy")
        meta_p = path.replace(IMG_EXT, "_meta.json")
        img = cv2.imread(path)
        init, how = None, None
        if os.path.exists(depth_p) and os.path.exists(meta_p) and K is not None:
            depth = np.load(depth_p)
            with open(meta_p) as f:
                meta = json.load(f)
            init, how = auto_box((img, depth, meta), K, D, maps)
            if init:
                nauto += 1
        if args.auto_all:
            if init:
                boxes[path] = init
            continue
        r = annotate(img, init, i, len(todo), os.path.basename(path))
        if r == "QUIT":
            print("👋 退出，已标注内容会保存")
            break
        if r == "SKIP" or r is None:
            continue
        boxes[path] = r

    if not boxes:
        print("没有新增标注")
        return

    keys = sorted(boxes.keys())
    random.seed(42)
    random.shuffle(keys)
    n = len(keys)
    ntr, nva = int(n * 0.8), int(n * 0.1)
    split = {}
    for j, k in enumerate(keys):
        split[k] = "train" if j < ntr else ("val" if j < ntr + nva else "test")

    for path in keys:
        s = split[path]
        stem = os.path.basename(path).replace(IMG_EXT, "")
        img = cv2.imread(path)
        h, w = img.shape[:2]
        cv2.imwrite(os.path.join(args.out, s, "images", stem + ".png"), img)
        with open(os.path.join(args.out, s, "labels", stem + ".txt"), "w") as f:
            f.write(yolo_line(0, boxes[path], w, h))

    yml = os.path.join(args.out, "data.yaml")
    with open(yml, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n\n")
        f.write("names:\n  0: " + args.cls + "\n")

    cnt = {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "test")}
    print("\n" + "=" * 58)
    print(f"✅ 新增 {len(keys)} 张（自动预标注 {nauto} 张）")
    print(f"   train {cnt['train']} / val {cnt['val']} / test {cnt['test']}")
    print(f"   数据集: {os.path.abspath(args.out)}")
    print(f"   配置:   {yml}")
    print("=" * 58)
    print("\n训练命令:")
    print(f"  yolo detect train model=yolo11n.pt data={yml} epochs=80 imgsz=640 device=0")


if __name__ == "__main__":
    main()
