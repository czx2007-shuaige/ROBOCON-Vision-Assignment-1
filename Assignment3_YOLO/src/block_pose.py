#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 队内赛 任务一 · 基础要求：静止方块的世界坐标位置 + yaw 角
------------------------------------------------------------------------------
 场景：相机固定于世界坐标系原点，观察静置在地面上的方块。
 输出：方块在世界坐标系下的位置 (x, y, z) 与 yaw 角。

 世界坐标系定义（自洽，且可用卷尺核对）：
   - 原点  = 相机 RGB 光心（题目：相机固定于世界坐标系原点）
   - Z 轴  = 竖直向上（由深度数据 RANSAC 拟合的地面平面法线确定）
   - X 轴  = 相机"右"方向投影到地面平面
   - Y 轴  = Z × X（右手系）
   - 单位  = 米；yaw 单位 = 度，绕竖直轴（Z）旋转，逆时针为正

 原理：
   1) 颜色分割得到方块掩码 → 掩码内像素 + 对齐深度 → 三维点云
   2) 非方块区域的深度点 RANSAC 拟合 → 地面平面（得到竖直方向）
   3) 方块所有可见点沿重力投影到地面 = 方块足印（顶面是足印的竖直平移，
      侧面投影只贡献边界线）→ 对足印做 minAreaRect
      → 中心(x,y)、尺寸(w×h)、朝向角 = yaw
   4) 方块高度 = 点云高度 95 分位 → 中心 z = 地面 + h/2

 操作（实时窗口）：
   鼠标左键点击方块 → 自动采样颜色（要求 S≥100，避免误取同色地面）
   1 → 红方块        2 → 蓝方块
   3 → 天空块(红+蓝合并, 200mm 塔顶专用)
   4 → 五色石(金)    5 → 建塔处(绿)
   p → 输出并保存当前结果（自动取最近 15 帧中位数，抗抖动）
   g → 重新估计地面
   q → 退出

 ⚠ 当前道具：只有天空块(200mm)。规则中塔顶是"半红半蓝"立方体，
   单色掩码可能只框住一半 → 请用 3 号"红+蓝合并"预设最稳；
   若顶面恰好是单一颜色，用 1 或 2 也可。建议加 --cube-size 0.20。

 女娲补天规则中的方块（立方体瓦楞纸盒）：
   地基块 = 塔基/中段：350 × 350 × 350 mm，红 223-34-34 / 蓝 50-0-255
   天空块 = 塔顶：     200 × 200 × 200 mm，半红半蓝
 场地地面为低饱和浅红(240-210-210)/浅蓝(170-210-230)，
 而方块为高饱和纯色，故用饱和度 S≥100 即可把方块与同色地面分离。

 用法：
   python3 block_pose.py                          # 自动识别方块类型
   python3 block_pose.py --sn 043322075459        # 指定相机
   python3 block_pose.py --cube-size 0.35         # 指定为地基块(350mm)
   python3 block_pose.py --cube-size 0.20         # 指定为天空块(200mm)
==============================================================================
"""

import os
import sys
import time
import argparse
import collections
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception:
    print("❌ 需要安装 pyrealsense2")
    sys.exit(1)

# ----------------------------- 参数 -----------------------------
IMG_W, IMG_H = 640, 480
MIN_AREA = 400          # 方块掩码最小面积(px)
MASK_STEP = 2           # 方块点云采样步长
GROUND_STEP = 8         # 地面点云采样步长
RANSAC_ITERS = 100      # 迭代次数（地面这类大平面 100 次足够；300 次会拖慢帧率）
RANSAC_TH = 0.012       # 地面内点阈值(m)
RANSAC_SUBSET = 2000    # RANSAC 工作集点数上限（超过则随机子采样，提速关键）
TOP_PCT = 95            # 高度分位数
SMOOTH_N = 15           # 输出取中位数的帧数

# ================== 女娲补天 官方配色（附录1 RGB → OpenCV HSV 实测） ==================
#   塔基/中段(红) 223-34-34   -> HSV (  0, 216, 223)
#   塔基/中段(蓝) 50-0-255    -> HSV (126, 255, 255)
#   五色石(金)    218-165-32  -> HSV ( 21, 218, 218)
#   地面区(红)    240-210-210 -> HSV (  0,  32, 240)   ← 同色系地面
#   地面区(蓝)    170-210-230 -> HSV (100,  67, 230)   ← 同色系地面
#
# 关键：方块饱和度 S≈216/255，而同色系地面 S 只有 32/67。
#       因此把饱和度下限设为 100，即可把方块与同色地面干净分离。
SAT_MIN = 100           # 饱和度下限（区分高饱和方块与低饱和地面）
VAL_MIN = 60            # 明度下限

# HSV 预设 (名称, [(H,S,V下限), (H,S,V上限)])，红色跨 0 需拆两段
#   天空块(塔顶) 是"半红半蓝"立方体，单色掩码只能框住一半，
#   所以专门提供"红∪蓝合并"预设：红蓝并集 = 整个方块
_PRE_RED   = [((0, SAT_MIN, VAL_MIN), (8, 255, 255)),
              ((172, SAT_MIN, VAL_MIN), (180, 255, 255))]
_PRE_BLUE  = [((118, SAT_MIN, VAL_MIN), (136, 255, 255))]
HSV_PRESETS = {
    "1": ("红方块", _PRE_RED),
    "2": ("蓝方块", _PRE_BLUE),
    "3": ("天空块(红+蓝合并,塔顶200mm)", _PRE_RED + _PRE_BLUE),
    "4": ("五色石(金)", [((14, 140, 80), (30, 255, 255))]),
    "5": ("建塔处(绿)", [((55, 100, 40), (75, 255, 255))]),
}

# 方块尺寸先验（男娲补天规则：立方体瓦楞纸盒）
#   地基块 = 塔基/中段：350 × 350 × 350 mm
#   天空块 = 塔顶：     200 × 200 × 200 mm
CUBE_SPECS = {
    0.35: "地基块(塔基/中段) 350mm",
    0.20: "天空块(塔顶) 200mm",
}
SIZE_TOL = 0.30         # 测得边长与先验的相对容差，超出则判定误检


# ----------------------------- 工具 -----------------------------
def ransac_plane(pts, iters=RANSAC_ITERS, th=RANSAC_TH, seed=0,
                 max_pts=RANSAC_SUBSET):
    """RANSAC 拟合平面。返回 (单位法线, 平面上一点, 内点掩码) 或 None。

    性能版（原版 300 次迭代 × 全量点要 595ms，只有 1.8 FPS）：
      1) 先随机子采样到 max_pts 个点再做 RANSAC（点数越多单次越贵）
      2) 迭代次数降到 100（地面这类大平面少量迭代就够）
      3) 用 rng.integers 取代 rng.choice(replace=False)（后者很慢）
      4) 最后用全量点重新评估内点 + SVD 精化，保证精度不降
    """
    n = len(pts)
    if n < 60:
        return None
    rng = np.random.default_rng(seed)
    # 1) 子采样工作集
    if n > max_pts:
        work = pts[rng.choice(n, max_pts, replace=False)]
    else:
        work = pts
    m = len(work)
    best = None
    for _ in range(iters):
        i = rng.integers(0, m, 3)           # 3) 快速取 3 个索引
        p1, p2, p3 = work[i]
        nrm = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        d = np.abs((work - p1) @ nrm)
        c = int((d < th).sum())
        if best is None or c > best[0]:
            best = (c, nrm, p1)
    if best is None:
        return None
    # 4) 用全量点评估内点 + 精化法向
    _, nrm, p1 = best
    inl = np.abs((pts - p1) @ nrm) < th
    if int(inl.sum()) < 0.25 * n:
        return None
    P = pts[inl]
    ctr = P.mean(axis=0)
    # 关键性能点：不要对 (N,3) 做 SVD（N~6000 时要 400+ms）！
    # 数学上等价的更快做法：构造 3x3 协方差矩阵，再做特征分解，
    # 最小特征值对应的特征向量就是平面法向。
    C = (P - ctr).T @ (P - ctr)          # 3x3
    _, vecs = np.linalg.eigh(C)          # 3x3 特征分解，极快
    nrm2 = vecs[:, 0]                    # eigh 特征值升序 → 第0个最小
    if nrm2 @ nrm < 0:                   # 保持与粗估计同向
        nrm2 = -nrm2
    return nrm2, ctr, inl


def build_world_basis(up, cam_x=np.array([1.0, 0, 0])):
    """由竖直方向构造世界基：Z=up, X=cam_x 投影到地面, Y=Z×X。"""
    Zw = up / np.linalg.norm(up)
    Xw = cam_x - np.dot(cam_x, Zw) * Zw
    nx = np.linalg.norm(Xw)
    if nx < 1e-6:                                # 相机右方向几乎竖直时退化
        cam_y = np.array([0.0, 1.0, 0])
        Xw = cam_y - np.dot(cam_y, Zw) * Zw
        nx = np.linalg.norm(Xw)
        if nx < 1e-6:
            return None
    Xw = Xw / nx
    Yw = np.cross(Zw, Xw)
    return Xw, Yw, Zw


def rot_rect_angle_norm(angle_deg, side_ratio):
    """minAreaRect 角度归一，并归到对称区间，避免 0°/90° 边界处跳变。

    - 正方形/立方体足印（边长比≈1）：绕竖直轴有 90° 对称 → 归到 [-45, 45)
      （否则 yaw=0° 会被报成 90°，评分时会被判成大误差）
    - 长方形足印：有 180° 对称 → 归到 [-90, 90)
    """
    a = angle_deg % 180.0
    if side_ratio > 0.85:
        a = a % 90.0
        if a >= 45.0:
            a -= 90.0
    else:
        if a >= 90.0:
            a -= 180.0
    return a


def infer_cube_from_cls(name):
    """由 YOLO 类别名推断方块边长先验（训练时若按类型分类即可自动识别）。"""
    if name is None:
        return None
    s = str(name).lower()
    if "350" in s or "地基" in s or "塔基" in s or "中段" in s or "base" in s:
        return 0.35
    if "200" in s or "天空" in s or "塔顶" in s or "top" in s:
        return 0.20
    return None


# 自标定内参文件（按相机序列号），用于去畸变
CALIB_FILES = {
    "043322075459": "../task2/calib_cam1.npz",
    "213622078104": "../task2/calib_cam2.npz",
}


def deproject_undist(K, D, u, v, Z):
    """像素 + 深度 -> 三维点（相机系）。有 D 时先去畸变。"""
    if D is not None:
        pt = np.array([[[float(u), float(v)]]], np.float32)
        und = cv2.undistortPoints(pt, K, D, P=K)
        u, v = float(und[0, 0, 0]), float(und[0, 0, 1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])


def build_undistort_maps(K, D, w, h):
    """预计算去畸变查找表（启动时算一次）。

    畸变大(k3≈-1.7)时 cv2.undistortPoints 内部要迭代求解，对几万个点
    非常慢(>500ms/帧)。改用 initUndistortRectifyMap 一次性算出映射表，
    之后每帧只需查表(索引)，速度提升几十倍。
    """
    if D is None:
        return None
    mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return (mx, my)


def deproject_batch(K, D, us, vs, Zs, maps=None):
    """批量反投影：像素数组 + 深度数组 -> (N,3) 三维点。

    maps 为 build_undistort_maps 的结果时走快速查表路径。
    """
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    Zs = np.asarray(Zs, dtype=np.float64)
    if len(us) == 0:
        return np.zeros((0, 3))
    if maps is not None:
        mx, my = maps
        H, W = mx.shape[:2]
        ui = np.clip(np.rint(us).astype(np.int64), 0, W - 1)
        vi = np.clip(np.rint(vs).astype(np.int64), 0, H - 1)
        u = mx[vi, ui].astype(np.float64)      # 查表 O(1)
        v = my[vi, ui].astype(np.float64)
    elif D is not None:
        pts = np.stack([us, vs], axis=1).astype(np.float32).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, D, P=K)      # 一次调用处理全部
        u = und[:, 0, 0].astype(np.float64)
        v = und[:, 0, 1].astype(np.float64)
    else:
        u, v = us, vs
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u - cx) * Zs / fx, (v - cy) * Zs / fy, Zs], axis=1)


def sample_points_from_mask(block_mask, depth_raw, ds, step,
                           zmin, zmax, K, D, maps=None):
    """从掩码区域批量采样三维点：返回 (N,3)，点数不足返回空数组。

    用 depth_raw[ys, xs] 数组索引一次性取深度（替代逐点 get_distance），
    再一次性批量反投影，速度快几十倍。
    """
    ys, xs = np.nonzero(block_mask[::step, ::step])
    if len(xs) == 0:
        return np.zeros((0, 3))
    xs = (xs * step).astype(np.int64)
    ys = (ys * step).astype(np.int64)
    Z = depth_raw[ys, xs].astype(np.float32) * ds
    ok = (Z > zmin) & (Z < zmax)
    if int(ok.sum()) == 0:
        return np.zeros((0, 3))
    return deproject_batch(K, D, xs[ok], ys[ok], Z[ok].astype(np.float64), maps)


def height_ground_and_mask(depth_raw, ds, K, D, hmin=0.08, hmax_th=0.40,
                           maps=None):
    """【深度高度法】不依赖颜色，直接用"离地高度"定位方块。

    原理：地面拟合后，方块是唯一"高出地面 8~40cm"的东西。
    优点：不看颜色 → 红面/蓝面朝上都行；同色地面、红色启动区
          （漆面高度=0）都会被高度过滤自动排除；且每帧独立拟合
          地面 → 手持或移动相机也能用。

    返回 ((nrm, p0), mask) 或 None。
    """
    H, W = depth_raw.shape

    # ---- 1) 地面：全图点云 RANSAC 找两个平面，离相机更远的为地面 ----
    vs, us = np.mgrid[0:H:6, 0:W:6]
    us, vs = us.ravel(), vs.ravel()
    Z = depth_raw[vs, us].astype(np.float32) * ds
    ok = (Z > 0.1) & (Z < 5.0)
    if int(ok.sum()) < 300:
        return None
    us, vs, Z = us[ok], vs[ok], Z[ok]
    pts = deproject_batch(K, D, us, vs, Z, maps)
    f1 = ransac_plane(pts)
    if f1 is None:
        return None
    n1, p1, inl = f1
    rest = pts[~inl]
    cands = [(n1, p1)]
    if len(rest) > 300:                       # 场景里通常还有"方块顶面"这个平面
        f2 = ransac_plane(rest, iters=60)
        if f2 is not None:
            cands.append((f2[0], f2[1]))
    # 相机(原点)到平面的距离越大 → 离相机越远 → 是地面（方块顶更近）
    cands.sort(key=lambda t: abs(float(np.dot(t[0], t[1]))), reverse=True)
    nrm, p0 = cands[0]
    # 法线定向：相机一定在地面之上
    if (np.array([0.0, 0, 0]) - p0) @ nrm < 0:
        nrm = -nrm

    # ---- 2) 高度图 -> 方块掩码 ----
    vs2, us2 = np.mgrid[0:H:2, 0:W:2]
    Z2 = depth_raw[vs2, us2].astype(np.float32) * ds
    ok2 = (Z2 > 0.1) & (Z2 < 5.0)
    if int(ok2.sum()) < 500:
        return None
    u2, v2, z2 = us2[ok2], vs2[ok2], Z2[ok2]
    P2 = deproject_batch(K, D, u2, v2, z2, maps)
    h2 = P2 @ nrm - float(p0 @ nrm)
    mask = np.zeros((H, W), np.uint8)
    band = (h2 > hmin) & (h2 < hmax_th)
    mask[v2[band], u2[band]] = 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return (nrm, p0), mask


def yolo_mask(img, model, conf=0.5, imgsz=640):
    """用训练好的 YOLO 模型得到方块掩码。同时兼容【分割模型】和【检测模型】。

    返回 (mask_uint8, 类别名)。
    - 分割模型：取面积最大的实例掩码（精确贴合方块轮廓）
    - 检测模型：把面积最大的检测框整块填成掩码（队内赛训练的 yolo11n
      是检测模型，无 mask）。框内地面像素会被下游"离地 5cm"过滤掉，
      所以填框当成掩码也能正确算出足印。
    没有检测到时返回全零掩码和 None。
    """
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)
    r = res[0]
    h, w = img.shape[:2]
    names = r.names if hasattr(r, "names") else {}

    # 1) 分割模型优先：取面积最大的实例掩码
    if r.masks is not None and len(r.masks.data) > 0:
        best, best_area, best_name = None, 0, None
        for i, m in enumerate(r.masks.data):
            arr = m.cpu().numpy()
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            b = (arr > 0.5).astype(np.uint8)
            a = int(b.sum())
            if a > best_area:
                best_area, best = a, b
                cls_id = int(r.boxes.cls[i]) if r.boxes is not None and i < len(r.boxes.cls) else None
                best_name = names.get(cls_id, str(cls_id)) if cls_id is not None else None
        if best is not None:
            return (best * 255).astype(np.uint8), best_name

    # 2) 检测模型回退：把面积最大的框填成掩码
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        bi = int(np.argmax(areas))
        x1, y1, x2, y2 = xyxy[bi].astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        m = np.zeros((h, w), np.uint8)
        m[y1:y2, x1:x2] = 255
        cls_id = int(r.boxes.cls[bi])
        return m, names.get(cls_id, str(cls_id))

    return np.zeros((h, w), np.uint8), None


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", default=None, help="相机序列号（默认第一台）")
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--cube-size", type=float, default=None,
                    help="方块边长先验(米)：0.35=地基块, 0.20=天空块；"
                         "不填则自动按测得尺寸就近判断类型")
    ap.add_argument("--mode", default="height", choices=["height", "color", "yolo"],
                    help="方块分割方式：height=深度高度法(默认，不依赖颜色，"
                         "可手持)、color=HSV颜色、yolo=训练好的模型(检测/分割均可)")
    ap.add_argument("--calib", default=None,
                    help="自标定内参 npz（用于去畸变）；默认按相机序列号自动匹配")
    ap.add_argument("--model", default="runs/detect/block_yolo11n/weights/best.pt",
                    help="YOLO 分割模型 .pt 路径（--mode yolo 时使用）")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="YOLO 置信度阈值（默认 0.5）")
    args = ap.parse_args()

    # ---------- 相机 ----------
    devs = list(rs.context().query_devices())
    if not devs:
        print("❌ 未检测到 RealSense 相机")
        sys.exit(1)
    serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
    sn = args.sn if args.sn else serials[0]
    if sn not in serials:
        print(f"❌ SN {sn} 不在 {serials}")
        sys.exit(1)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    align = rs.align(rs.stream.color)

    # ---------- YOLO 模型（仅 --mode yolo 时加载） ----------
    yolo_model = None
    if args.mode == "yolo":
        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"❌ 未安装 ultralytics（请先 pip install ultralytics）: {e}")
            sys.exit(1)
        if not os.path.exists(args.model):
            print(f"❌ 找不到模型文件: {args.model}")
            print("   训练好后用 --model /路径/best.pt 指定，"
                  "或先建 models/ 目录把权重放进去")
            sys.exit(1)
        yolo_model = YOLO(args.model)
        print(f"✅ YOLO 模型已加载: {args.model}  (classes={yolo_model.names})")

    started = False
    click_hsv = None            # 点击采样的 HSV
    click_marker = None         # 点击落点标记（诊断坐标偏移用）
    preset = None               # 当前预设键（手动模式）
    auto_mode = True            # 自动模式：直接用红∪蓝合并掩码，无需点击
    debug = False               # 调试开关（按 d）：打印每个候选被剔除的原因
    undist_maps = None          # 去畸变查找表（首帧惰性构建）
    ground = None               # (nrm, p0) 地面
    results = collections.deque(maxlen=SMOOTH_N)
    frame_id = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal click_hsv, click_marker, auto_mode
        if event == cv2.EVENT_LBUTTONDOWN and param is not None:
            img_h, img_w = param.shape[:2]
            # 高分屏(系统缩放)或窗口被拉伸时，OpenCV 回传的鼠标坐标可能与
            # 图像像素坐标存在比例偏差；用窗口实际显示区域做映射校正
            try:
                rx, ry, rw, rh = cv2.getWindowImageRect("block_pose")
                if rw > 0 and rh > 0 and (rw != img_w or rh != img_h):
                    nx = int((x - rx) * img_w / rw)
                    ny = int((y - ry) * img_h / rh)
                    print(f"🖱 坐标校正: 原始({x},{y}) -> 图像({nx},{ny})  "
                          f"[窗口显示 {rw}x{rh}, 图像 {img_w}x{img_h}]")
                    x, y = nx, ny
            except Exception:
                pass
            x = min(max(x, 0), img_w - 1)
            y = min(max(y, 0), img_h - 1)
            click_marker = (x, y)                 # 在画面上画出落点，便于核对
            h0, s0, v0 = (int(v) for v in param[y, x])
            # 要求足够饱和：场地地面是低饱和的浅红/浅蓝(S≈32/67)，
            # 方块是高饱和纯色(S≈216/255)，采样时强制 S>=SAT_MIN 避免取到地面
            if s0 >= SAT_MIN and v0 >= VAL_MIN:
                click_hsv = (h0, s0, v0)
                auto_mode = False
                print(f"🎨 采样颜色 HSV=({h0},{s0},{v0})  (S={s0} ≥ {SAT_MIN} ✅，已切到手动模式)")
            else:
                print(f"⚠ 点击落点 ({x},{y}) 饱和度不足 HSV=({h0},{s0},{v0}) —— "
                      f"十有八九是高分屏坐标偏移或点到了地面。"
                      f"画面上的白圈就是这次点击实际落点；"
                      f"建议直接用 AUTO 模式(按 a)或预设 3，无需点击")

    # ---------- 去畸变用的标定内参 ----------
    calib_path = args.calib
    if calib_path is None:
        calib_path = CALIB_FILES.get(sn)
    Dvec = None
    if calib_path and os.path.exists(calib_path):
        _c = np.load(calib_path)
        Dvec = _c["D"].reshape(-1, 1) if _c["D"].ndim == 1 else _c["D"]
        print(f"✅ 载入标定畸变参数: {calib_path}  D={np.round(Dvec.ravel(), 4)}")
    else:
        print("ℹ 未找到自标定文件，将不做去畸变（精度略降）")

    try:
        try:
            _prof = pipe.start(cfg)
            started = True
        except Exception as e:
            print(f"❌ 打开相机失败: {e}")
            return
        try:
            depth_scale = _prof.get_device().first_depth_sensor().get_depth_scale()
        except Exception:
            depth_scale = 0.001
        print(f"相机 SN={sn} 已启动  模式={args.mode}")
        print(f"  height=深度高度法(默认,可手持)  color=HSV颜色  yolo=分割模型")
        print(f"  按键: p=输出结果  g=重估地面  d=调试  a=切AUTO颜色  q=退出")

        cv2.namedWindow("block_pose")
        # OpenCV5/Qt 后端要等首次 imshow+waitKey 后窗口句柄才真正创建，
        # 否则循环里的 setMouseCallback 会报 "NULL window handler"
        cv2.imshow("block_pose", np.zeros((IMG_H, IMG_W, 3), np.uint8))
        cv2.waitKey(30)
        while True:
            frames = align.process(pipe.wait_for_frames())
            df = frames.get_depth_frame()
            cf = frames.get_color_frame()
            if not df or not cf:
                continue
            color = np.asanyarray(cf.get_data())
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            intr = df.profile.as_video_stream_profile().intrinsics
            depth_raw = np.asanyarray(df.get_data())
            Kmat = np.array([[intr.fx, 0, intr.ppx],
                             [0, intr.fy, intr.ppy],
                             [0, 0, 1.0]], dtype=np.float64)
            if undist_maps is None and Dvec is not None:
                # 首帧构建一次去畸变查找表，之后每帧只查表
                undist_maps = build_undistort_maps(Kmat, Dvec, IMG_W, IMG_H)
                print("✅ 去畸变查找表已构建（后续每帧查表，保证帧率）")

            # ---------- 1. 掩码：深度高度法 / YOLO / 颜色 ----------
            cube_hint = None
            if args.mode == "height":
                gh = height_ground_and_mask(depth_raw, depth_scale, Kmat,
                                              Dvec, maps=undist_maps)
                if gh is None:
                    mask = np.zeros(hsv.shape[:2], np.uint8)
                    ground = None
                    mode_txt = "HEIGHT 深度高度法：未找到地面/方块"
                else:
                    # 每帧独立拟合地面 → 手持或移动相机都能用
                    ground, mask = gh
                    mode_txt = "HEIGHT 深度高度法（不依赖颜色，可手持）"
            elif yolo_model is not None:
                mask, cls_name = yolo_mask(color, yolo_model, args.conf)
                cube_hint = infer_cube_from_cls(cls_name)
                mode_txt = f"YOLO {os.path.basename(args.model)} conf={args.conf}"
                if cls_name is not None:
                    mode_txt += f"  cls={cls_name}"
            elif auto_mode:
                # AUTO：直接用"红∪蓝合并"掩码，无需点击。
                # 高饱和红/蓝区域 = 方块（同色浅地面 S 只有 32/67，会被滤掉）
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for lo, hi in HSV_PRESETS["3"][1]:
                    mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
                mode_txt = "AUTO 红∪蓝合并 (a=切换自动/手动)"
            elif click_hsv is not None:
                h0, s0, v0 = click_hsv
                # 饱和度下限用 SAT_MIN，明度下限用 VAL_MIN：
                # 这样同色的浅色地面(S≈32/67)不会混进来
                lo1 = np.array([max(0, h0 - 12), SAT_MIN, VAL_MIN])
                hi1 = np.array([min(180, h0 + 12), 255, 255])
                mask = cv2.inRange(hsv, lo1, hi1)
                if h0 - 12 < 0:                       # 色相绕回
                    mask |= cv2.inRange(hsv,
                                        np.array([180 + h0 - 12, SAT_MIN, VAL_MIN]),
                                        np.array([180, 255, 255]))
                if h0 + 12 > 180:
                    mask |= cv2.inRange(hsv,
                                        np.array([0, SAT_MIN, VAL_MIN]),
                                        np.array([h0 + 12 - 180, 255, 255]))
                mode_txt = f"click HSV h={h0} (S>={SAT_MIN})"
            elif preset is not None:
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for lo, hi in HSV_PRESETS[preset][1]:
                    mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
                mode_txt = f"preset {HSV_PRESETS[preset][0]}"
            else:
                mask = np.zeros(hsv.shape[:2], np.uint8)
                mode_txt = "未选颜色：点击方块 或 按 1/2/3/4"

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

            # ---- 膨胀桥接：让"半红半蓝"方块的两个颜色区域连成一个轮廓 ----
            # 两个面交界处有一条阴影过渡带（既不够红也不够蓝），直接找轮廓
            # 会断成两块，绿框就只框得住单色。先膨胀把近距区域连起来再找轮廓。
            # 溢出到地板的点随后会被"离地>3cm"过滤器清除，所以这里放心膨胀。
            mask_d = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
            cnts, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= args.min_area]
            cnts.sort(key=cv2.contourArea, reverse=True)
            frame_id += 1
            if debug and frame_id % 15 == 0:
                print(f"[debug] 掩码像素={int(mask.sum() / 255)}  "
                      f"有效轮廓数={len(cnts)}  模式={'AUTO' if auto_mode else '手动'}")

            vis = color.copy()
            res = None

            if cnts:
                # 所有候选轮廓都画出来：灰=候选，绿=最终选中的
                cv2.drawContours(vis, cnts[:5], -1, (160, 160, 160), 1)

                # ---------- 2. 地面平面（排除所有候选区域，只估一次） ----------
                all_mask = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(all_mask, cnts, -1, 255, -1)
                if ground is None:
                    # 批量采样（替代逐点 get_distance，速度快几十倍）
                    gp = sample_points_from_mask(
                        (all_mask == 0).astype(np.uint8) * 255,
                        depth_raw, depth_scale, GROUND_STEP,
                        0.1, 5.0, Kmat, Dvec, undist_maps)
                    if len(gp) >= 200:
                        fit = ransac_plane(np.asarray(gp))
                        if fit is not None:
                            nrm, p0, _ = fit
                            # 相机一定在地面之上：用"原点相对平面的位置"定向法线，
                            # 不再依赖方块（否则方块误检时会把上下搞反）
                            if (np.array([0.0, 0, 0]) - p0) @ nrm < 0:
                                nrm = -nrm
                            ground = (nrm, p0)
                            print(f"🌍 地面已估计  法线={np.round(nrm, 3)}  "
                                  f"相机离地约 {abs(float(p0 @ nrm)):.3f} m")

                # ---------- 3. 逐候选验证：选第一个满足"立方体先验"的轮廓 ----------
                # （上次误检成 2cm×4cm×47cm 细条，就是因为只看最大轮廓
                #   且没有"足印边长 ≈ 高度"的立方体一致性校验）
                if ground is not None:
                    nrm, p0 = ground
                    W = build_world_basis(nrm)
                    if W is not None:
                        Xw, Yw, Zw = W
                        d0 = float(p0 @ Zw)                 # 地面在 Z 轴上的截距
                        for ci, c in enumerate(cnts[:5]):
                            block_mask = np.zeros(mask.shape, np.uint8)
                            cv2.drawContours(block_mask, [c], -1, 255, -1)
                            # 批量采样（替代逐点 get_distance）
                            P = sample_points_from_mask(
                                block_mask, depth_raw, depth_scale,
                                MASK_STEP, 0.05, 5.0, Kmat, Dvec, undist_maps)
                            if len(P) < 80:
                                if debug: print(f"  [候选{ci}] 深度点太少({len(P)}<80) -> 剔除")
                                continue
                                                            # P 已是 (N,3) 相机系
                            XYZ = np.stack([P @ Xw, P @ Yw, P @ Zw], axis=1)
                            heights = XYZ[:, 2] - d0        # 每个点离地高度
                            hmax = float(np.percentile(heights, TOP_PCT))
                            if hmax < 0.02:
                                if debug: print(f"  [候选{ci}] 高度{hmax*1000:.0f}mm(<20mm) -> 剔除"
                                                f"（地面可能拟合不对，按 g 重估）")
                                continue

                            # ---- 关键：剔除贴地噪声点 ----
                            # 掩码可能从方块边缘"渗"到同色系地板上（木地板红棕色
                            # 部分区域 S>100）。地板点高度≈0，会把足印撑大 25~30%。
                            # 只用离地 5cm 以上的点算足印：顶面投影仍是完整足印，
                            # 侧面(竖直面)投影仍是边界线，几何结论不变。
                            # （5cm 比 3cm 更能滤掉贴地噪声，200mm 方块不受影响）
                            sel = heights > 0.05
                            if np.count_nonzero(sel) < 40:
                                if debug: print(f"  [候选{ci}] 离地3cm以上的点只有"
                                                f"{np.count_nonzero(sel)}个 -> 剔除")
                                continue
                            foot = XYZ[sel][:, :2] * 1000.0  # 足印(mm)，数值稳定
                            hull = cv2.convexHull(foot.astype(np.float32).reshape(-1, 1, 2))
                            (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
                            cx, cy = cx / 1000.0, cy / 1000.0
                            wmm, hmm = min(rw, rh), max(rw, rh)
                            side_xy = (wmm + hmm) / 2000.0  # 足印边长均值(m)

                            # ---- 立方体一致性：足印边长必须 ≈ 高度 ----
                            big, small = max(side_xy, hmax), min(side_xy, hmax)
                            if big < 1e-6 or small / big < 0.6:
                                if debug: print(f"  [候选{ci}] 非立方体形状: 足印{rw:.0f}x{rh:.0f}mm "
                                                f"高{hmax*1000:.0f}mm 比{small/big:.2f}(<0.6) -> 剔除")
                                continue

                            yaw = rot_rect_angle_norm(ang, wmm / max(hmm, 1e-6))
                            side = (side_xy + hmax) / 2.0

                            # ---- 方块尺寸先验校验 + 类型识别 ----
                            # 优先：命令行指定 > YOLO类别推断 > 按测得尺寸就近匹配
                            cube = args.cube_size if args.cube_size else cube_hint
                            if cube is None:
                                # 自动匹配最接近的规格（0.35 地基块 / 0.20 天空块）
                                cube = min(CUBE_SPECS, key=lambda s: abs(s - side))
                            size_err = abs(side - cube) / cube
                            if size_err > SIZE_TOL:
                                if debug: print(f"  [候选{ci}] 形状是立方体但尺寸不符: 边长"
                                                f"{side*1000:.0f}mm vs 先验{cube*1000:.0f}mm "
                                                f"(偏{size_err*100:.0f}% > {SIZE_TOL*100:.0f}%) -> 剔除")
                                continue                    # 与先验尺寸不符 → 误检
                            kind = CUBE_SPECS.get(round(cube, 2), f"{cube*1000:.0f}mm")
                            # z 用已知边长先验更稳（深度噪声不影响先验）
                            z_world = d0 + cube / 2.0
                            center_cam = Xw * cx + Yw * cy + Zw * z_world
                            res = dict(x=cx, y=cy, z=z_world, yaw=yaw,
                                       w=wmm / 1000.0, l=hmm / 1000.0,
                                       h=hmax, side=side, cube=cube,
                                       kind=kind, size_err=size_err,
                                       cand=ci, cam=center_cam)
                            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)  # 选中=绿
                            if debug: print(f"  [候选{ci}] ✅ 接受: 足印{rw:.0f}x{rh:.0f}mm"
                                            f"(均值{side_xy*1000:.0f}mm) 高{hmax*1000:.0f}mm "
                                            f"-> 边长{side*1000:.0f}mm 偏差{size_err*100:.1f}%")
                            break
            if res is not None:
                results.append(res)

            # ---------- 4. 显示 ----------
            cv2.setMouseCallback("block_pose", on_mouse, hsv)
            txt = [mode_txt]
            if res is not None:
                txt.append(f"pos=({res['x']:+.3f},{res['y']:+.3f},{res['z']:+.3f})m")
                txt.append(f"yaw={res['yaw']:6.1f}deg  {res['kind']}"
                           f"  测得边长={res['side']*1000:.0f}mm(偏差{res['size_err']*100:.0f}%)")
                cv2.putText(vis, f"x={res['x']:+.3f} y={res['y']:+.3f} z={res['z']:+.3f} m",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis, f"yaw={res['yaw']:.1f} deg",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis, f"{res['kind']}  side~{res['side']*1000:.0f}mm",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                txt.append("未检测到方块/地面")
            for i, t in enumerate(txt):
                cv2.putText(vis, t, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            if click_marker is not None:
                # 画出点击实际落点：若白圈不在你点的位置，说明存在高分屏坐标偏移
                cv2.circle(vis, click_marker, 6, (255, 255, 255), 1)
                cv2.drawMarker(vis, click_marker, (255, 0, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.imshow("block_pose", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_mode = True
                preset, click_hsv = None, None
                print("🤖 已切回 AUTO 模式（红∪蓝合并掩码，无需点击）")
            elif key in HSV_PRESETS:
                auto_mode = False
                preset, click_hsv = key, None
                print(f"手动预设: {HSV_PRESETS[key][0]}")
            elif key == ord('g'):
                ground = None
                print("🌍 将重新估计地面")
            elif key == ord('d'):
                debug = not debug
                print(f"🔍 调试输出: {'开' if debug else '关'}（会打印每个候选被剔除的原因）")
            elif key == ord('p'):
                if not results:
                    print("⚠ 还没有有效结果")
                    continue
                xs_ = np.median([r["x"] for r in results])
                ys_ = np.median([r["y"] for r in results])
                zs_ = np.median([r["z"] for r in results])
                yw_ = np.median([r["yaw"] for r in results])
                w_ = np.median([r["w"] for r in results])
                l_ = np.median([r["l"] for r in results])
                h_ = np.median([r["h"] for r in results])
                sd_ = np.median([r["side"] for r in results])
                cb_ = np.median([r["cube"] for r in results])
                er_ = np.median([r["size_err"] for r in results])
                kd_ = results[-1]["kind"]
                print("\n" + "=" * 70)
                print(f"【任务一·基础要求】方块位姿（最近 {len(results)} 帧中位数）")
                print("=" * 70)
                print(f"  世界坐标位置  x = {xs_:+.4f} m   y = {ys_:+.4f} m   z = {zs_:+.4f} m")
                print(f"  yaw 角        = {yw_:+.2f} deg （绕竖直Z轴；立方体按 mod 90° 报告）")
                print(f"  方块类型      = {kd_}")
                print(f"  测得边长      = {sd_*1000:.1f} mm   （标准 {cb_*1000:.0f} mm，"
                      f"偏差 {er_*100:.1f}%）")
                print(f"  包围盒(估)    = {w_:.3f} x {l_:.3f} x {h_:.3f} m")
                print(f"  （世界系：原点=相机光心，Z=竖直向上，X=相机右方向投影到地面，单位 m）")
                print("=" * 66 + "\n")
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(f"results/task1_vis_{time.strftime('%H%M%S')}.png", vis)
                with open("results/task1_basic.txt", "a") as f:
                    f.write(f"x={xs_:+.5f} y={ys_:+.5f} z={zs_:+.5f} yaw={yw_:+.3f} "
                            f"kind={kd_} side={sd_*1000:.1f}mm std={cb_*1000:.0f}mm "
                            f"err={er_*100:.1f}% bbox={w_:.3f}x{l_:.3f}x{h_:.3f}\n")
                print("✅ 已追加保存 -> results/task1_basic.txt")
    finally:
        if started:
            try:
                pipe.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
