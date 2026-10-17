#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时相机测试训练好的 YOLO 方块检测模型（Assignment3）。

连相机，每帧把彩色图送进 best.pt 推理，绿框标出方块 + 类别 + 置信度。
相机来源：
  - 默认：RealSense D435i（pyrealsense2）
  - --cam N：普通 USB/笔记本摄像头（OpenCV VideoCapture，N 为设备号，通常 0）
操作：q = 退出   空格 = 保存当前帧快照(yolo_test_NNN.png)
"""
import sys
import os
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_pose as bp
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

IMG_W, IMG_H = 640, 480
MODEL = "runs/detect/block_yolo11n/weights/best.pt"
CONF = 0.15
IMGSZ = 960      # 远距小方块在 640 下仅几十像素、轮廓糊掉会漏检；960 实测可救回


def detect_frame(model, color):
    """对一帧彩色图推理，返回画好框的图。"""
    disp = color.copy()
    res = model.predict(color, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy().reshape(-1, 4)
        confs = boxes.conf.cpu().numpy().reshape(-1)
        for (x1, y1, x2, y2), cf_ in zip(xyxy, confs):
            if cf_ < CONF:
                continue
            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(disp, f"block {cf_:.2f}",
                        (int(x1), max(int(y1) - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(disp, "no detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None,
                    help="用普通 USB 相机（指定设备号，如 0）替代 RealSense")
    ap.add_argument("--sn", default=None,
                    help="RealSense 序列号（锁定训练用的相机 043322075459）")
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print(f"❌ 找不到权重 {MODEL}，请先跑 train_detect.py")
        sys.exit(1)
    model = YOLO(MODEL)

    win = "YOLO live test"
    cv2.namedWindow(win)
    cv2.imshow(win, np.zeros((IMG_H, IMG_W, 3), np.uint8))
    cv2.waitKey(30)
    cnt = 0

    def handle_key():
        nonlocal cnt
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return "quit"
        if k == ord(' '):
            cnt += 1
            cv2.imwrite(f"yolo_test_{cnt:03d}.png", disp_saved)
            print(f"✅ 保存 yolo_test_{cnt:03d}.png")
        return None

    if args.cam is not None:
        # ---- 普通 USB 相机 ----
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            print(f"❌ 打不开 USB 相机 device={args.cam}")
            sys.exit(1)
        print(f"USB 相机 device={args.cam} 已打开，实时检测中。q=退出 空格=存图")
        try:
            while True:
                ok, color = cap.read()
                if not ok:
                    continue
                disp_saved = detect_frame(model, color)
                cv2.imshow(win, disp_saved)
                if handle_key() == "quit":
                    break
        finally:
            cap.release()
    else:
        # ---- RealSense D435i ----
        if rs is None:
            print("❌ 未安装 pyrealsense2，请用 --cam N 指定 USB 相机")
            sys.exit(1)
        ctx = rs.context()
        devs = list(ctx.query_devices())
        if not devs:
            print("❌ 未检测到 RealSense 相机")
            sys.exit(1)
        serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
        print(f"检测到 RealSense 相机: {serials}")
        if args.sn:
            if args.sn not in serials:
                print(f"❌ 指定的 SN {args.sn} 不在已连接设备 {serials}")
                sys.exit(1)
            sn = args.sn
        else:
            sn = serials[0]
            print(f"未指定 --sn，默认用第一台 {sn}。"
                  f"⚠ 若框不准，多半是选错相机——训练数据用的是 043322075459。")
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
        pipe.start(cfg)      # 必须传 cfg！否则 libRealsense 按默认 RGB8 给数据，
                             # cv2/训练都按 BGR 解释 → 红蓝互换 + 检测域偏移
        print(f"相机 SN={sn} 已启动，实时检测中。q=退出 空格=存图 "
              f"(训练相机=043322075459)")
        try:
            while True:
                cf = pipe.wait_for_frames().get_color_frame()
                if not cf:
                    continue
                color = np.asanyarray(cf.get_data())
                disp_saved = detect_frame(model, color)
                cv2.imshow(win, disp_saved)
                if handle_key() == "quit":
                    break
        finally:
            pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
