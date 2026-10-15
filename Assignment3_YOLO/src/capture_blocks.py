#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 任务一数据采集：保存「彩色 + 对齐深度 + 内参」成组快照
------------------------------------------------------------------------------
 用途：把现场画面存下来，供离线调试 block_pose.py 的算法与阈值
      （不接相机也能复现整条流水线）。

 操作：
   空格 或 s → 保存一组快照（彩色png + 深度npy + 内参json）
   d         → 切换深度伪彩窗口
   q         → 退出

 建议采集内容（每个场景拍 1~2 张，共拍 8~15 组）：
   1) 方块在画面中央、正对        （基准）
   2) 方块偏左 / 偏右
   3) 方块远（~1m） / 近（~0.4m）
   4) 方块旋转不同 yaw（0°、30°、60°…）
   5) 顶面多露出一些 / 少露出一些（俯视角变化）
   6) 你觉得"看起来难"的场景（光照暗、背景干扰等）

 ⚠ 每拍一组，最好用卷尺量一下并记在纸上（之后告诉我）：
      相机离地高度、方块相对相机的左右/前后偏移、yaw 大约多少
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception:
    print("❌ 需要安装 pyrealsense2")
    sys.exit(1)

IMG_W, IMG_H = 640, 480
OUT_DIR = "captures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", default=None, help="相机序列号（默认第一台）")
    ap.add_argument("--out", default=OUT_DIR, help="保存目录（默认 captures/）")
    args = ap.parse_args()

    devs = list(rs.context().query_devices())
    if not devs:
        print("❌ 未检测到 RealSense 相机")
        sys.exit(1)
    serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
    sn = args.sn if args.sn else serials[0]
    if sn not in serials:
        print(f"❌ SN {sn} 不在 {serials}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    align = rs.align(rs.stream.color)

    started = False
    show_depth = True
    count = len([f for f in os.listdir(args.out) if f.endswith("_color.png")])
    try:
        try:
            prof = pipe.start(cfg)
            started = True
        except Exception as e:
            print(f"❌ 打开相机失败: {e}")
            return
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        print(f"相机 SN={sn} 已启动  depth_scale={depth_scale}")
        print(f"空格/s=保存快照(已存{count}组)  d=切换深度图  q=退出")

        while True:
            frames = align.process(pipe.wait_for_frames())
            df = frames.get_depth_frame()
            cf = frames.get_color_frame()
            if not df or not cf:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())          # uint16
            intr = df.profile.as_video_stream_profile().intrinsics

            vis = color.copy()
            cv2.putText(vis, f"saved={count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, "SPACE/s=save  d=depth  q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if show_depth:
                dcol = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                vis = np.hstack([vis, cv2.resize(dcol, (IMG_W, IMG_H))])
            cv2.imshow("capture", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord(' '), ord('s')):
                count += 1
                tag = f"{count:03d}_{time.strftime('%H%M%S')}"
                cv2.imwrite(os.path.join(args.out, f"{tag}_color.png"), color)
                np.save(os.path.join(args.out, f"{tag}_depth.npy"), depth)
                meta = {
                    "serial": sn,
                    "depth_scale": float(depth_scale),
                    "fx": float(intr.fx), "fy": float(intr.fy),
                    "ppx": float(intr.ppx), "ppy": float(intr.ppy),
                    "width": int(intr.width), "height": int(intr.height),
                    "distortion": [float(c) for c in intr.coeffs],
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(os.path.join(args.out, f"{tag}_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"✅ 已保存 {tag}  (彩色/深度/内参)")
    finally:
        if started:
            try:
                pipe.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
