import cv2
import numpy as np
import os
import sys

# ========== 启动立刻打印 PID、PPID、Python路径、版本 ==========
pid = os.getpid()
ppid = os.getppid()
python_path = sys.executable
python_version = sys.version
print(f"PID: {pid}")
print(f"PPID: {ppid}")
print(f"Python可执行文件路径: {python_path}")
print(f"Python版本: {python_version}")

# 打开默认内置摄像头 0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头打开失败")
    exit()

# 获取视频宽高、帧率，用于视频保存
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("raw_capture.mp4", fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # 写入原始画面到视频文件
    out.write(frame)

    # 1.原图
    original = frame.copy()
    # 2.灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3.轮廓提取
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 50,150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(frame)
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)

    # 三个窗口：原始画面、灰度画面、轮廓提取
    cv2.imshow("original", original)
    cv2.imshow("gray", gray)
    cv2.imshow("contour", contour_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源，保存视频
cap.release()
out.release()
cv2.destroyAllWindows()
print("程序结束，原始视频已保存至 raw_capture.mp4")

