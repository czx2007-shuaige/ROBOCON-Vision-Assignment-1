# Project A 摄像头+进程观察
## 环境创建、激活、安装依赖、运行命令
1. conda create -n vision python=3.10
2. conda activate vision
3. python --version
4. which python
5. pip install numpy opencv-python
6. python camera.py

## 功能说明
1. 读取摄像头实时画面
2. 窗口1：原始彩色图像
3. 窗口2：灰度图像
4. 窗口3：Canny轮廓提取图像
5. 程序运行至少30秒，按下q退出
6. 退出自动保存原始视频 raw_capture.mp4
