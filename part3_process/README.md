# Part III — 进程观察实验
## 实验目的
在Project A（camera.py摄像头程序）运行时，使用 ps、htop 工具观测进程，获取进程PID、PPID、CPU、内存占用等信息，理解操作系统进程相关概念。

## 实验步骤
1. 打开终端A，进入 part2_python 目录，执行
```bash
cd part2_python
python camera.py
## 实验采集数据
- PID：13506
- PPID：8563
- CPU使用率：15.8%
- 内存占用：0.3%
## htop监控截图
![htop_camera](./htop_camera.png)
