# Part IV Python Project B
## 实验目标
创建第二个独立Conda环境，Project B要求Python版本 >=3.12 且 <3.14，与Project A的Python3.10版本相互独立。读取Part2录制的原始视频 raw_capture.mp4，对视频每一帧图像进行图像处理，输出处理完毕的新视频文件。全程不修改pyproject.toml中的版本约束，两套环境互不干扰。

## 1. 环境部署与运行命令
```bash
# 1. 创建Project B专属conda环境，指定Python3.13
conda create -n vision-b python=3.13

# 2. 激活Project B环境
conda activate vision-b

# 3. 查看当前Python版本
python --version

# 4. 查看解释器路径
which python

# 5. 根据pyproject.toml自动安装全部依赖包
pip install .

# 6. 运行视频处理程序，读取part2目录的原始视频
python analyze_video.py ../part2_python/raw_capture.mp4
