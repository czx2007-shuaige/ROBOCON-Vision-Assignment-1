# Part5 — C++ g++手动编译OpenCV视频处理

## 实验目的
使用g++直接手写编译命令，完成多文件C++项目编译，读取视频，将画面转为灰度图并输出新视频。

## 项目目录结构
```
cpp_task
├── include
│   └── transform.hpp
├── src
│   ├── main.cpp
│   └── transform.cpp
└── README.md

## 完整编译命令
```bash
g++ src/main.cpp src/transform.cpp -Iinclude -I/usr/include/eigen3 `pkg-config --cflags --libs opencv4` -o video_process

## 运行程序
```bash
./video_process ../part2_python/raw_capture.mp4 output.mp4
```

## 实验截图

### 1. 编译成功截图
![编译成功截图](../assets/part5_compile.png)

### 2. 程序运行完成截图
![运行成功截图](../assets/part5_run.png)

## 编译命令相关说明

### 1. `-I` 的作用是什么？
`-I`（大写i）用来**指定头文件的搜索目录**。
编译的时候，编译器除了系统默认头文件路径外，还会去 `-I` 后面给出的文件夹里面查找 `.h / .hpp` 头文件。
本项目 `-Iinclude` 就是告诉编译器，去当前目录下的 `include` 文件夹寻找自定义头文件 `transform.hpp`。

### 2. 为什么 transform.hpp 不单独作为一个 cpp？
`.hpp` 是**头文件**，一般只放函数声明、类定义、宏定义，**不写完整可编译的实现代码**；
函数具体实现写在 `.cpp` 文件（transform.cpp）。
头文件是用来给别的源文件`#include`引用的，不需要单独拿去g++编译。

### 3. 为什么只写 main.cpp 往往无法得到完整程序？
main.cpp里面调用了在`transform.cpp`中实现的函数。
如果只编译main.cpp，编译器找不到那些函数的具体实现，链接阶段就会报未定义引用的错误。
**所有包含函数实现的cpp源码，都必须一同参与编译**。

### 4. 编译成功后产生的文件是什么？
由命令最后的 `-o video_process` 指定，编译链接完成生成可执行程序：`video_process`。
这是二进制可执行文件，使用 `./video_process` 就可以运行。
