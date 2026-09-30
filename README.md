## 7. Git / GitHub
本作业使用Git进行版本控制，本次采用分支开发模式：新建dev分支用于文档完善，修改完成后合并回main主分支。
通过.gitignore配置忽略编译产物build文件夹、mp4视频文件、Python缓存文件，大型视频文件不上传到代码仓库，保存在本地。

## 本次实操的Git关键命令
```bash
# 1. 查看当前仓库变更状态
git status

# 2. 将修改的文件加入暂存区
git add .

# 3. 提交改动并填写提交备注
git commit -m "dev:完成PartVII Git章节文档"

# 4. 查看本地所有分支
git branch

# 5. 分支切换（git switch等价git checkout）
git switch main
git switch dev

# 6. 将本地分支推送至远程GitHub仓库
git push origin main
git push origin dev

# 7. 图形化查看全部提交记录，校验分支合并情况
git log --oneline --graph --all
```
## 8. Problems and Notes

1. Markdown图片链接如果存在隐形不可见字符，会造成GitHub网页图片无法渲染，图片链接建议手动重新录入。

2. CMake构建自动生成build目录，需要加入.gitignore，不要提交该文件夹。

3. 文件命名尽量不要使用中文，中文文件名容易触发路径识别异常。

4. OpenCV多窗口同时显示图像时，窗口配置不当会导致弹窗失败。

5. 使用htop观测进程前，需要提前启动目标程序，才能在进程列表找到对应PID。
.
