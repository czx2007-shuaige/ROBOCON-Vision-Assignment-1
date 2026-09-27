import imageio
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("使用方式：python analyze_video.py 输入视频路径")
        return
    input_path = sys.argv[1]
    output_path = "processed_output.mp4"

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()["fps"]
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in reader:
        # 简单提亮，运算量很小，不会卡死
        frame = np.clip(frame * 1.2, 0, 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
    print(f"✅视频处理完成，已保存至 {output_path}")

if __name__ == "__main__":
    main()
