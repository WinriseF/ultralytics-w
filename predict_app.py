import cv2
import pyautogui
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO
import os

# 加载YOLO模型
model = YOLO("weights-train/Flappybird-11.pt")  # 替换为你自己的权重文件路径

# 获取FlappyBird程序的窗口，假设程序名为 'FlappyBird'（你需要根据实际情况修改程序名）
window_title = "Flappy Bird"  # 替换为你的程序窗口名称

# 获取窗口句柄
windows = gw.getWindowsWithTitle(window_title)

if len(windows) == 0:
    print(f"Error: Cannot find window with title '{window_title}'")
    exit()

# 获取窗口位置和大小
window = windows[0]
left, top, right, bottom = window.left, window.top, window.right, window.bottom

# 确保窗口已经激活
if not window.isActive:
    window.activate()

# 询问用户是否需要保存视频
save_video = input("Do you want to save the video (y/n)? ").strip().lower()

# 创建视频保存路径和视频写入器
if save_video == 'y':
    save_folder = "video_inference_results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 设置视频保存的文件名
    video_filename = os.path.join(save_folder, "flappybird_inference_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (right - left, bottom - top))

while True:
    # 捕获程序窗口区域的屏幕
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    # 转换为 numpy 数组
    frame = np.array(screenshot)
    # 将 RGB 转换为 BGR（OpenCV 使用 BGR 格式）
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(frame, imgsz=(128, 96))  # 添加imgsz参数
    # 对帧进行推理
    # results = model(frame)

    # 提取推理结果，绘制检测框
    frame_with_boxes = results[0].plot()  # 使用 results[0].plot() 来绘制检测框

    # 显示视频帧
    cv2.imshow("FlappyBird Detection", frame_with_boxes)

    # 如果需要保存视频，写入当前帧
    if save_video == 'y':
        out.write(frame_with_boxes)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
if save_video == 'y':
    out.release()

cv2.destroyAllWindows()
