from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog

def choose_input_files():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    image_paths = filedialog.askopenfilenames(
        title="选择图片文件",
        filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif"), ("所有文件", "*.*")]
    )
    if not image_paths:
        print("未选择任何图片文件，程序退出")
        exit(0)
    return image_paths

def detect_images():
    # ======================= 配置区 =======================
    # 模型配置
    model_config = {
        'model_path': r'E:\git-project\YOLOV11\ultralytics-main\weights\yolo11n.pt',  # 本地模型路径
        'download_url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'  # 如果没有模型文件可在此处添加下载URL
    }
    
    # 推理参数
    predict_config = {
        'conf_thres': 0.25,     # 置信度阈值
        'iou_thres': 0.45,      # IoU阈值
        'imgsz': 640,           # 输入分辨率
        'line_width': 2,        # 检测框线宽
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    }
    # ====================== 配置结束 ======================

    try:
        # 选择图片文件
        image_paths = choose_input_files()

        # 创建保存目录：代码文件所在目录下的 predict 文件夹
        save_dir = os.path.join(os.getcwd(), "predict", "exp")
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(save_dir):
            i = 1
            while os.path.exists(f"{save_dir}{i}"):
                i += 1
            save_dir = f"{save_dir}{i}"
            os.makedirs(save_dir)

        # 加载模型（带异常捕获）
        if not Path(model_config['model_path']).exists():
            if model_config['download_url']:
                print("开始下载模型...")
                YOLO(model_config['download_url']).download(model_config['model_path'])
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_config['model_path']}")

        # 初始化模型
        model = YOLO(model_config['model_path']).to(predict_config['device'])
        print(f"✅ 模型加载成功 | 设备: {predict_config['device'].upper()}")

        # 处理每个选定的图片文件
        for image_path in image_paths:
            print(f"正在处理图片: {image_path}")
            img = cv2.imread(image_path)

            if img is None:
                print(f"无法读取图片: {image_path}")
                continue

            # 执行推理
            results = model.predict(
                source=img,  # 输入图片
                stream=False,  # 禁用流模式
                verbose=False,
                conf=predict_config['conf_thres'],
                iou=predict_config['iou_thres'],
                imgsz=predict_config['imgsz'],
                device=predict_config['device']
            )

            # 解析并绘制结果（取第一个结果）
            for result in results:
                annotated_img = result.plot(line_width=predict_config['line_width'])
                break

            # 保存推理结果图像到文件
            output_image_path = os.path.join(save_dir, f"output_{os.path.basename(image_path)}")
            cv2.imwrite(output_image_path, annotated_img)
            print(f"推理结果已保存至: {output_image_path}")

            # 显示实时画面
            # cv2.imshow('YOLO Real-time Detection', annotated_img)

            # 等待按键退出当前图片查看
            if cv2.waitKey(0) & 0xFF == ord('q') :
                break

        cv2.destroyAllWindows()
        print("✅ 检测结束")

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("问题排查建议：")
        print("1. 检查图片文件路径是否正确")
        print("2. 确认模型文件路径正确")
        print("3. 检查CUDA是否可用（如需GPU加速）")
        print("4. 尝试降低分辨率设置")

if __name__ == "__main__":
    detect_images()
