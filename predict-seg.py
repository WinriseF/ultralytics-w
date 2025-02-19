from ultralytics import YOLO
import os
import torch
from pathlib import Path

def detect_images():
    # ======================= 配置区 =======================
    # 模型配置
    model_config = {
        'model_path': r'E:\git-project\YOLOV11\ultralytics-main\weights\yolo11n-seg.pt',  # 本地模型路径
    }
    
    # 路径配置
    path_config = {
        'input_folder': r'E:\git-project\YOLOV11\ultralytics-main\picture',
        'output_folder': r'E:\git-project\YOLOV11\ultralytics-main\predict',
        'auto_create_dir': True  # 自动创建输出目录
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
        # 验证输入目录
        if not Path(path_config['input_folder']).exists():
            raise FileNotFoundError(f"输入目录不存在: {path_config['input_folder']}")

        # 自动创建输出目录
        if path_config['auto_create_dir']:
            Path(path_config['output_folder']).mkdir(parents=True, exist_ok=True)

        # 加载模型（直接加载本地模型）
        if not Path(model_config['model_path']).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_config['model_path']}")

        # 初始化模型
        model = YOLO(model_config['model_path']).to(predict_config['device'])
        print(f"✅ 模型加载成功 | 设备: {predict_config['device'].upper()}")

        # 执行推理
        results = model.predict(
            source=path_config['input_folder'],
            project=path_config['output_folder'],
            name="exp",
            save=True,
            conf=predict_config['conf_thres'],
            iou=predict_config['iou_thres'],
            imgsz=predict_config['imgsz'],
            line_width=predict_config['line_width'],
            show_labels=True,
            show_conf=True
        )

        # 统计信息
        success_count = len(results)
        save_dir = Path(results[0].save_dir) if success_count > 0 else None
        print(f"\n🔍 推理完成 | 处理图片: {success_count} 张")
        print(f"📁 结果目录: {save_dir.resolve() if save_dir else '无'}")

        # 显示首张结果（可选）
        if success_count > 0:
            results[0].show()

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("问题排查建议：")
        print("1. 检查模型文件路径是否正确")
        print("2. 确认图片目录包含支持的格式（jpg/png等）")
        print("3. 查看CUDA是否可用（如需GPU加速）")
        print("4. 确保输出目录有写入权限")

if __name__ == "__main__":
    detect_images()
