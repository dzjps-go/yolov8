"""
YOLOv8 使用示例脚本
===================

依次演示 YOLOv8 的三项常用功能：
    1. 图片推理（predict）：用预训练模型检测 tupian2.jpg 中的目标；
    2. 视频推理（predict）：对 shiping.mp4 逐帧检测并保存结果视频；
    3. 模型训练（train）：在 coco128.yaml 数据集上用 CPU 训练若干轮。

运行前提：
    - 已安装 ultralytics（pip install ultralytics）；
    - 当前目录下存在 yolov8n.pt、tupian2.jpg、shiping.mp4；
    - 首次训练会自动下载 COCO128 数据集。

所有推理/训练结果默认保存在 runs/detect/ 目录下。
"""

from ultralytics import YOLO


def main():
    # --------------------------------------------------------------
    # 1. 加载模型
    # --------------------------------------------------------------
    # "yolov8n.pt" 是官方预训练的 YOLOv8 Nano 权重：
    # n 代表 nano，是体积最小、速度最快的一档。
    # 如果本地没有该文件，ultralytics 会自动联网下载。
    model = YOLO("yolov8n.pt")

    # --------------------------------------------------------------
    # 2. 图片推理
    # --------------------------------------------------------------
    results = model.predict(
        source="tupian2.jpg",  # 输入源：图片路径，也可传目录、URL 或列表
        imgsz=416,             # 推理分辨率，越大越准但越慢
        conf=0.25,             # 置信度阈值，低于该值的检测框会被丢弃
        device="cpu",          # 运行设备："cpu" 或 "0"（第 0 块 GPU）
        save=True              # 是否把画好检测框的结果图保存到磁盘
    )
    # results 是列表，每张输入图片对应一个 Results 对象：
    #   results[i].boxes  -> 该图中所有检测框
    #   model.names       -> 类别 id 到类别名的映射

    # --------------------------------------------------------------
    # 3. 视频推理
    # --------------------------------------------------------------
    # 对视频逐帧检测，save=True 会把结果写回视频文件
    # （runs/detect/predict*/shiping.avi）。
    # 这里没有接收返回值，因为下面只打印图片的检测结果。
    model.predict(
        source="shiping.mp4",  # 输入源：视频文件路径
        device="cpu",          # 运行设备
        imgsz=416,             # 推理分辨率
        save=True              # 保存带检测框的结果视频
    )

    # --------------------------------------------------------------
    # 4. 模型训练
    # --------------------------------------------------------------
    # 在 COCO128 小数据集上训练，主要用于验证训练流程能否跑通。
    model.train(
        data="coco128.yaml",    # 数据集配置：类别名、训练/验证集路径
        epochs=5,               # 训练总轮数
        imgsz=320,              # 训练时的输入分辨率
        batch=2,                # 每个 batch 的图片数量
        workers=2,              # 数据加载子进程数（Windows 下建议设小）
        device="cpu",           # 使用 CPU 训练
        name="yolov8_cpu_text"  # 本次训练结果保存的子目录名
    )

    # --------------------------------------------------------------
    # 5. 打印图片推理结果
    # --------------------------------------------------------------
    for result in results:        # 遍历每张图片的结果
        for box in result.boxes:  # 遍历该图片中的每个检测框
            cls_id = int(box.cls[0])     # 类别 id（张量转 Python int）
            conf = float(box.conf[0])    # 置信度（张量转 Python float）
            xyxy = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]

            print(
                "类别：",
                model.names[cls_id],  # 用 id 查出类别名，如 person、car
                "置信度：",
                round(conf, 2),       # 置信度保留两位小数
                "坐标：",
                xyxy
            )


if __name__ == "__main__":
    # 仅当直接运行本文件时执行 main()；
    # 被其他模块 import 时不会自动执行。
    main()