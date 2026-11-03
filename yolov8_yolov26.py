
from ultralytics import YOLO
import time
import statistics

image = "/home/jianmo/yolo_projet/tuoian1.jpg"

models = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO26n": "yolo26n.pt"
}

for name, weight in models.items():

    model = YOLO(weight)

    # 预热
    for _ in range(5):
        model.predict(
            image,
            imgsz=320,
            device="cpu",
            verbose=False
        )

    times = []

    # 正式测试30次
    for _ in range(30):

        start = time.perf_counter()

        model.predict(
            image,
            imgsz=320,
            device="cpu",
            verbose=False
        )

        end = time.perf_counter()

        times.append(end - start)

    avg_time = statistics.mean(times)

    fps = 1 / avg_time

    print(f"\n{name}")
    print(f"平均耗时: {avg_time * 1000:.2f} ms")
    print(f"平均FPS: {fps:.2f}")