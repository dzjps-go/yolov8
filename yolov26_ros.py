from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.predict(
    source = "tupian2.jpg",
    save = True,
    conf = 0.25,
)

model.train(
    data = "coco128.yaml",
    epochs = 5,
    imgsz = 320,
    batch = 2,
    workers = 2,
    device = 'cpu',
    name = "yolov26——shujuji"
)


print("运行成功！")