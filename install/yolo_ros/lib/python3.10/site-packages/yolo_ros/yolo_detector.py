import json

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("imgsz", 320)

        model_path = self.get_parameter("model_path").value
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10,
        )
        self.image_publisher = self.create_publisher(
            Image,
            "/yolo/image",
            10,
        )
        self.detection_publisher = self.create_publisher(
            String,
            "/yolo/detections",
            10,
        )

        self.get_logger().info(f"YOLO模型运行成功: {model_path}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                device="cpu",
                conf=0.25,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()
            image_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8",
            )
            image_msg.header = msg.header
            self.image_publisher.publish(image_msg)

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    xyxy = [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ]
                    detections.append({
                        "class": self.model.names[cls_id],
                        "confidence": round(confidence, 4),
                        "bbox_xyxy": xyxy,
                    })

            output = String()
            output.data = json.dumps(detections, ensure_ascii=False)
            self.detection_publisher.publish(output)
            self.get_logger().info(f"检测到{len(detections)}个目标")
        except Exception as e:
            self.get_logger().error(f"图像处理失败：{e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
