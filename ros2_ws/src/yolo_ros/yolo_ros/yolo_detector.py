import json  # 导入 json 模块，用于把检测结果转成 JSON 字符串

import rclpy  # 导入 ROS 2 的 Python 客户端库
from cv_bridge import CvBridge  # 导入 CvBridge，用于 ROS 图像消息与 OpenCV 图像互转
from rclpy.node import Node  # 导入 Node 基类，用来创建 ROS 2 节点
from sensor_msgs.msg import Image  # 导入 Image 消息类型，用于收发图像
from std_msgs.msg import String  # 导入 String 消息类型，用于收发字符串
from ultralytics import YOLO  # 导入 Ultralytics 的 YOLO 模型类


class YoloDetector(Node):  # 定义 YOLO 检测节点，继承 ROS 2 的 Node 类
    def __init__(self):  # 构造函数，节点创建时执行一次
        super().__init__("yolo_detector")  # 调用父类构造函数，节点命名为 yolo_detector

        self.declare_parameter("model_path", "yolov8n.pt")  # 声明参数 model_path，默认模型为 yolov8n.pt
        self.declare_parameter("imgsz", 320)  # 声明参数 imgsz（推理输入尺寸），默认 320

        model_path = self.get_parameter("model_path").value  # 读取 model_path 参数的值
        self.imgsz = int(self.get_parameter("imgsz").value)  # 读取 imgsz 参数并转成整数，存为成员变量

        self.model = YOLO(model_path)  # 加载 YOLO 模型
        self.bridge = CvBridge()  # 创建 CvBridge 对象，用于图像格式转换

        self.subscription = self.create_subscription(  # 创建图像订阅者，并保存引用防止被回收
            Image,  # 订阅的消息类型是 Image
            "/image_raw",  # 订阅的话题名称是 /image_raw
            self.image_callback,  # 收到图像后调用 image_callback
            10,  # 消息队列深度为 10
        )  # 订阅者创建结束
        self.image_publisher = self.create_publisher(  # 创建发布者，用于发布画好框的图像
            Image,  # 发布的消息类型是 Image
            "/yolo/image",  # 发布的话题名称是 /yolo/image
            10,  # 消息队列深度为 10
        )  # 图像发布者创建结束
        self.detection_publisher = self.create_publisher(  # 创建发布者，用于发布检测结果文本
            String,  # 发布的消息类型是 String
            "/yolo/detections",  # 发布的话题名称是 /yolo/detections
            10,  # 消息队列深度为 10
        )  # 检测结果发布者创建结束

        self.get_logger().info(f"YOLO模型运行成功: {model_path}")  # 打印日志，提示模型加载成功

    def image_callback(self, msg):  # 图像回调函数，每收到一帧图像就执行一次
        try:  # 用 try 包裹，避免单帧出错导致节点崩溃
            frame = self.bridge.imgmsg_to_cv2(  # 把 ROS 图像消息转成 OpenCV 图像
                msg,  # 待转换的 ROS 图像消息
                desired_encoding="bgr8",  # 指定输出编码为 OpenCV 常用的 bgr8
            )  # 图像转换结束
            results = self.model.predict(  # 用 YOLO 对当前帧做推理
                frame,  # 输入的图像帧
                imgsz=self.imgsz,  # 推理时的输入尺寸
                device="cpu",  # 指定使用 CPU 推理
                conf=0.25,  # 置信度阈值，低于该值的检测框被丢弃
                verbose=False,  # 关闭推理过程的详细输出
            )  # 推理结束，results 是结果列表

            result = results[0]  # 取第一张图（也就是当前帧）的检测结果
            annotated = result.plot()  # 在图像上绘制检测框和标签，返回新图像
            image_msg = self.bridge.cv2_to_imgmsg(  # 把画好框的 OpenCV 图像转回 ROS 消息
                annotated,  # 已绘制检测框的图像
                encoding="bgr8",  # 指定图像编码为 bgr8
            )  # 转换结束
            image_msg.header = msg.header  # 复制原始消息头，保留时间戳和坐标系信息
            self.image_publisher.publish(image_msg)  # 发布带检测框的图像

            detections = []  # 初始化检测结果列表
            if result.boxes is not None:  # 如果这一帧确实检测到了目标框
                for box in result.boxes:  # 逐个遍历每个检测框
                    cls_id = int(box.cls[0].item())  # 取出类别编号并转成整数
                    confidence = float(box.conf[0].item())  # 取出置信度并转成浮点数
                    xyxy = [  # 把检测框坐标整理成列表
                        round(float(value), 2)  # 每个坐标转成浮点数并保留两位小数
                        for value in box.xyxy[0].tolist()  # 遍历左上角和右下角的四个坐标值
                    ]  # 坐标列表构造结束
                    detections.append({  # 把一个检测结果加入列表
                        "class": self.model.names[cls_id],  # 类别编号对应的类别名称
                        "confidence": round(confidence, 4),  # 置信度保留四位小数
                        "bbox_xyxy": xyxy,  # 检测框坐标
                    })  # 单个检测结果添加结束

            output = String()  # 创建一条 String 消息
            output.data = json.dumps(detections, ensure_ascii=False)  # 把检测结果转成 JSON 文本，保留中文
            self.detection_publisher.publish(output)  # 发布检测结果
            self.get_logger().info(f"检测到{len(detections)}个目标")  # 打印本帧检测到的目标数量
        except Exception as e:  # 捕获处理过程中的任何异常
            self.get_logger().error(f"图像处理失败：{e}")  # 打印错误日志，不中断节点运行


def main(args=None):  # 程序入口函数
    rclpy.init(args=args)  # 初始化 ROS 2 客户端库
    node = YoloDetector()  # 创建 YOLO 检测节点实例
    try:  # 用 try 包裹，便于退出时清理资源
        rclpy.spin(node)  # 进入事件循环，持续处理回调直到节点停止
    except KeyboardInterrupt:  # 捕获 Ctrl+C 中断
        pass  # 中断时什么也不做，交给 finally 清理
    finally:  # 无论正常退出还是异常退出都会执行
        node.destroy_node()  # 销毁节点，释放资源
        rclpy.shutdown()  # 关闭 ROS 2 客户端库


if __name__ == "__main__":  # 当脚本被直接运行时
    main()  # 调用入口函数
