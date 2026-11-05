# ROS 2 的 Python 客户端库：负责初始化、spin、关闭
import rclpy
# Node 是所有 ROS 2 节点的基类
from rclpy.node import Node
# ROS 2 自带的图像消息类型，与 cv_bridge 配合使用
from sensor_msgs.msg import Image
# 在 OpenCV 图像（numpy 数组）和 ROS 图像消息之间做转换
from cv_bridge import CvBridge
# OpenCV：用来读取图片文件
import cv2


class ImagePublisher(Node):
    """定时把一张静态图片发布到 /image_raw 话题的节点。"""

    def __init__(self):
        # 初始化节点，节点名 image_publisher
        super().__init__('image_publisher')
        # 声明参数 image_path，默认值 tuoian1.jpg；启动时可用 -p 覆盖
        self.declare_parameter("image_path", "tuoian1.jpg")
        # 读取参数值，赋给成员变量供后面 imread 使用
        self.image_path = self.get_parameter("image_path").value

        # 创建发布者：消息类型 Image，话题 /image_raw，队列长度 10
        self.publisher = self.create_publisher(
            Image,
            "/image_raw",
            10
        )

        # 创建 OpenCV <-> ROS 消息的转换器
        self.bridge = CvBridge()

        # 按参数路径读入图片，得到 numpy 数组
        self.image = cv2.imread(self.image_path)

        # 路径错误或格式不支持时读出来是 None，这里直接报错退出
        if self.image is None:
            raise RuntimeError(
                f"无法解析图片：{self.image_path}"
            )

        # 创建定时器：每 0.5 秒调用一次 publish_image
        self.timer = self.create_timer(
            0.5,
            self.publish_image
        )

        # 打印启动日志
        self.get_logger().info("节点启动成功")

    def publish_image(self):
        # 把 numpy 图像转成 ROS 的 Image 消息，编码为 bgr8
        msg = self.bridge.cv2_to_imgmsg(
            self.image,
            encoding="bgr8"
        )

        # 消息头写当前时间戳，便于下游做同步或延迟计算
        msg.header.stamp = self.get_clock().now().to_msg()
        # 消息头写坐标系名，与相机对应的 frame 一致
        msg.header.frame_id = "camera_frame"

        # 真正发出消息
        self.publisher.publish(msg)
        # 打印发布日志
        self.get_logger().info("图片已发布到 /image_raw")


def main(args=None):
    # 初始化 rclpy 上下文，并解析命令行参数
    rclpy.init(args=args)
    # 创建节点实例（此时会读图、建发布者、启定时器）
    node = ImagePublisher()
    try:
        # 进入事件循环：持续处理定时器回调等任务
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 退出时不当作错误处理
        pass
    finally:
        # 无论怎么退出，都销毁节点并关闭 rclpy 上下文
        node.destroy_node()
        rclpy.shutdown()


# 直接运行本文件时才执行 main（被 import 时不执行）
if __name__ == '__main__':
    main()