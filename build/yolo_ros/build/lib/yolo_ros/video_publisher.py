
import rclpy
from rclpy.node import Node  # 导入ROS2节点类
from sensor_msgs.msg import Image  # 导入ROS2图像消息类型
from cv_bridge import CvBridge  # 导入OpenCV与ROS2图像转换工具
import cv2  # 导入OpenCV库


class VideoPublisher(Node):  # 定义视频发布节点
    def __init__(self):  # 节点初始化函数
        super().__init__('video_publisher')  # 设置节点名称

        self.publisher = self.create_publisher(  # 创建图像发布者
            Image,  # 发布的消息类型
            '/image_raw',  # 发布的话题名称
            10  # 消息队列长度
        )

        self.video_path = 'shiping.mp4'  # 设置视频绝对路径

        self.cap = cv2.VideoCapture(self.video_path)  # 打开本地MP4视频

        if not self.cap.isOpened():  # 判断视频是否成功打开
            raise RuntimeError('无法打开视频文件')  # 打开失败则报错

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获取视频原始帧率

        if self.fps <= 0:  # 判断视频帧率是否有效
            self.fps = 30.0  # 无效时使用默认30帧

        self.bridge = CvBridge()  # 创建图像转换对象

        self.timer = self.create_timer(  # 创建定时器
            1.0 / self.fps,  # 根据视频帧率设置发布间隔
            self.publish_frame  # 定时执行视频发布函数
        )

        self.get_logger().info('视频发布节点启动成功')  # 输出启动信息

    def publish_frame(self):  # 定义视频逐帧发布函数
        ret, frame = self.cap.read()  # 从视频中读取一帧

        if not ret:  # 判断是否读取到画面
            self.get_logger().info('视频播放结束')  # 输出结束信息
            self.timer.cancel()  # 停止定时器
            self.cap.release()  # 释放视频资源
            return  # 结束本次函数

        msg = self.bridge.cv2_to_imgmsg(  # 将OpenCV图像转换成ROS2消息
            frame,  # 当前读取的视频帧
            encoding='bgr8'  # 指定图像编码格式
        )

        self.publisher.publish(msg)  # 发布当前视频帧

        self.get_logger().info('视频帧已发布')  # 输出发布信息


def main(args=None):  # 定义程序入口
    rclpy.init(args=args)  # 初始化ROS2

    node = VideoPublisher()  # 创建视频发布节点

    try:  # 开始异常处理
        rclpy.spin(node)  # 保持节点运行
    except KeyboardInterrupt:  # 捕获Ctrl+C中断
        pass  # 正常退出

    node.cap.release()  # 释放视频资源
    node.destroy_node()  # 销毁节点
    rclpy.shutdown()  # 关闭ROS2


if __name__ == '__main__':  # 判断是否直接运行当前文件
    main()  # 执行主函数