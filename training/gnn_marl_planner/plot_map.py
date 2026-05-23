import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class OccupancyMapSubscriber(Node):
    def __init__(self):
        super().__init__('occupancy_map_subscriber')
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            'occupancy_map_rgb',
            self.map_callback,
            10)
        
        # 创建保存目录
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"map_saves_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Saving maps to directory: {self.save_dir}')
        self.save_counter = 0
        
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(12, 10))  # 增大图形尺寸
        self.image = None

    def map_callback(self, msg):
        self.get_logger().info('Receiving RGB occupancy map')
        
        # 重建 numpy 数组
        height = msg.layout.dim[0].size
        width = msg.layout.dim[1].size
        channels = msg.layout.dim[2].size
        rgb_map = np.array(msg.data, dtype=np.uint8).reshape(height, width, channels)
        
        # 递增计数器并保存为 .npy 文件
        self.save_counter += 1
        save_path = os.path.join(self.save_dir, f"{self.save_counter}.npy")
        np.save(save_path, rgb_map)
        self.get_logger().info(f'Saved occupancy map to {save_path}')
        
        # 显示 RGB 数据
        if self.image is None:
            self.image = self.ax.imshow(rgb_map, origin='lower')
            # 移除了颜色条
        else:
            self.image.set_data(rgb_map)
        
        self.ax.set_title("Occupancy Map")
        self.fig.tight_layout()  # 调整布局以充分利用图形空间
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)  # 暂停一小段时间以允许图形更新

def main(args=None):
    rclpy.init(args=args)
    occupancy_map_subscriber = OccupancyMapSubscriber()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(occupancy_map_subscriber)
            plt.pause(0.1)  # 允许 matplotlib 事件循环运行
    except KeyboardInterrupt:
        pass
    
    occupancy_map_subscriber.destroy_node()
    rclpy.shutdown()
    plt.ioff()  # 关闭交互模式
    plt.show()  # 保持图形窗口打开

if __name__ == '__main__':
    main()