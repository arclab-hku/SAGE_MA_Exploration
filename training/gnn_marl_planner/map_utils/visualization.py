import numpy as np
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension

class MapVisualizer:
    def __init__(self):
        pass
        
    @staticmethod
    def occupancy_to_rgb(map_data, map_size_height, map_size_width, frontier_dict=None, 
                        viewpoint_dict=None, goal_of_drone=None, uav_odoms=None):
        """将占据栅格地图转换为RGB图像"""
        rgb_map = np.zeros((map_size_height, map_size_width, 3), dtype=np.uint8)
        
        # 设置未知区域为灰色
        rgb_map[map_data == -1] = [128, 128, 128]
        
        # 设置已知区域
        known = map_data != -1
        values = np.interp(map_data[known], [0, 100], [255, 0]).astype(np.uint8)
        rgb_map[known] = np.stack([values, values, values], axis=-1)
        
        # 绘制frontier点
        if frontier_dict:
            for frontier in frontier_dict:
                rgb_map[frontier[0], frontier[1]] = [0, 255, 0]
        
        # 绘制viewpoint点
        if viewpoint_dict:
            for view_point in viewpoint_dict:
                rgb_map[view_point[0], view_point[1]] = [0, 0, 255]

        # 绘制目标点
        if goal_of_drone:
            for goal in goal_of_drone:
                rgb_map[int(goal[0]), int(goal[1])] = [128, 0, 128]
                
        # 绘制UAV位置
        if uav_odoms:
            for uav_id in uav_odoms:
                rgb_map[uav_odoms[uav_id][0], uav_odoms[uav_id][1]] = [255, 0, 0]
                
        return rgb_map

    @staticmethod
    def create_rgb_message(rgb_map):
        """创建RGB消息"""
        msg = UInt8MultiArray()
        msg.data = rgb_map.flatten().tolist()
        
        msg.layout.dim = [
            MultiArrayDimension(label="height", size=rgb_map.shape[0], 
                              stride=rgb_map.shape[0]*rgb_map.shape[1]*3),
            MultiArrayDimension(label="width", size=rgb_map.shape[1], 
                              stride=rgb_map.shape[1]*3),
            MultiArrayDimension(label="channel", size=3, stride=3)
        ]
        
        return msg 