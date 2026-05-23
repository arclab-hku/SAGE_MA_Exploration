import numpy as np
from skimage.measure import block_reduce

class MapProcessor:
    def __init__(self, map_origin, resolution, downsample_factor):
        self.map_origin = map_origin
        self.resolution = resolution
        self.downsample_factor = downsample_factor
        self.map_size_height = int(2 * map_origin[1] / (resolution * downsample_factor))
        self.map_size_width = int(2 * map_origin[0] / (resolution * downsample_factor))
        self.downsampled_map_data = -1 * np.ones((self.map_size_height, self.map_size_width), dtype=np.int8)
        # 边缘设为障碍物
        self.downsampled_map_data[0, :] = 100
        self.downsampled_map_data[-1, :] = 100
        self.downsampled_map_data[:, 0] = 100
        self.downsampled_map_data[:, -1] = 100
        
    def update_map(self, origin_map_data, height, width):
        """更新并下采样地图"""
        # 计算下采样后的宽度和高度
        new_width = int(width / self.downsample_factor)
        new_height = int(height / self.downsample_factor)

        # 使用block_reduce进行下采样
        map_data = block_reduce(origin_map_data, 
                              block_size=(self.downsample_factor, self.downsample_factor), 
                              func=np.max)
        return map_data, new_width, new_height

    def update_downsampled_map(self, map_x, map_y, viewrange, map_data):
        """根据UAV位置更新下采样地图"""
        height_0 = max(0, map_x - viewrange)
        height_1 = min(self.map_size_height, map_x + viewrange + 1)
        width_0 = max(0, map_y - viewrange)
        width_1 = min(self.map_size_width, map_y + viewrange + 1)
        
        # 创建局部坐标系
        x, y = np.ogrid[height_0:height_1, width_0:width_1]
        # 计算到中心的距离
        dist_from_center = np.sqrt((x - map_x)**2 + (y - map_y)**2)
        
        # 不更新的区域
        value = np.logical_and(self.downsampled_map_data[x, y] == 0, map_data[x, y] == -1)                 
        value_complement = np.logical_not(value)     
        mask = np.logical_and(dist_from_center < viewrange, value_complement)
        
        # 更新地图
        self.downsampled_map_data[height_0:height_1, width_0:width_1][mask] = \
            map_data[height_0:height_1, width_0:width_1][mask]
            
        # 处理障碍物
        self._process_obstacles(height_0, height_1, width_0, width_1, map_data)
        
    def _process_obstacles(self, height_0, height_1, width_0, width_1, map_data):
        """处理障碍物"""
        obstacle_mask = self.downsampled_map_data[height_0:height_1, width_0:width_1] == 100
        obstacle_rows, obstacle_cols = np.nonzero(obstacle_mask)

        # 计算原始地图中对应的起始位置
        original_row_starts = (height_0 + obstacle_rows) * self.downsample_factor
        original_col_starts = (width_0 + obstacle_cols) * self.downsample_factor

        # 创建网格索引
        row_indices = np.arange(self.downsample_factor)[:, np.newaxis]
        col_indices = np.arange(self.downsample_factor)

        # 使用广播来创建所有需要检查的索引
        check_rows = original_row_starts[:, np.newaxis, np.newaxis] + row_indices
        check_cols = original_col_starts[:, np.newaxis, np.newaxis] + col_indices

        # 在原始地图中提取所有相关区域
        original_regions = map_data[check_rows, check_cols]

        # 计算每个区域中值为100的像素数量
        counts_100 = np.sum(original_regions == 100, axis=(1, 2))

        # 确定哪些下采样像素应该被设置为非障碍物
        clear_obstacles = counts_100 < self.downsample_factor * self.downsample_factor / 2

        # 更新下采样地图
        self.downsampled_map_data[height_0 + obstacle_rows[clear_obstacles], 
                                width_0 + obstacle_cols[clear_obstacles]] = 0

    @staticmethod
    def ray_point(x0, y0, x1, y1):
        """射线追踪算法"""
        points = []
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
    
        if dx == 0:
            while y0 != y1:
                y0 += sy
                points.append((x0, y0))
            return points
        elif dy == 0:
            while x0 != x1:
                x0 += sx
                points.append((x0, y0))
            return points
        
        while True:
            if x0 == x1 and y0 == y1:
                break
            if x0 == x1:
                while y0 != y1:
                    y0 += sy
                    points.append((x0, y0))
                break
            if y0 == y1:
                while x0 != x1:
                    x0 += sx
                    points.append((x0, y0))
                break
            x0 += sx
            y0 += sy
            points.append((x0, y0))
        return points 