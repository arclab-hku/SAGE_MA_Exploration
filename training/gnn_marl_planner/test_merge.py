import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

class MapChunk:
    def __init__(self, size=200):
        self.size = size
        self.data = np.full((size, size), -1, dtype=np.int8)  # -1 for unknown
        self.is_empty = True
        self.need_update = False

class OccupancyGridMap:
    def __init__(self, resolution=0.1):
        self.resolution = resolution
        self.chunks = defaultdict(MapChunk)
        self.chunk_size = 200

    def update_cell(self, x, y, value):
        chunk_x, local_x = divmod(x, self.chunk_size)
        chunk_y, local_y = divmod(y, self.chunk_size)
        chunk = self.chunks[(chunk_x, chunk_y)]
        if chunk.data[local_y, local_x] != value:
            chunk.data[local_y, local_x] = value
            chunk.is_empty = False
            chunk.need_update = True

    def get_cell(self, x, y):
        chunk_x, local_x = divmod(x, self.chunk_size)
        chunk_y, local_y = divmod(y, self.chunk_size)
        return self.chunks[(chunk_x, chunk_y)].data[local_y, local_x]

class MapMerger:
    def __init__(self, base_map):
        self.merged_map = base_map
        self.updated_chunks = set()

    def merge_map(self, new_map):
        for chunk_coord, chunk in new_map.chunks.items():
            if not chunk.is_empty:
                self._merge_chunk(chunk_coord, chunk)

    def _merge_chunk(self, chunk_coord, new_chunk):
        merged_chunk = self.merged_map.chunks[chunk_coord]
        mask = (new_chunk.data != -1) & ((merged_chunk.data == -1) | (new_chunk.data > merged_chunk.data))
        if np.any(mask):
            merged_chunk.data[mask] = new_chunk.data[mask]
            merged_chunk.is_empty = False
            merged_chunk.need_update = True
            self.updated_chunks.add(chunk_coord)

    def get_updated_chunks(self):
        updated = list(self.updated_chunks)
        self.updated_chunks.clear()
        return updated

class MapVisualizer:
    def __init__(self, map_merger):
        self.map_merger = map_merger
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.im = None
        self.cmap = self._create_custom_colormap()

    def _create_custom_colormap(self):
        return LinearSegmentedColormap.from_list("custom", 
                                                 [(0, 'gray'),    # Unknown
                                                  (0.5, 'white'),  # Free
                                                  (1, 'black')])   # Occupied

    def update_plot(self):
        merged_map = self.map_merger.merged_map
        grid = self._create_full_grid(merged_map)
        
        if self.im is None:
            self.im = self.ax.imshow(grid, cmap=self.cmap, interpolation='nearest', vmin=-1, vmax=100)
            plt.colorbar(self.im)
        else:
            self.im.set_data(grid)
        
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    def _create_full_grid(self, map):
        chunks = map.chunks
        if not chunks:
            return np.full((map.chunk_size, map.chunk_size), -1)
        
        min_x = min(x for x, _ in chunks.keys())
        max_x = max(x for x, _ in chunks.keys())
        min_y = min(y for _, y in chunks.keys())
        max_y = max(y for _, y in chunks.keys())
        
        width = (max_x - min_x + 1) * map.chunk_size
        height = (max_y - min_y + 1) * map.chunk_size
        
        grid = np.full((height, width), -1)
        
        for (cx, cy), chunk in chunks.items():
            x = (cx - min_x) * map.chunk_size
            y = (cy - min_y) * map.chunk_size
            grid[y:y+map.chunk_size, x:x+map.chunk_size] = chunk.data
        
        return grid

    def highlight_changes(self, updated_chunks):
        merged_map = self.map_merger.merged_map
        min_x = min(x for x, _ in merged_map.chunks.keys())
        min_y = min(y for _, y in merged_map.chunks.keys())
        
        for cx, cy in updated_chunks:
            x = (cx - min_x) * merged_map.chunk_size
            y = (cy - min_y) * merged_map.chunk_size
            rect = plt.Rectangle((x, y), merged_map.chunk_size, merged_map.chunk_size, 
                                 fill=False, edgecolor='red', linewidth=2)
            self.ax.add_patch(rect)
        
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

def create_random_map(width, height):
    map = OccupancyGridMap()
    for y in range(height):
        for x in range(width):
            value = np.random.choice([-1, 0, 100], p=[0.1, 0.8, 0.1])
            map.update_cell(x, y, value)
    return map

def main():
    base_map = create_random_map(500, 500)
    merger = MapMerger(base_map)
    visualizer = MapVisualizer(merger)

    visualizer.update_plot()

    for _ in range(5):  # Simulate 5 incremental updates
        new_map = create_random_map(500, 500)
        merger.merge_map(new_map)
        updated_chunks = merger.get_updated_chunks()
        
        visualizer.update_plot()
        visualizer.highlight_changes(updated_chunks)

    plt.show()

if __name__ == "__main__":
    main()