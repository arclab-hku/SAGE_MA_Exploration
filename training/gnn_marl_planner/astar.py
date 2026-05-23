import heapq

def astar(map_data, current_pos):
    """
    A* algorithm for searching the nearest unknown area in the map.

    Args:
    - map_data (numpy array): The map data, where -1 represents unknown area, 0 represents blank area, and 100 represents obstacle.
    - current_pos (tuple of two int): The current position of the agent.

    Returns:
    - path (list of tuples of two int): The path from the current position to the nearest unknown area.
    """
    # 前置检查：当前位置即未知区域
    if map_data[current_pos[0], current_pos[1]] == -1:
        return None
    
    # Define the possible movements
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Create a priority queue and add the current position
    queue = [(0, current_pos)]
    came_from = {current_pos: None}
    
    # Create a set to store the visited positions
    visited = set()
    
    while queue:
        # Get the position with the lowest cost
        cost, pos = heapq.heappop(queue)
        
        # Mark the position as visited
        visited.add(pos)
        
        # Explore the neighbors of the position
        for movement in movements:
            new_pos = (pos[0] + movement[0], pos[1] + movement[1])
            if (0 <= new_pos[0] < map_data.shape[0] and 0 <= new_pos[1] < map_data.shape[1] and
                    map_data[new_pos[0], new_pos[1]] != 100 and new_pos not in visited):
                # Calculate the cost of the new position
                new_cost = cost + 1
                # Add the new position to the queue
                heapq.heappush(queue, (new_cost, new_pos))
                # Update the came_from dictionary
                came_from[new_pos] = pos
        
        # If the position is the goal, return the path
        if map_data[pos[0], pos[1]] == -1:
            path = []
            while pos is not None:
                path.append(pos)
                pos = came_from[pos]
            return path[::-1]
    
    # If no path is found, return None
    return None
