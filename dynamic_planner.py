class DynamicEnvironmentManager:
    def __init__(self, grid_size=(100,100)):
        self.grid = np.zeros(grid_size)
        self.obstacle_tracker = DynamicObstacleTracker()
        self.path_cache = LRUCache(maxsize=1000)
        
    def update_environment(self, lidar_data):
        obstacles = self.obstacle_tracker.process_lidar(lidar_data)
        self.grid = self._update_grid(obstacles)
        return obstacles

class DynamicObstacleTracker:
    def process_lidar(self, scan):
        # Convert 360° LiDAR to obstacle coordinates
        angles = np.linspace(0, 2*np.pi, len(scan.ranges))
        obstacles = []
        for r, theta in zip(scan.ranges, angles):
            if r < scan.range_max:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                obstacles.append((x, y))
        return obstacles
