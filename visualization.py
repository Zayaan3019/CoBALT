class SwarmVisualizer:
    def __init__(self):
        self.markers_pub = rospy.Publisher('/swarm_markers', MarkerArray, queue_size=10)
        self.heatmap_pub = rospy.Publisher('/pheromone_heatmap', PointCloud2, queue_size=10)
        
    def update_display(self, states, pheromones, paths):
        self._publish_robot_markers(states)
        self._publish_pheromone_heatmap(pheromones)
        self._publish_aco_paths(paths)
        
    def _publish_robot_markers(self, states):
        marker_array = MarkerArray()
        for i, state in enumerate(states):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.pose.position = state.position
            marker.color = self._status_color(state)
            marker_array.markers.append(marker)
        self.markers_pub.publish(marker_array)
