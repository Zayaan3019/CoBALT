class AdvancedRewardShaping:
    def calculate_reward(self, robot_state, action, collision):
        reward = 0
        
        # Core rewards
        if robot_state.delivered:
            reward += 10  # Base delivery reward
            reward += 5 * (1 - robot_state.delivery_time/60)  # Time bonus
            
        if collision:
            reward -= 10  # Increased collision penalty
            
        # Motion efficiency
        angular_penalty = -0.1 * abs(action[1])  # Penalize sharp turns
        path_deviation = -0.05 * self._path_deviation(robot_state)
        
        # Energy conservation
        energy_cost = -0.01 * np.linalg.norm(action)
        
        return reward + angular_penalty + path_deviation + energy_cost

    def _path_deviation(self, robot_state):
        # Calculate deviation from ACO path
        return np.mean(np.abs(robot_state.actual_path - robot_state.planned_path))
