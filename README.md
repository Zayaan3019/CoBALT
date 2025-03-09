# CoBALT - Collaborative Bots for Automated Logistics and Transport

CoBALT is an intelligent, multi-agent system designed to optimize warehouse logistics and swarm robotics. By combining Reinforcement Learning (TD3) with Ant Colony Optimization (ACO), CoBALT enables efficient task allocation, path planning, and dynamic coordination among autonomous bots in real-time. This hybrid approach leverages the strengths of ACO for global optimization and TD3 for adaptive decision-making in dynamic environments, making it a cutting-edge solution for warehouse automation.

# Features
1. Hybrid Optimization Framework: Combines ACO for global path optimization with TD3 reinforcement learning for adaptive decision-making.
2. Dynamic Task Allocation: Efficiently assigns tasks to bots based on workload, location, and priority using the dynamic_planner.py module.
3. Swarm Coordination: Implements decentralized communication among bots using swarm_coordinator.py and swarm_node.py.
4. Simulation Environment: Includes a ROS-compatible simulate.launch file for testing in a virtual warehouse simulation.
5. Visualization Tools: Real-time visualization of bot movements and task execution via visualization.py.
6. Reward Mechanism: Custom reward functions (reward.py) to optimize bot behavior for task completion, collision avoidance, and energy efficiency.

# Technologies Used
1. Reinforcement Learning: Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm combined with ACO implemented in warehouse_rl.py and trained the model using OpenAI Gym.
2. Ant Colony Optimization (ACO): Hybridized with RL for global path planning in hybrid_aco_td3.py.
3. Python ecosystem: Core logic implemented in Python with libraries like NumPy, Matplotlib, and PyTorch.
4. Gazebo ROS: Used for simulating multi-agent systems and coordinating swarm behaviour.
5. Visualization Tools: Real-time data visualization using Python-based plotting libraries.

# Installation & Setup

1. Clone the repository:
git clone https://github.com/your-repo/RL-ACO.git

2. Set up the ROS environment:
Ensure ROS is installed and sourced correctly.
Add the project to your ROS workspace.

3. Launch the simulation:
roslaunch simulate.launch


4. Run the main program:
python main.py

# Usage
1. Start the simulation using the provided ROS launch file.
2. Use main.py to initialize the hybrid ACO-TD3 framework.
3. Monitor bot coordination and task execution via real-time visualization (visualization.py).
4. Modify task allocation or reward parameters in dynamic_planner.py or reward.py to experiment with different strategies.

# Project Highlights
1. Hybrid Algorithm Design: Seamlessly integrates ACO's global search capabilities with TD3's local adaptability for real-world performance.
2. Multi-Agent Swarm Coordination: Implements decentralized communication protocols for scalable task management.
3. Dynamic Simulation Environment: Enables testing of various configurations in a realistic virtual warehouse setup.

Future Enhancements
Extend support for larger-scale warehouses with hundreds of bots.

Integrate advanced RL algorithms like PPO or SAC for enhanced learning efficiency.

Add predictive analytics to forecast task completion times based on historical data.
