def train_swarm():
    # Initialize components
    env = WarehouseEnvironment()
    model = HybridACOTD3(state_dim=24, action_dim=2, num_robots=5)
    visualizer = SwarmVisualizer()
    coordinator = SwarmORCA(num_robots=5)
    
    # Hyperparameters
    config = {
        'episodes': 2000,
        'batch_size': 1024,
        'exploration': DecaySchedule(1.0, 0.1, 1000),
        'aco_update_interval': 10
    }
    
    for episode in range(config['episodes']):
        states = env.reset()
        episode_reward = 0
        
        while not env.done:
            # Dynamic environment updates
            obstacles = env.update_obstacles()
            
            # Hybrid path planning
            aco_paths = model.aco.plan_paths(states, env.targets)
            
            # RL action selection
            actions = model(states)
            
            # Swarm coordination
            safe_actions = coordinator.resolve_conflicts(states, actions)
            
            # Environment step
            next_states, rewards, done = env.step(safe_actions)
            
            # Store experiences
            model.replay_buffer.add(states, actions, rewards, next_states, done)
            
            # Train agents
            if len(model.replay_buffer) > config['batch_size']:
                model.update_networks(config['batch_size'])
            
            # Visualization
            visualizer.update_display(states, model.aco.pheromones, aco_paths)
            
            states = next_states
            episode_reward += sum(rewards)
            
        # Periodic updates
        if episode % config['aco_update_interval'] == 0:
            model.aco.update_pheromones(aco_paths, env.collision_stats)
            
        # Adaptive exploration
        exploration_rate = config['exploration'].value(episode)
        model.set_exploration(exploration_rate)
