# dqn action values
action_values = [-0.75, -0.5, -0.25, -0.15, -0.1, -0.05, 0,
                0.05, 0.1, 0.15, 0.25, 0.5, 0.75]
action_map = {i:x for i, x in enumerate(action_values)}

env_params = {
    'target_speed' :30 , 
    'max_iter': 1000,  # Reduced from 4000 for faster episodes
    'start_buffer': 5,  # Reduced from 10
    'train_freq': 1,
    'save_freq': 50,   # Save more frequently to see progress
    'start_ep': 0,
    'max_dist_from_waypoint': 20,
    'visuals': True,
    
    # Enhanced environment settings
    'num_other_vehicles': 15,  # Increased NPC vehicles
    'num_pedestrians': 20,     # Added pedestrians
    'weather_types': ['Fog'],  # Force Foggy weather
    'vehicle_types': [  # Available vehicle types for variety
        'vehicle.nissan.patrol',
        'vehicle.tesla.model3', 
        'vehicle.ford.mustang',
        'vehicle.audi.tt',
        'vehicle.nissan.micra'
    ]
}