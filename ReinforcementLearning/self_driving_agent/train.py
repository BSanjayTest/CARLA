import torch
import sys

# Allow running specific scenario via command line argument
# Usage: python train.py ClearNight
# Or: python train.py (uses config.py settings)
scenario_arg = sys.argv[1] if len(sys.argv) > 1 else None

# Show available scenarios if requested
if len(sys.argv) > 1 and sys.argv[1] == '--help':
    from config import WEATHER_PRESETS
    print("Available weather scenarios:")
    for s in WEATHER_PRESETS.keys():
        print(f"  - {s}")
    print("\nUsage: python train.py <scenario>")
    print("Example: python train.py ClearNight")
    sys.exit(0)

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import action_map, env_params, TRAINING_SCENARIOS, TRAINING_SCENARIO, WEATHER_PRESETS
from utils import *
from environment import SimEnv

def run():
    """Train the agent on one or more weather scenarios"""
    env = None
    try:
        # Determine scenarios to train on
        if scenario_arg and scenario_arg in WEATHER_PRESETS:
            scenarios = [scenario_arg]
            print(f"Running single scenario from command line: {scenario_arg}")
        else:
            scenarios = TRAINING_SCENARIOS if isinstance(TRAINING_SCENARIOS, list) else [TRAINING_SCENARIO]
            print(f"Training on scenarios: {scenarios}")
        
        # Create weights directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        
        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        in_channels = 1
        episodes = 500  # Good balance for faster results

        # Initialize model and replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)
        
        # Training loop for each scenario
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Starting training for scenario: {scenario}")
            print(f"{'='*60}\n")
            
            # Create scenario-specific weights directory
            scenario_weights_dir = os.path.join('weights', scenario)
            os.makedirs(scenario_weights_dir, exist_ok=True)
            
            # Create scenario-specific log file
            log_file = f'training_log_{scenario}.csv'
            
            # Try to load existing model for this scenario
            import glob
            scenario_models = glob.glob(os.path.join(scenario_weights_dir, 'model_ep_*_Q'))
            
            start_ep = 0
            if scenario_models:
                # Find the model with highest episode number for this scenario
                model_numbers = []
                for m in scenario_models:
                    try:
                        basename = os.path.basename(m)
                        parts = basename.split('_')
                        num = int(parts[2])
                        model_numbers.append(num)
                    except:
                        continue
                if model_numbers:
                    latest_ep = max(model_numbers)
                    model_path = os.path.join(scenario_weights_dir, f'model_ep_{latest_ep}')
                    try:
                        model.load(model_path)
                        print(f"Resuming training for {scenario} from episode {latest_ep}")
                        start_ep = latest_ep + 1
                    except Exception as e:
                        print(f"Could not load model: {e}")
            
            # Set up environment params for this scenario
            scenario_env_params = env_params.copy()
            scenario_env_params['start_ep'] = start_ep
            scenario_env_params['weather_preset'] = scenario  # Use the specific scenario
            
            # Override log file for this scenario
            scenario_env_params['log_file'] = log_file
            
            # Force a pre-cleanup of the simulator to avoid connection/sensor crashes
            print(f"Initializing environment for {scenario}...")
            env = SimEnv(**scenario_env_params)
            
            for ep in range(start_ep, episodes):
                env.create_actors()
                should_continue = env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
                env.reset()
                if should_continue is False:
                    break
            
            # Save final model for this scenario
            if 'ep' in locals():
                model.save(os.path.join(scenario_weights_dir, 'model_ep_{}'.format(ep)))
                print(f"Final model for {scenario} saved at episode {ep}!")
                print(f"Log file saved: {log_file}")
            else:
                print(f"No episodes were trained for {scenario}, skipping save.")
            
            # Clean up environment before next scenario
            if env is not None:
                try:
                    env.quit()
                except Exception as e:
                    print(f"Error during cleanup: {e}")
                env = None
        
        print(f"\n{'='*60}")
        print("Training complete for all scenarios!")
        print(f"Models saved in: weights/<scenario_name>/")
        print(f"Logs saved as: training_log_<scenario_name>.csv")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.quit()
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    run()