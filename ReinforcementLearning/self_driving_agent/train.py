import torch


from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import action_map, env_params
from utils import *
from environment import SimEnv

def run():
    env = None
    try:
        # Create weights directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        
        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        in_channels = 1
        episodes = 500  # Good balance for faster results

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)
        
        # Try to load existing model to resume training
        import glob
        models = glob.glob(os.path.join('weights', 'model_ep_*'))
        if models:
            # Find the model with highest episode number
            model_numbers = []
            for m in models:
                try:
                    num = int(os.path.basename(m).split('_')[-1].split('.')[0])
                    model_numbers.append(num)
                except:
                    continue
            if model_numbers:
                latest_ep = max(model_numbers)
                model_path = f'weights/model_ep_{latest_ep}'
                try:
                    model.load(model_path)
                    print(f"Resuming training from episode {latest_ep}")
                except Exception as e:
                    print(f"Could not load model: {e}")

        # Find starting episode from loaded model
        start_ep = 0
        if models and model_numbers:
            start_ep = max(model_numbers) + 1
        
        # Use correct start_ep based on loaded model
        env_params['start_ep'] = start_ep
        
        # Force a pre-cleanup of the simulator to avoid connection/sensor crashes
        print("Initializing environment...")
        env = SimEnv(**env_params)
        
        for ep in range(start_ep, episodes):
            env.create_actors()
            should_continue = env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            env.reset()
            if should_continue is False:
                break
        
        # Save final model with correct episode number
        if 'ep' in locals():
            model.save('weights/model_ep_{}'.format(ep))
            print(f"Final model saved at episode {ep}!")
        else:
            print("No episodes were trained, skipping save.")
        
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