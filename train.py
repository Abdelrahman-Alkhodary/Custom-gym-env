import numpy as np
from sofa_arm_env import SofaArmEnv

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

'''
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/home/abdelrahman/MyProject/gym/sentdex/logs/',
                                         name_prefix='rl_model')
'''
env = SofaArmEnv()
env.reset()

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1, reset_num_timesteps=False)

episdoes = 10
steps = 0
for ep in range(episdoes):
    print(f'episode: {ep}, with step: {steps}')
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps <= 300:
        action, states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)
        steps += 1