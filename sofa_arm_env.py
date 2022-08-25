import random
import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fk_net import FK_Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SofaArmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SofaArmEnv, self).__init__()
        # Define action and observation space
        # The action will be tha displacement in the cables
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # observation or the state of the env will be the tip position, goal position, cables' displacement 
        # Eff_X, Eff_Y, Eff_Z, g_X, g_Y, g_Z, c_L0, c_L1, c_L2, c_L3, c_S0, c_S1, c_S2, c_S3
        self.observation_space = spaces.Box(
            low=np.array([-150.0, -150.0, -30.0, -150.0, -150.0, -30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            high=np.array([150.0, 150.0, 195.0, 150.0, 150.0, 195.0, 60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0, 40.0]), 
            shape=(14,),
            dtype=np.float32
        )

        self.FK_model = FK_Net().to(device)
        checkpoint_file = torch.load('./weights/FK/model_FK_tip-disp.pt')
        self.FK_model.load_state_dict(checkpoint_file['model_state_dict'])

        # dataframe that contains all the possible goals
        self.goals_df = pd.read_csv('./goals_xyz.csv')
        # data to normalize the input to the FK model
        self.disp_range = np.array([60, 60, 60, 60, 40, 40, 40, 40]).astype(np.float32)
        self.data_ranges = np.load('./coordinates_ranges_dict.npy', allow_pickle=True).tolist()
        x_min = self.data_ranges['x_min']
        x_max = self.data_ranges['x_max']
        y_min = self.data_ranges['y_min']
        y_max = self.data_ranges['y_max']
        z_min = self.data_ranges['z_min']
        z_max = self.data_ranges['z_max']
        self.tip_min = np.array([x_min, y_min, z_min]).astype(np.float32)
        self.tip_max = np.array([x_max, y_max, z_max]).astype(np.float32)
        # set the done as False
        self.done = False 
        self.dummy_count = 1

    def step(self, delta_action):
        self.dummy_count += 1
        if self.dummy_count % 1000 == 0:
            print('1000 step')
        # action taken by the policy will be the delta how much increase or decrease each cable
        # the action taken will be in range [-1, 1] and we need to scale it to [-3, 3] 
        scaled_delta_action = delta_action * 3
        # add delta action to the current action
        self.action = self.action + scaled_delta_action
        # clip the action to the upper and lower limits
        self.action = np.clip(self.action, a_min=[0, 0, 0, 0, 0, 0, 0, 0], a_max=[60, 60, 60, 60, 40, 40, 40, 40])
        # normalize the action 
        action_nrom = self.action / self.disp_range
        # normalize the tip position
        tip_pos_norm = (self.tip_pos - self.tip_min) / (self.tip_max - self.tip_min)
        # stack them together to input to the FK model
        model_input = np.hstack([tip_pos_norm, action_nrom]).astype(np.float32)
        model_input = torch.tensor(model_input).to(device)
        self.new_tip_pos = self.FK_model(model_input)
        # The output of the FK model is the new tip position
        self.new_tip_pos = self.new_tip_pos.detach().cpu().numpy()
        # The observation or the state is the stacking of the goal position, tip position and the actuations of the cables
        observation = np.hstack([self.goal_pos, self.new_tip_pos, self.action]).astype(np.float32)
        # the reward is the negative distance between the tip position and the goal
        reward = self.get_reward(self.new_tip_pos)
        # update the tip position for the next step
        self.tip_pos = self.new_tip_pos
        if self.distance(self.new_tip_pos) < 5:
            self.done = True
        else: 
            self.done = False
        info = {}
        return observation, reward, self.done, info
    
    def reset(self):
        x, y, z = self.goals_df.iloc[random.randrange(0,len(self.goals_df)),:]
        self.goal_pos = [x, y, z]
        self.tip_pos = [0, 0, 195]
        self.action = [0, 0, 0, 0, 0, 0, 0, 0]
        observation = np.hstack([self.goal_pos, self.tip_pos, self.action]).astype(np.float32)
        self.done = False
        return observation  # reward, done, info can't be included

    def get_reward(self, tip_pos):
        current_distance = self.distance(tip_pos)
        return - current_distance 

    def distance(self, tip_pos):
        eff_goal_dist = np.sqrt((self.goal_pos[0] - tip_pos[0]) ** 2 +
                                (self.goal_pos[1] - tip_pos[1]) ** 2 +
                                (self.goal_pos[2] - tip_pos[2]) ** 2) 
        return eff_goal_dist

    def close(self):
        print('close method called')