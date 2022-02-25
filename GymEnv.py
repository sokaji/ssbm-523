import os
import random

import gym
import melee
import numpy as np
import wandb
from gym.spaces import Box, Discrete

import Args
import GameManager
import MovesList
from Agent import Agent
from ModelCreator import Algorithm


class SmashEnv(gym.Env):

    def __init__(self, wandb):
        self.wandb = wandb
        args = Args.get_args()
        character = melee.Character.FOX
        opponent = melee.Character.CPTFALCON if not args.compete else character

        self.game = GameManager.Game(args)
        self.game.enterMatch(cpu_level=args.cpu_level if not args.compete else 0, opponant_character=opponent,
                             player_character=character,
                             stage=melee.Stage.FINAL_DESTINATION)

        self.agent = Agent(player_port=args.port, opponent_port=args.opponent, game=self.game, use_wandb=args.wandb,
                           algorithm=Algorithm.DQN)

        self.prev_gamestate = self.game.get_gamestate()
        obs = self.agent.get_observation(self.prev_gamestate)

        self.observation_space = Box(low=-1, high=1, shape=[len(obs)])
        self.action_space = Discrete(len(MovesList.moves_list))

        self.step_counter = 0

    def step(self, action):
        self.step_counter += 1
        self.agent.controller.act(action_index=action)
        new_gamestate = self.game.get_gamestate()
        new_obs = self.agent.get_observation(new_gamestate)
        reward = self.agent.get_reward(new_gamestate, self.prev_gamestate)
        self.agent.update_kdr(new_gamestate, self.prev_gamestate)
        self.prev_gamestate = new_gamestate
        wandb.log({'KDR': np.sum(self.agent.kdr), 'Reward': reward})
        return new_obs, reward, self.step_counter % (60*10) == 0, {}

    def reset(self):
        self.prev_gamestate = self.game.get_gamestate()
        new_obs = self.agent.get_observation(self.prev_gamestate)
        return new_obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    env = SmashEnv()
    check_env(env)
