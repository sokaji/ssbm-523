import time

import gym
import random
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import utils
import copy

# class Action:
#     def __init__(self, ):
#

class FoxEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game):
        self.num_actions = 5

        self.stage = melee.Stage.BATTLEFIELD

        self.game = game

        self.controller: melee.Controller = self.game.getController(player_port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)

        super(FoxEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.console.step()
        self.old_gamestate = self.game.console.step()

        nun_inputs = self.get_observation(self.gamestate).shape[0]
        self.observation_space = spaces.Box(shape=np.array([nun_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

        self.rewards = []

        self.move_x = 0

        self.kills = 0
        self.deaths = 0

        self.obs = self.reset()

    def step(self, action: int):
        self.kills = 0
        self.deaths = 0
        self.queue_action(action)
        self.old_gamestate = self.gamestate
        self.gamestate = self.game.console.step()

        obs = self.get_observation(self.gamestate)


        r = self.calculate_reward()
        return [obs, r, False, {}]

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        opponent_action = opponent.action
        self_action = player.action

        opponent_attacking = 1 if opponent_action in utils.attacking_list else 0
        self_attacking = 1 if self_action in utils.attacking_list else 0

        player_facing = 1 if player.facing else -1
        opponent_facing = 1 if player.facing else -1

        opponent_vel_x = max((opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack)/1000, 1)
        opponent_vel_y = max((opponent.speed_y_self + opponent.speed_y_attack)/1000, 1)

        self_vel_x = max((player.speed_air_x_self + player.speed_ground_x_self + player.speed_x_attack)/1000, 1)
        self_vel_y = max((player.speed_y_self + player.speed_y_attack)/1000, 1)

        blastzones = melee.BLASTZONES.get(self.stage)


        obs = np.array([player.x/blastzones[1], player.y/blastzones[2], opponent.x/blastzones[1], opponent.y/blastzones[2], player_facing, opponent_attacking, opponent_facing, self_attacking,opponent_vel_x, opponent_vel_y, self_vel_x, self_vel_y])
        # print(obs)

        return obs

    def calculate_reward(self):
        old_gamestate = self.old_gamestate
        new_gamestate = self.gamestate

        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        distance = math.dist((new_player.x, new_player.y), (new_opponent.x, new_opponent.y))

        damage_dealt = max(0, new_opponent.percent - old_opponent.percent)
        damage_recieved = max(0, new_player.percent - old_player.percent)


        blast_thresh = 30
        blastzones = melee.BLASTZONES.get(self.stage)
        # deaths = 1 if new_player.percent < old_player.percent or math.fabs(new_player.x) > blastzones[1] - blast_thresh or new_player.y > blastzones[2] - blast_thresh or new_player.y < blastzones[3] + blast_thresh else 0
        # kills = 1 if new_opponent.percent < old_opponent.percent or math.fabs(new_opponent.x) > blastzones[1] - blast_thresh or new_opponent.y > blastzones[2] - blast_thresh or new_opponent.y < blastzones[3] + blast_thresh else 0
        if new_player.action == melee.Action.ON_HALO_DESCENT and self.deaths == 0:
            self.deaths = 1
        if new_opponent.action == melee.Action.ON_HALO_DESCENT and self.kills == 0:
            self.kills = 1

        # print(deaths)

        reward = -distance/1000 + (damage_dealt - damage_recieved) / 5000 + self.kills * 0.7 - self.deaths * 1

        sigmoid_reward = 2/(1 + math.pow(math.e, -4.4 * reward)) -1


        return sigmoid_reward

    def reset(self):
        self.old_gamestate = self.game.console.step()
        return self.get_observation(self.old_gamestate)

    def queue_action(self, action: int):
        # self.controller.release_all()
        # print(action)
        def frame_delay(num_frames):
            if num_frames > 1:
                num_frames -= 1
            for i in range(num_frames):
                game_state = self.game.console.step()
                new_player: melee.PlayerState = game_state.players.get(self.player_port)
                new_opponent: melee.PlayerState = game_state.players.get(self.opponent_port)
                if new_player.action == melee.Action.ON_HALO_DESCENT and self.deaths == 0:
                    self.deaths = 1
                if new_opponent.action == melee.Action.ON_HALO_DESCENT and self.kills == 0:
                    self.kills = 1

            # fps = 60
            # time.sleep(num_frames/fps)

        def flick_button(button, num_frames):
            self.controller.press_button(button)
            self.controller.flush()
            frame_delay(num_frames)
            self.controller.release_button(button)
            self.controller.flush()


        def flick_axis(button, x, y, num_frames):
            self.controller.tilt_analog_unit(button, x, y)
            frame_delay(num_frames)
            self.controller.tilt_analog_unit(button, 0, 0)

        def button_axis(button, axis, x, y, num_frames):
            self.controller.tilt_analog_unit(axis, x, y)
            frame_delay(1)
            self.controller.press_button(button)
            self.controller.flush()

            frame_delay(num_frames)

            self.controller.release_button(button)
            self.controller.tilt_analog_unit(axis, 0, 0)

            self.controller.flush()

        move_stick = melee.Button.BUTTON_MAIN

        if action == 0:  # Move Left
            flick_axis(move_stick, -1, 0, num_frames=10)
        elif action == 1:  # Move Right
            flick_axis(move_stick, 1, 0, num_frames=10)
        elif action == 2:  # Jump
            flick_button(melee.Button.BUTTON_Y, num_frames=5)

        elif action == 3:  # Drop
            flick_axis(move_stick, 0, -1, num_frames=10)

        elif action == 4:  # Drop
            flick_button(melee.Button.BUTTON_A, num_frames=35)