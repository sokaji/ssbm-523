import math
import pickle
import time

import keras

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

import MovesList

import random
from TransformerBot import Bot
from keras.models import load_model
from keras.utils import custom_object_scope
from train_transformer import MultiHeadSelfAttention

args = Args.get_args()
smash_last = False

#player_character = melee.Character.PIKACHU
#opponent_character = melee.Character.FOX
player_character = melee.Character.FOX
opponent_character = melee.Character.FALCO
stage = melee.Stage.FINAL_DESTINATION
level=9

multAgent = True  ## dont forget to change this based on the type of fight


if __name__ == '__main__':

    if multAgent:
        file_name = f'models2/{player_character.name}_v_MANY_on_{stage.name}.h5'
    else:
        file_name = f'models2/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.h5'

    print(file_name)

    # Use the Keras load_model function to load the model
    #model = load_model(file_name)
    with custom_object_scope({'MultiHeadSelfAttention': MultiHeadSelfAttention}):
        model = load_model(file_name)
    
    game = GameManager.Game(args)
    game.enterMatch(cpu_level=level, opponant_character=opponent_character,
                    player_character=player_character,
                    stage=stage, rules=False)

    bot1 = Bot(model=model, controller=game.controller, opponent_controller=game.opponent_controller)
    # bot2 = Bot(model=model, controller=game.opponent_controller, opponent_controller=game.controller)

    while True:
        gamestate = game.get_gamestate()
        bot1.act(gamestate)
        # bot2.act(gamestate)