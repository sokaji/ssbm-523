#!/usr/bin/python3

# This code is modified to handle better models, plot loss and training accuracy, and run multiple epochs
import copy
from collections import deque

import melee

import os
import json
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import pickle

import Args
from DataHandler import get_ports, controller_states_different, generate_input, generate_output
import MovesList

args = Args.get_args()


def create_model(X: np.ndarray, Y: np.ndarray, player_character: melee.Character, opponent_character: melee.Character,
                 stage: melee.Stage,
                 folder: str, lr: float, multi: bool):
    print(len(X), len(Y))
    print(len(X[0]), len(Y[0]))

    # train
    #default model: 
    '''
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(len(X[0]),)),
        Dense(128, activation='tanh'),
        Dense(128, activation='tanh'),
        Dense(len(Y[0]), activation='tanh'),
    ])
    '''
    
    # ReLU Model
    '''
    model = Sequential([
        Dense(128, activation='relu', input_shape=(len(X[0]),)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(Y[0]), activation='relu'),
    ])
    '''
    # Batch Normalization
    #'''
    model = Sequential([
        Dense(2048, input_shape=(len(X[0]),)),
        BatchNormalization(),
        Activation('relu'),
        Dense(2048),
        BatchNormalization(),
        Activation('relu'),
        Dense(2048),
        BatchNormalization(),
        Activation('relu'),
        Dense(1024),
        BatchNormalization(),
        Activation('relu'),
        Dense(1024),
        BatchNormalization(),
        Activation('relu'),
        Dense(1024),
        BatchNormalization(),
        Activation('relu'),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dense(len(Y[0]), activation='softmax'),
    ])
    #'''
    

    #Dropout - Really bad results
    '''
    dropout_rate = 0.9
    model = Sequential([
        Dense(128, input_shape=(len(X[0]),)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        Dense(len(Y[0]), activation='relu'),
    ])
    '''

    #default optimizer
    #opt = keras.optimizers.Adam(
    #    learning_rate=lr,
    #    name="Adam",
    #)

    #optimizer with clipping
    #opt = keras.optimizers.Adam(
    #    learning_rate=lr,
    #    name="Adam",
    #    clipvalue=1.0
    #)

    #SGD with momentum
    opt = SGD(
        learning_rate=lr,
        momentum=0.9,  # adjust the momentum value as needed
        name="SGD",
        clipvalue=1.0
    )

    # default loss value
    #model.compile(
    #    optimizer=opt,
    #    loss='mean_squared_error',
    #    metrics=['accuracy'],
    #)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        X,  # training data
        Y,  # training targets
        epochs=5,  # number of epochs
        shuffle=True
    )

    # Change the file extension from .pkl to .h5
    if multi:
        model_file_path = f'{folder}/{player_character.name}_v_MANY_on_{stage.name}.h5'
    else:
        model_file_path = f'{folder}/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.h5'

    if not os.path.isdir(folder):
        os.mkdir(f'{folder}/')

    # Replace the pickle code with the save method from Keras
    model.save(model_file_path)

    plot_history(history)


# Introducing a loss and accuracy plot
def plot_history( history):
    # Loss plot
    plt.figure(1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.figure(2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    player_character = melee.Character.FOX
    '''
    opponent_characters = [melee.Character.FOX,
                            melee.Character.FALCO,
                            melee.Character.MARTH,
                            melee.Character.CPTFALCON,
                            melee.Character.DOC,
                            melee.Character.LUIGI,
                            melee.Character.PEACH,
                            melee.Character.GANONDORF,
                            melee.Character.NESS,
                            melee.Character.NESS,
                            melee.Character.SAMUS,
                            melee.Character.ZELDA,
                            melee.Character.LINK,
                            melee.Character.MEWTWO,
                            melee.Character.GAMEANDWATCH,
                            melee.Character.ROY]
    #'''
    opponent_characters = [melee.Character.FALCO]
    stage = melee.Stage.FINAL_DESTINATION
    lr = 5e-5

    multiAgent = True

    X = None
    Y = None

    for opponent in opponent_characters:
        print(f'Loading {opponent.name} Data...')
        raw = open(f'Data/{player_character.name}_{opponent.name}_on_{stage.name}_data.pkl', 'rb')
        data = pickle.load(raw)
        X_in = data['X']
        Y_in = data['Y']

        if X is not None:
            X = np.concatenate((X,X_in), axis=0)
            Y = np.concatenate((Y,Y_in), axis=0)
        else:
            X = X_in
            Y = Y_in
    
        print("  X:" + str(X.shape) + "  Y:" + str(Y.shape) + "  X2:" + str(X_in.shape) + "  Y2:" + str(Y_in.shape))
    #print("  X3:" + str(X3.shape) + "  Y3:" + str(Y3.shape) + "  X4:" + str(X2.shape) + "  Y4:" + str(Y2.shape))

    if len(opponent_characters) > 1:
        opponent_character = melee.Character.POPO
    else: 
        opponent_character = opponent_characters[0]

    create_model(X, Y, player_character=player_character,
                 opponent_character=opponent_character, stage=stage, folder='models2', lr=lr, multi=multiAgent)
    
