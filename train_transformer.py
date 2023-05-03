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
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
import tensorflow_addons as tfa

import pickle

import Args
from DataHandler import get_ports, controller_states_different, generate_input, generate_output
import MovesList

args = Args.get_args()
sequence_length = 5
feature_length = 37  # Adjust this according to your input data shape
lr = 5e-5


class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, d_model, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.attention_layer = tfa.layers.MultiHeadAttention(
            num_heads=num_heads,
            head_size=d_model,
            dropout=dropout_rate
        )

    def call(self, inputs, training=None):
        q, k, v = inputs, inputs, inputs
        return self.attention_layer([q, k, v], training=training)


def transformer_block(inputs, num_heads, d_model, dense_dim, dropout_rate):
    attention = MultiHeadSelfAttention(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)(inputs)
    attention = Dropout(dropout_rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    feed_forward = Dense(dense_dim, activation='relu')(attention)
    feed_forward = Dropout(dropout_rate)(feed_forward)
    feed_forward = Dense(d_model)(feed_forward)
    feed_forward = Dropout(dropout_rate)(feed_forward)
    feed_forward = LayerNormalization(epsilon=1e-6)(attention + feed_forward)

    return feed_forward


def create_model(X: np.ndarray, Y: np.ndarray, player_character: melee.Character, opponent_character: melee.Character,
                 stage: melee.Stage,
                 folder: str, lr: float, multi: bool):
    print(len(X), len(Y))
    print(len(X[0]), len(Y[0]))

    # Reshape your input data if needed (X)
    X = X.reshape((-1, sequence_length, feature_length))
    # create transformer model: 
    batch_size = 32  # Set the batch size you want to use for training
    inputs = Input(shape=(sequence_length, feature_length))
    #inputs = Input(shape=(sequence_length, feature_length), batch_size=batch_size)
    x = inputs

    num_transformer_blocks = 10

    for _ in range(num_transformer_blocks):
        x = transformer_block(x, num_heads=8, d_model=37, dense_dim=512, dropout_rate=0.1)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(len(Y[0]), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
  
    #SGD with momentum
    opt = SGD(
        learning_rate=lr,
        momentum=0.9,  # adjust the momentum value as needed
        name="SGD",
        clipvalue=1.0
    )

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        X,  # training data
        Y,  # training targets
        epochs=200,  # number of epochs
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
    #player_character = melee.Character.PIKACHU
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
    #opponent_characters = [melee.Character.FOX]
    opponent_characters = [melee.Character.FALCO]
    stage = melee.Stage.FINAL_DESTINATION
    

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
    
    # Find the number of samples in your dataset
    num_samples = X.shape[0]

    # Find the next multiple of sequence_length
    next_multiple = (num_samples + sequence_length - 1) // sequence_length * sequence_length

    # Calculate how many samples you need to pad
    pad_samples = next_multiple - num_samples
    print(X.shape)
    # Pad X with zeros to make it divisible by sequence_length
    X = np.pad(X, ((0, pad_samples), (0, 0)), mode='constant')
    Y = np.pad(Y, ((0, pad_samples), (0, 0)), mode='constant')
    print(X.shape)
    # Now reshape X without errors
    X = X.reshape((-1, sequence_length, feature_length))

    # Create a new Y array corresponding to the last element of each sequence in X
    Y = Y[sequence_length - 1::sequence_length, :]

    if len(opponent_characters) > 1:
        opponent_character = melee.Character.POPO
    else: 
        opponent_character = opponent_characters[0]

    create_model(X, Y, player_character=player_character,
                 opponent_character=opponent_character, stage=stage, folder='models2', lr=lr, multi=multiAgent)
    
