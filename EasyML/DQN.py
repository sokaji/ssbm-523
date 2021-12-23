import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import time

from collections import deque

import numpy as np


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, num_inputs, num_outputs, learning_rate=0.001, min_replay_size=10_000, max_replay_size=50_000,
                 minibatch_size=128, discount_factor=0.9, update_target_every=5):
        # Gets Trained
        self.model = self.create_model(num_inputs=num_inputs, num_outputs=num_outputs, learning_rate=learning_rate)

        # Gets predicted from
        self.target_model = self.create_model(num_inputs=num_inputs, num_outputs=num_outputs,
                                              learning_rate=learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        self.min_replay_size = min_replay_size
        self.replay_memory = deque(maxlen=max_replay_size)

        self.tensorboard = ModifiedTensorBoard(Log_dir=f"logs/dqn_model_{time.time}")

        self.target_update_counter = 0
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.update_target_every = update_target_every

    def create_model(self, num_inputs, num_outputs, learning_rate):
        model = Sequential()
        model.add(Dense(num_inputs))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_outputs, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory < self.min_replay_size):
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount_factor * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())