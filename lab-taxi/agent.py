import random

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.alpha = 1.0
        self.gamma = 1.0
        self.episode = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])

        return np.random.choice(self.nA)


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.epsilon /= self.episode

        if done:
            self.Q[state][action] += self.alpha * reward - self.Q[state][action]
        else:
            self.Q[state][action] += self.alpha * (
                reward + self.gamma * self.Q[next_state][np.argmax(self.Q[next_state])] - self.Q[state][action])

        self.episode += 1