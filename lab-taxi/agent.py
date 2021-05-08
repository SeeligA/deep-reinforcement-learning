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
        self.alpha = 0.2
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

        def decode(i):
            out = []
            out.append(i % 4)
            i = i // 4
            out.append(i % 5)
            i = i // 5
            out.append(i % 5)
            i = i // 5
            out.append(i)
            assert 0 <= i < 5
            return out[::-1]

        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        s = decode(state)
        taxi_loc = tuple(s[:2])
        pass_idx = s[2]
        dest_idx = s[3]

        if pass_idx == 4:
            if taxi_loc == locs[dest_idx]:
                 return 5

        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])

        return np.random.choice(4)

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
            probs = np.ones(self.nA) * self.epsilon / self.nA
            probs[np.argmax(self.Q[next_state])] += 1 - self.epsilon
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.dot(probs, self.Q[next_state]) - self.Q[state][action])


        self.episode += 1