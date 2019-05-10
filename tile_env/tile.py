import sys
from enum import Enum
import random
from gym import spaces
import gym
import numpy as np
import pdb
from itertools import permutations

U = 0
D = 1
L = 2
R = 3

ACTION_MAP = {
    U: (-1, 0),
    D: (1, 0),
    L: (0, -1),
    R: (1, 0)
}

def solveable(env):
    '''
    env: TileEnv
    A puzzle configuration is solveable if the sum of the permutation parity and the L1 distance of the
    empty tile to the corner location is even.
    '''
    parity = 0 if even_perm(env.perm_state()) else 1
    l1_dist = (env.n - 1 - env._empty_x + env.n - 1 - env._empty_y)
    return ((l1_dist + parity) % 2 == 0)

def n_inversions(perm):
    '''
    perm: list/tuple of ints
    Returns: number of inversions of the given permutation
    '''
    n_invs = 0
    for idx, x in enumerate(perm):
        for ridx in range(idx+1, len(perm)):
            if x > perm[ridx]:
                n_invs += 1

    return n_invs

def even_perm(perm):
    '''
    perm: iterable
    Returns: True if perm is an even permutation
    '''
    return ((n_inversions(perm) % 2) == 0)

def random_alternating_perm(n):
    '''
    n: int, size of permutation
    Returns: A random permutation of A_n
    '''
    x = list(range(1, n * n + 1))
    random.shuffle(x)
    while not even_perm(x):
        random.shuffle(x)

    return x

class TileEnv(gym.Env):
    def __init__(self, n):
        self.grid = np.array([i+1 for i in range(n * n)], dtype=int).reshape(n, n)
        self.n = n
        self.action_space = spaces.Discrete(4)

        self._initted = False
        self._empty_x = n - 1
        self._empty_y = n - 1
        self.reset()

    @property
    def actions(self):
        return self.action_space.n

    def _inbounds(self, x, y):
        return (0 <= x <= (self.n - 1)) and (0 <= y <= (self.n - 1))

    def step(self, action, ignore_oob=True):
        '''
        Actions: U/D/L/R
        Move swap the empty tile with the tile in the given location
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        dx, dy = ACTION_MAP[action]
        new_x = self._empty_x + dx
        new_y = self._empty_y + dy

        # what should we do with illegal moves?
        oob = not self._inbounds(new_x, new_y)
        if oob:
            if ignore_oob:
                done = self.is_solved()
                reward = 1 if done else 0
                return self.grid, reward, done, {}
            else:
                raise Exception('Taking action {} will take you out of bounds'.format(action))

        self.grid[new_x, new_y], self.grid[self._empty_x, self._empty_y] = self.grid[self._empty_x, self._empty_y], self.grid[new_x, new_y]
        self._empty_x = new_x
        self._empty_y = new_y

        done = self.is_solved()
        reward = 1 if done else 0
        return self.grid, reward, done, {}

    def _pp(self, x):
        if self.n <= 3:
            if x != (self.n * self.n):
                return '[{}]'.format(x)
            return '[_]'
        else:
            if x != (self.n * self.n):
                return '[{:2}]'.format(x)
            return '[__]'


    def render(self):
        for r in range(self.n):
            row = self.grid[r, :]
            for x in row:
                print(self._pp(x), end='')
            print()

    def reset(self, nmoves=1000):
        '''
        Scramble the tile puzzle by taking some number of random moves
        This is actually really quite bad at scrambling
        '''
        self._initted = True
        perm = random_alternating_perm(self.n)
        self._assign_perm(perm)
        return self.grid

    def _assign_perm(self, perm):
        self.grid = np.array(perm, dtype=int).reshape(self.n, self.n)

    @staticmethod
    def from_perm(perm):
        '''
        perm: tuple/list of ints
        Ex: The identity permutation for n = 4 is: (1, 2, 3, 4) and will yield the grid:
                [1][2]
                [3][4]
        '''
        n = int(np.sqrt(len(perm)))
        env = TileEnv(n)
        env._initted = True
        env._assign_perm(perm)

        return env

    def is_solved(self):
        # 1-indexed
        idx = 1
        for i in range(self.n):
            for j in range(self.n):
                if self.grid[i, j] != idx:
                    return False

        return True

    def perm_state(self):
        return self.grid.ravel()

def grid_to_tup(grid):
    '''
    Get the permutation tuple representation of a grid
    Elements of grid are 1-indexed (contains values 1 - n**2), where n is the num rows of grid
    Returns tuple where index i of the tuple is where element i got mapped to

    Ex:
    (1, 2, 3, 4) is the permutation tuple corresponding to the grid:
    [1, 2]
    [3, 4]

    (3, 2, 4, 1) is the permutation corresponding to the grid:
    [4, 2]
    [1, 3]
    '''
    locs = [0] * grid.size

    for i in range(len(grid)):
        for j in range(len(grid)):
            x = grid[i, j]
            idx = (i * n) + j
            locs[x - 1] = idx

if __name__ == '__main__':
    n = 3 if len(sys.argv) < 2 else int(sys.argv[1])
    env = TileEnv(n)
    env.reset()
    env.render()
