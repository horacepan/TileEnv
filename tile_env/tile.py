import sys
import random
from gym import spaces
import gym
import numpy as np
import pdb

U = 0
D = 1
L = 2
R = 3

ACTION_MAP = {
    U: (-1, 0),
    D: (1, 0),
    L: (0, -1),
    R: (0, 1)
}

STR_ACTION_MAP = {
    U: 'U',
    D: 'D',
    L: 'L',
    R: 'R',
}

def solveable(env):
    '''
    env: TileEnv
    A puzzle configuration is solveable if the sum of the permutation parity and the L1 distance of the
    empty tile to the corner location is even.

    If the grid width is odd, then the number of inversions in a solvable situation is even.
    If the grid width is even, and the blank is on an even row counting from the bottom (second-last, fourth-last etc), then the number of inversions in a solvable situation is odd.
    If the grid width is even, and the blank is on an odd row counting from the bottom (last, third-last, fifth-last etc) then the number of inversions in a solvable situation is even.

    Source:
    https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    '''
    perm = [i for i in env.perm_state() if i != (env.n * env.n)] # exclude empty tile
    if env.n % 2 == 1:
        return even_perm(perm)
    else:
        nth_from_bot = env.n - env._empty_x
        return ((n_inversions(perm) % 2 == 1) and (nth_from_bot % 2 == 0)) or \
               ((n_inversions(perm) % 2 == 0) and (nth_from_bot % 2 == 1))

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

def random_perm(n):
    '''
    n: int, size of permutation
    Returns: A random permutation of A_n (an even permutation)
    '''
    x = list(range(1, n + 1))
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

    def step(self, action, ignore_oob=True, one_hot=True):
        '''
        Actions: U/D/L/R
        ignore_oob: bool. If true, invalid moves on the boundary of the cube don't do anything.
        one_hot: bool.
            If true, this returns the one hot vector representation of the puzzle state
            If false, returns the grid representation.
        Move swap the empty tile with the tile in the given location
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        try:
            assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        except:
            pdb.set_trace()

        dx, dy = ACTION_MAP[action]
        new_x = self._empty_x + dx
        new_y = self._empty_y + dy

        # what should we do with illegal moves?
        oob = not self._inbounds(new_x, new_y)
        if oob:
            if ignore_oob:
                print('Taking step {} moves you oob! Not moving anything'.format(STR_ACTION_MAP[action]))
                done = self.is_solved()
                reward = 1 if done else 0
                return self.grid, reward, done, {}
            else:
                raise Exception('Taking action {} will take you out of bounds'.format(action))

        # TODO: Make one hot state the default and construct grid only for rendering
        self.grid[new_x, new_y], self.grid[self._empty_x, self._empty_y] = self.grid[self._empty_x, self._empty_y], self.grid[new_x, new_y]
        self._empty_x = new_x
        self._empty_y = new_y
        done = self.is_solved()
        reward = 1 if done else 0
        if one_hot:
            state = self.one_hot_state()
        else:
            state = self.grid

        assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        return state, reward, done, {}

    def penalty_reward(self, done=None):
        return 1 if self.is_solved() else -1

    def sparse_reward(self):
        return 1 if self.is_solved() else 0

    def _pretty_print(self, x):
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
                print(self._pretty_print(x), end='')
            print()

    def reset(self):
        '''
        Scramble the tile puzzle by taking some number of random moves
        This is actually really quite bad at scrambling
        '''
        self._initted = True
        self._assign_perm(random_perm(self.n * self.n))

        while not solveable(self):
            self._assign_perm(random_perm(self.n * self.n))

        assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        return self.grid

    def _assign_perm(self, perm):
        self.grid = np.array(perm, dtype=int).reshape(self.n, self.n)
        empty_loc = np.where(self.grid == (self.n * self.n))
        self._empty_x, self._empty_y = empty_loc[0][0], empty_loc[1][0]


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

    def one_hot_state(self):
        vec = np.zeros(self.grid.size * self.grid.size)
        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                num = self.grid[i, j] - 1 # grid is 1-indexed
                vec[idx + num] = 1
                idx += self.grid.size

        return vec

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
    random.seed(1)
    n = 3 if len(sys.argv) < 2 else int(sys.argv[1])
    env = TileEnv(n)
    env.reset()
    env.render()
    print('------------------')
    env.step(U)
    env.render()
    print('------------------')
    env.step(U)
    env.render()
    print('------------------')
    env.step(U)
    env.render()
