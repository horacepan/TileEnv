import unittest
import numpy as np
from tile import even_perm, random_alternating_perm, solveable, TileEnv

class TestTile(unittest.TestCase):
    def test_even_perm(self):
        perm = (1, 2, 3, 4)
        self.assertTrue(even_perm(perm))

        perm = (1, 2, 3, 4, 5, 6, 7, 8)
        self.assertTrue(even_perm(perm))

        perm = (2, 1, 3, 4, 5, 6, 7, 8)
        self.assertFalse(even_perm(perm))

        perm = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 16)
        self.assertFalse(even_perm(perm))

    def test_random_alt_perm(self):
        n = 12
        for _ in range(100):
            self.assertTrue(even_perm(random_alternating_perm(n)))

    def test_env(self):
        perm = [8, 2, 3, 4, 6, 5, 7, 9, 1]
        env = TileEnv.from_perm(perm)
        exp_grid = np.array([[8, 2, 3], [4, 6, 5], [7, 9, 1]])
        self.assertTrue(np.all(np.equal(exp_grid, env.grid)))

    def test_invariant(self):
        env = TileEnv(4)

        for _ in range(100):
            env.reset()
            self.assertTrue(solveable(env))            

    def test_one_hot(self):
        env = TileEnv(4)
        env._assign_perm(list(range(1, 16 + 1)))
        self.assertTrue(np.allclose(np.eye(16).ravel(), env.one_hot_state()))

if __name__ == '__main__':
    unittest.main()
