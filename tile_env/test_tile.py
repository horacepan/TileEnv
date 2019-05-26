import unittest
import numpy as np
from tile import even_perm, random_perm, solveable, TileEnv, neighbors
import pdb

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

    def test_env(self):
        perm = [8, 2, 3, 4, 6, 5, 7, 9, 1]
        env = TileEnv.from_perm(perm)
        exp_grid = np.array([[8, 2, 3], [4, 6, 5], [7, 9, 1]])
        self.assertTrue(np.all(np.equal(exp_grid, env.grid)))

    def test_reset(self):
        env = TileEnv(4)
        for _ in range(100):
            env.reset()
            self.assertTrue(solveable(env))

    def test_one_hot(self):
        env = TileEnv.from_perm(list(range(1, 16 + 1)))
        self.assertTrue(np.allclose(np.eye(16).ravel(), env.one_hot_state()))

    def test_moves(self):
        env = TileEnv.from_perm([1, 2, 3, 4, 5, 6, 7, 8, 9])
        env.step(TileEnv.U)
        self.assertTrue(np.allclose(env.grid, np.array([[1, 2, 3], [4, 5, 9], [7, 8, 6]])))
        env.step(TileEnv.L)
        self.assertTrue(np.allclose(env.grid, np.array([[1, 2, 3], [4, 9, 5], [7, 8, 6]])))
        env.step(TileEnv.L)
        self.assertTrue(np.allclose(env.grid, np.array([[1, 2, 3], [9, 4, 5], [7, 8, 6]])))
        env.step(TileEnv.D)
        self.assertTrue(np.allclose(env.grid, np.array([[1, 2, 3], [7, 4, 5], [9, 8, 6]])))
        env.step(TileEnv.R)
        self.assertTrue(np.allclose(env.grid, np.array([[1, 2, 3], [7, 4, 5], [8, 9, 6]])))

    def test_solveable(self):
        env = TileEnv.from_perm([1, 4, 3, 2])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([1, 8, 2, 9, 4, 3, 7, 6, 5])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 16])
        self.assertFalse(solveable(env))

        env = TileEnv.from_perm([13, 2, 10, 3, 1, 12, 8, 4, 5, 16, 9, 6, 15, 14, 11, 7])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([13, 2, 10, 3, 1, 12, 8, 4, 5, 16, 9, 6, 15, 14, 11, 7])
        self.assertTrue(solveable(env))

        env = TileEnv.from_perm([3, 9, 1, 15, 14, 11, 4, 6, 13, 16, 10, 12, 2, 7, 8, 5])
        self.assertFalse(solveable(env))

    def test_is_solved(self):
        env = TileEnv.from_perm(list(i for i in range(1, 17)))
        self.assertTrue(env.is_solved())

        env = TileEnv.from_perm([1, 3, 2, 4])
        self.assertFalse(env.is_solved())

        env = TileEnv.from_perm([1, 3, 2, 4, 5, 7, 6, 9, 8])
        self.assertFalse(env.is_solved())

        env = TileEnv.from_perm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15])
        self.assertFalse(env.is_solved())
        env.step(TileEnv.R)
        self.assertTrue(env.is_solved())

    def test_neighbors(self):
        env = TileEnv.from_perm((1, 2, 3, 4), one_hot=False)
        nbrs = env.neighbors()
        self.assertTrue(TileEnv.U in nbrs)
        self.assertTrue(TileEnv.D not in nbrs)
        self.assertTrue(TileEnv.L in nbrs)
        self.assertTrue(TileEnv.R not in nbrs)

        ugrid = np.array([[1, 4], [3, 2]])
        lgrid = np.array([[1, 2], [4, 3]])
        self.assertTrue(np.allclose(nbrs[TileEnv.U], ugrid))
        self.assertTrue(np.allclose(nbrs[TileEnv.L], lgrid))

    def test_neighbors2(self):
        env = TileEnv(3)
        for _ in range(10):
            env.shuffle(100)
            static_nbrs = neighbors(env.grid, env.x, env.y)
            nbrs = env.neighbors()
            for a, grid in nbrs.items():
                try:
                    self.assertTrue(np.allclose(static_nbrs[a], grid))
                except:
                    pdb.set_trace()

if __name__ == '__main__':
    unittest.main()
