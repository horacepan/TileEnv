TileEnv
===
This is a simple implementation of a Sliding Tile Puzzle gym-like environment.

### Requirements:
* python3
* numpy
* gym

### Installation:
Run the following in the repository's base directory
```
pip install -e .
```

### Usage
```
from tile_env import TileEnv, U, D, L, R
env = TileEnv(4)
env.reset()
env.render()
env.step(U)
env.render()
env.step(L)
```

