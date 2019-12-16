import numpy as np
import matplotlib.pylab as plt

from painter.environments.EnvInterface import EnvInterface
from painter.environments.libmypaint import LibMyPaint

BRUSHES_BASEDIR = "third_party/libmypaint_brushes/"

env_settings = dict(
            episode_length=32,                 # Number of frames in each episode.
            canvas_width=64,                                    # The width of the canvas in pixels.
            grid_width=32,                          # The width of the action grid.
            brushes_basedir="third_party/libmypaint_brushes/",  # The location of libmypaint brushes.
            brush_type="classic/dry_brush",                     # The type of the brush.
            brush_sizes=[1, 2, 4, 8, 16],                       # The sizes of the brush to use.
            use_color=True,                                     # Color or black & white output?
            use_pressure=True,                                  # Use pressure parameter of the brush?
            use_alpha=False,                                    # Drop or keep the alpha channel of the canvas?
            background="white",                                  # Background could either be "white" or "transparent".
            rewards=1
)

env = LibMyPaint(**env_settings)

time_step = env.reset()
plt.imshow(time_step.observation["canvas"])
plt.show()

action = {"control": np.ravel_multi_index((0, 0), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((8, 8), (32, 32)).astype("int32"),
          "flag": np.int32(0),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}
time_step = env.step(action)

action = {"control": np.ravel_multi_index((9, 9), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((10, 9), (32, 32)).astype("int32"),
          "flag": np.int32(1),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}
time_step = env.step(action)
plt.imshow(time_step.observation["canvas"])
plt.show()

action = {"control": np.ravel_multi_index((8, 31-8), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((8, 31-8), (32, 32)).astype("int32"),
          "flag": np.int32(0),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}
time_step = env.step(action)


action = {"control": np.ravel_multi_index((9, 31-9), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((10, 31-9), (32, 32)).astype("int32"),
          "flag": np.int32(1),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}
time_step = env.step(action)
plt.imshow(time_step.observation["canvas"])
plt.show()

action = {"control": np.ravel_multi_index((0, 0), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((20, 5), (32, 32)).astype("int32"),
          "flag": np.int32(0),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}
time_step = env.step(action)

action = {"control": np.ravel_multi_index((25, 15), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((20, 25), (32, 32)).astype("int32"),
          "flag": np.int32(1),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}

time_step = env.step(action)
plt.imshow(time_step.observation["canvas"])
plt.show()