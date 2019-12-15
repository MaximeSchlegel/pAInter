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

action_spec = env.action_spec()
for i, j in action_spec.items():
    print(i + "  : ", j)
print("\n")

time_step0 = env.reset()
plt.imshow(time_step0.observation["canvas"])
plt.show()

action = {"control": np.ravel_multi_index((0, 31), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((31, 31), (32, 32)).astype("int32"),
          "flag": np.int32(1),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(0),
          "green": np.int32(0),
          "blue": np.int32(0)}

# for i, j in action.items():
#     print(i + "  : ", j)
# print("\n")

time_step1 = env.step(action)
plt.imshow(time_step1.observation["canvas"])
plt.show()

action = {"control": np.ravel_multi_index((14, 31), (32, 32)).astype("int32"),
          "end": np.ravel_multi_index((14, 14), (32, 32)).astype("int32"),
          "flag": np.int32(1),
          "pressure": np.int32(1),
          "size": np.int32(1),
          "red": np.int32(1),
          "green": np.int32(1),
          "blue": np.int32(1),
         }

# for i, j in action.items():
#     print(i + "  : ", j)
# print("\n")

time_step2 = env.step(action)
print(time_step2)
plt.imshow(time_step2.observation["canvas"])
plt.show()

obs = env.observation()
print(obs)

# print(np.where(time_step0.observation["canvas"] - time_step1.observation["canvas"] != 0), "\n")
# print(np.where(time_step1.observation["canvas"] - time_step2.observation["canvas"] != 0), "\n")

# env = EnvInterface(env_settings)
#
# print(env.get_canvas())
# env.draw_to([0, 0,  # start pos
#              0, 1,  # control pos
#              1, 1,  # end pos
#              1, 1,  # brush option
#              1, 1, 1])  # rgb color
#
# # print(env.get_canvas())
# plt.imshow(env.get_canvas())
# plt.show()

print(env.reward_spec())