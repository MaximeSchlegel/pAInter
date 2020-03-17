import colorsys as cs
import dm_env as environment
import math as m
import matplotlib.pyplot as plt
import numpy as np

from hilbertcurve.hilbertcurve import HilbertCurve
from painter.environments.libmypaint import LibMyPaint
from painter.environments.libmypaint_hsv import LibMyPaint_hsv
from PIL import Image


########################################################################################################################
# Action space
class ActionSpace(object):

    def __init__(self, observation_spec):
        self.action_mask = observation_spec["action_mask"]
        self.episode_length = observation_spec["episode_length"]


########################################################################################################################
# Oberservation space
class ObservationSpace:

    def __init__(self, observation_spec):
        self.shape = tuple([2] + list(observation_spec["canvas"].shape))


########################################################################################################################
# Interface between the agent and the environment
class LibMyPaintInterface:

    def __init__(self, episode_length, color_type="rgb", coord_type="cart", grid_size=32, canvas_size=64):
        self.grid_size = grid_size  # size of the action grid
        self.episode_length = 2 * episode_length  # nombre d'action à prédire pour chaque episode

        env_settings = dict(
            episode_length=self.episode_length,  # Number of frames in each episode.
            canvas_width=canvas_size,  # The width of the canvas in pixels.
            grid_width=self.grid_size,  # The width of the action grid.
            brushes_basedir="third_party/libmypaint_brushes/",  # The location of libmypaint brushes.
            brush_type="classic/dry_brush",  # The type of the brush.
            brush_sizes=[1, 2, 4, 8, 16],  # The sizes of the brush to use.
            use_color=color_type != "grey",  # Color or black & white output?
            use_pressure=True,  # Use pressure parameter of the brush?
            use_alpha=False,  # Drop or keep the alpha channel of the canvas?
            background="white"  # Background could either be "white" or "transparent".
        )

        self.color_type = color_type
        if (color_type == "rgb") or (color_type == "grey"):
            self.env = LibMyPaint(**env_settings)
            self.color_name = ["red", "green", "blue"]
        elif color_type == "hsv":
            self.env = LibMyPaint_hsv(**env_settings)
            self.color_name = ["hue", "saturation", "value"]
        else:
            raise ValueError("color_type must be 'grey', 'rgb' or 'hsv'")

        self.coord_type = coord_type
        if (coord_type != "cart" and
                coord_type != "hilb"):
            raise ValueError("coord_type must be 'cart' or 'hilb'")
        elif coord_type == "hilb":
            l = m.log(self.grid_size, 2)
            assert l - int(l) == 0, "the action grid size can't be converted to an hilbert curve"
            self.hilbert_curve = HilbertCurve(p=int(l), n=2)

        self.action_space = ActionSpace(self.env.observation_spec())
        self.observation_space = ObservationSpace(self.env.observation_spec())

        self.state = self.env.reset()
        self.target = None

        self.actions = []  # TODO

    @staticmethod
    def _map_to_int_interval(to_map, start, end):
        i = start + to_map * (end - start)
        return int(round(i))

    @staticmethod
    def _distance_l2(matrix1, matrix2):
        return np.linalg.norm(matrix1 - matrix2)

    def reset(self, target):
        """
        Reinitializes the reinforcement learning environment
        
        Takes as inputs
        - target : 3d numpy array of shape (height, width, channels)
        Returns
        - observable : 3d numpy array of shape (height, width, channels) representing the new state of the environment
        """
        self.state = self.env.reset()
        self.actions = []
        self.target = target

        return self.getObservable()

    def getObservable(self):
        """
        Returns the observable data of the environment

        Returns
        - observable :  the current target and
                        3d numpy array of shape (height, width, channels) representing the new state of the environment
        """
        assert self.target is not None, "The target not define, to do so reset the environment"

        return np.array([self.target,
                self.state.observation["canvas"]])

    def step(self, action):
        """
        Updates the environment with the given action
        
        Takes as inputs
        - action : array of value € [0,1] representing an action
        Returns
        - observable : 3d numpy array of shape (height, width, channels) representing the new state of the environment
        - reward : reward given to the agent for the performed action
        - done : boolean indicating if new state is a terminal state
        - infos : dictionary of informations (for debugging purpose)
        """

        assert self.target is not None, "The target not define, to do so reset the environment"

        if self.state.step_type == environment.StepType.LAST:
            return (self.getObservable(),
                    LibMyPaintInterface._distance_l2(self.state.observation["canvas"], self.target),
                    True,
                    {"info": "To continue reset the environment"})

        self.actions.append(action)

        action_spec = self.env.action_spec()

        # extract the values
        if self.coord_type == "cart":
            x_start, y_start, x_control, y_control, x_end, y_end, brush_pressure, brush_size, color_1, color_2, color_3 = action
            # map the coordinates to the right interval
            x_start = LibMyPaintInterface._map_to_int_interval(x_start, 0, self.grid_size - 1)
            y_start = LibMyPaintInterface._map_to_int_interval(y_start, 0, self.grid_size - 1)
            x_control = LibMyPaintInterface._map_to_int_interval(x_control, 0, self.grid_size - 1)
            y_control = LibMyPaintInterface._map_to_int_interval(y_control, 0, self.grid_size - 1)
            x_end = LibMyPaintInterface._map_to_int_interval(x_end, 0, self.grid_size - 1)
            y_end = LibMyPaintInterface._map_to_int_interval(y_end, 0, self.grid_size - 1)

        elif self.coord_type == "hilb":
            start, control, end, brush_pressure, brush_size, color_1, color_2, color_3 = action
            # map the coordonates to the right interval
            start = int(LibMyPaintInterface._map_to_int_interval(start, 0, pow(2, m.log(self.grid_size, 2) * 2) - 1))
            control = int(LibMyPaintInterface._map_to_int_interval(control, 0, pow(2, m.log(self.grid_size, 2) * 2) - 1))
            end = int(LibMyPaintInterface._map_to_int_interval(end, 0, pow(2, m.log(self.grid_size, 2) * 2) - 1))
            x_start, y_start = self.hilbert_curve.coordinates_from_distance(start)
            x_control, y_control = self.hilbert_curve.coordinates_from_distance(control)
            x_end, y_end = self.hilbert_curve.coordinates_from_distance(end)

        brush_pressure = LibMyPaintInterface._map_to_int_interval(brush_pressure,
                                                                  action_spec["pressure"].minimum,
                                                                  action_spec["pressure"].maximum)

        brush_size = LibMyPaintInterface._map_to_int_interval(brush_size,
                                                              action_spec["size"].minimum,
                                                              action_spec["size"].maximum)

        color_1 = LibMyPaintInterface._map_to_int_interval(color_1,
                                                           action_spec[self.color_name[0]].minimum,
                                                           action_spec[self.color_name[1]].maximum)

        color_2 = LibMyPaintInterface._map_to_int_interval(color_2,
                                                           action_spec[self.color_name[1]].minimum,
                                                           action_spec[self.color_name[1]].maximum)

        color_3 = LibMyPaintInterface._map_to_int_interval(color_3,
                                                           action_spec[self.color_name[2]].minimum,
                                                           action_spec[self.color_name[2]].maximum)

        # go to the start of the curve
        move = {"control": np.ravel_multi_index((x_start, y_start),
                                                (self.grid_size, self.grid_size)).astype("int32"),
                "end": np.ravel_multi_index((x_start, y_start),
                                            (self.grid_size, self.grid_size)).astype("int32"),
                "flag": np.int32(0),
                "pressure": np.int32(brush_pressure),
                "size": np.int32(brush_size),
                self.color_name[0]: np.int32(color_1),
                self.color_name[1]: np.int32(color_2),
                self.color_name[2]: np.int32(color_3)}
        self.state = self.env.step(move)

        if self.state.step_type == environment.StepType.LAST:
            return (self.getObservable(),
                    LibMyPaintInterface._distance_l2(self.state.observation["canvas"], self.target),
                    True,
                    {})

        # draw the curve
        draw = {"control": np.ravel_multi_index((x_control, y_control),
                                                (self.grid_size, self.grid_size)).astype("int32"),
                "end": np.ravel_multi_index((x_end, y_end),
                                            (self.grid_size, self.grid_size)).astype("int32"),
                "flag": np.int32(1),
                "pressure": np.int32(brush_pressure),
                "size": np.int32(brush_size),
                self.color_name[0]: np.int32(color_1),
                self.color_name[1]: np.int32(color_2),
                self.color_name[2]: np.int32(color_3)}
        self.state = self.env.step(draw)

        if self.state.step_type == environment.StepType.LAST:
            return (self.getObservable(),
                    LibMyPaintInterface._distance_l2(self.state.observation["canvas"], self.target),
                    True,
                    {})

        return (self.getObservable(),
                0,
                False,
                {})

    def close(self):
        self.env.close()

    def render(self, mode=None):
        """
        Returns a graphic representation of the environment
        
        Takes as inputs
        - mode : ø
        Returns
        - a render of the environment state, given in the requested mode
        """
        img = self.getObservable()[1]
        if self.color_type == "hsv":
            img = np.array([[cs.hsv_to_rgb(img[i, j][0],
                                    img[i, j][1],
                                    img[i, j][2]) for j in range(64)] for i in range(64)])
        plt.imshow(img)
        plt.show()
        return img
