import numpy as np

# Importing libmypaint environment
from painter.environments.libmypaint import LibMyPaint


# Utility fonction
def map_to_int_interval(to_map, start, end):
    i = start + to_map * (end - start)
    return int(i)

def distance_l2(matrix1, matrix2):
    return np.linalg.norm(matrix1-matrix2)


# Action space
class ActionSpace(object):

    def __init__(self, n):
        self.n = n


# Interface between the agent and the environment
class LibMyPaintEnvironment(object):

    def __init__(self, episode_length):
        """
        Initializes a reinforcement learning environment

        Takes as inputs
        - number of action in an episode
        Returns
        - nothing
        """

        self.grid_size = 32  # size of the action grid
        self.episode_length = 2 * episode_length  # nombre d'action à prédire pour chaque episode

        env_settings = dict(
            episode_length=self.episode_length,                 # Number of frames in each episode.
            canvas_width=64,                                    # The width of the canvas in pixels.
            grid_width=self.grid_size,                          # The width of the action grid.
            brushes_basedir="third_party/libmypaint_brushes/",  # The location of libmypaint brushes.
            brush_type="classic/dry_brush",                     # The type of the brush.
            brush_sizes=[1, 2, 4, 8, 16],                       # The sizes of the brush to use.
            use_color=True,                                     # Color or black & white output?
            use_pressure=True,                                  # Use pressure parameter of the brush?
            use_alpha=False,                                    # Drop or keep the alpha channel of the canvas?
            background="white"                                  # Background could either be "white" or "transparent".
        )

        self.env = LibMyPaint(**env_settings)
        self.state = self.env.reset()
        self.actions = []  # TODO

        self.action_space = ActionSpace(self.episode_length)

    def reset(self, objective):
        """
        Reinitializes the reinforcement learning environment
        
        Takes as inputs
        - objective : 3d numpy array of shape (height, width, channels)
        - n_actions : integer indicating how many components there are in one agent action
        Returns
        - observable : 3d numpy array of shape (height, width, channels) representing the new state of the environment
        """
        # objectif -> l'image target ndarray hxlx3
        self.state = self.env.reset()
        self.actions = []
        self.objective = objective
        return self.getObservable()

    def getObservable(self):
        """
        Returns the observable data of the environment
        
        Takes as inputs
        - nothing
        Returns
        - observable : 3d numpy array of shape (height, width, channels) representing the new state of the environment
        """

        return self.state.observation["canvas"]

    def step(self, action):
        """
        Updates the environment with the given action
        
        Takes as inputs
        - action : dictionnary representing an action
        Returns
        - observable : 3d numpy array of shape (height, width, channels) representing the new state of the environment
        - reward : reward given to the agent for the performed action
        - done : boolean indicating if new state is a terminal state
        - infos : dictionary of informations (for debugging purpose)
        """

        if LibMyPaint.step_type.LAST == self.state["step_type"]:
            return self.getObservable(), distance_l2(self.getObservable(), self.objective), True, {}

        self.actions.append(action)

        action_spec = self.env.action_spec()

        # extract the values
        # TODO
        x_start, y_start, x_control, y_control, x_end, y_end, brush_pressure, brush_size, r, g, b = action

        # map them to the right interval
        x_start = map_to_int_interval(x_start, 0, self.grid_size - 1)
        y_start = map_to_int_interval(y_start, 0, self.grid_size - 1)
        x_control = map_to_int_interval(x_control, 0, self.grid_size - 1)
        y_control = map_to_int_interval(y_control, 0, self.grid_size - 1)
        x_end = map_to_int_interval(x_end, 0, self.grid_size - 1)
        y_end = map_to_int_interval(y_end, 0, self.grid_size - 1)
        brush_pressure = map_to_int_interval(brush_pressure,
                                             action_spec["pressure"].minimum, action_spec["pressure"].maximum)
        brush_size = map_to_int_interval(brush_size,
                                         action_spec["size"].minimum, action_spec["size"].maximum)
        r = map_to_int_interval(r,
                                action_spec["red"].minimum, action_spec["red"].maximum)
        g = map_to_int_interval(g,
                                action_spec["green"].minimum, action_spec["green"].maximum)
        b = map_to_int_interval(b,
                                action_spec["blue"].minimum, action_spec["blue"].maximum)

        # go to the strat of the curve
        move = {"control": np.ravel_multi_index((x_start, y_start),
                                                (self.grid_size - 1, self.grid_size - 1)).astype("int32"),
                "end": np.ravel_multi_index((x_start, y_start),
                                            (self.grid_size - 1, self.grid_size - 1)).astype("int32"),
                "flag": np.int32(0),
                "pressure": np.int32(brush_pressure),
                "size": np.int32(brush_size),
                "red": np.int32(r),
                "green": np.int32(g),
                "blue": np.int32(b)}
        self.state = self.env.step(move)

        # draw the curve
        draw = {"control": np.ravel_multi_index((x_control, y_control),
                                                (self.grid_size - 1, self.grid_size - 1)).astype("int32"),
                "end": np.ravel_multi_index((x_end, y_end),
                                            (self.grid_size - 1, self.grid_size - 1)).astype("int32"),
                "flag": np.int32(1),
                "pressure": np.int32(brush_pressure),
                "size": np.int32(brush_size),
                "red": np.int32(r),
                "green": np.int32(g),
                "blue": np.int32(b)}
        self.state = self.env.step(draw)

        if LibMyPaint.step_type.LAST == self.state["step_type"]:
            return self.getObservable(), distance_l2(self.getObservable(), self.objective), True, {}

        return self.getObservable(), 0, False, {}

    def close(self):
        return

    def render(self, mode):
        """
        Returns a graphic representation of the environment
        
        Takes as inputs
        - mode : among ("rgb_array",)
        Returns
        - a render of the environment state, given in the requested mode
        """
        if mode == "hight_res":
            ### compléter
            return self.getObservable()
        else:
            return self.getObservable()


class SimpleSequentialReproductionInstantEnvironment(object):

    def __init__(self):
        """
        Initializes a reinforcement learning environment
        
        Takes as inputs
        - nothing
        Returns
        - nothing
        """
        n_actions = 1
        self.action_space = ActionSpace(n_actions)
        self.goal = None
        self.current = None
        self.actionsLeft = None
        return

    def reset(self, objective):
        """
        Reinitializes the reinforcement learning environment
        
        Takes as inputs
        - objective : 1d numpy array
        Returns
        - observable : 1d numpy array representing the new state of the environment
        """
        self.goal = objective
        self.current = np.zeros(self.goal.shape)
        self.actionsLeft = 3

        return self.getObservable()

    def getObservable(self):
        """
        Returns the observable data of the environment
        
        Takes as inputs
        - nothing
        Returns
        - observable : 1d numpy array representing the new state of the environment
        """
        observable = self.current
        return observable

    def step(self, action):
        """
        Updates the environment with the given action
        
        Takes as inputs
        - action : dictionnary representing an action
        Returns
        - observable : 1d numpy array representing the new state of the environment
        - reward : reward given to the agent for the performed action
        - done : boolean indicating if new state is a terminal state
        - infos : dictionary of informations (for debugging purpose)
        """
        pixel = map_to_int_interval(action["pixel"], 0, self.goal.shape[0] - 1)
        self.current[pixel] = 1
        self.actionsLeft -= 1

        observable = self.getObservable()
        reward = np.sum(self.goal - self.current)
        done = (self.actionsLeft == 0)
        infos = {"pixel": pixel}
        return observable, reward, done, infos

    def close(self):
        return

    def render(self, mode):
        """
        Returns a graphic representation of the environment
        
        Takes as inputs
        - mode : among ("rgb_array",)
        Returns
        - a render of the environment state, given in the requested mode
        """
        return self.getObservable()
