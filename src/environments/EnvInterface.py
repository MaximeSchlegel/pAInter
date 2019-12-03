import numpy as np

from src.environments.libmypaint import LibMyPaint


class EnvInterface:

    def __init__(self, env_setting):
        self.grid_size = env_setting["grid_width"]
        self.env = LibMyPaint(**env_setting)
        self.state = self.env.reset()
        self.brush_position = 0, 0
        self.actions = []

    @staticmethod
    def _map_to_int_interval(f, start, end):
        i = start + f * (end - start)
        return int(i)

    def draw_to(self, action_input):
        # check that the input is normalized
        assert len(action_input) == 11, "Expected input of size 8"
        for value in action_input:
            assert 0 <= value <= 1, "Expect all values in [0;1]"

        # saves the action
        self.actions.append(action_input)

        action_spec = self.env.action_spec()

        # extract the values
        x_start, y_start, x_control, y_control, x_end, y_end, brush_pressure, brush_size, r, g, b = action_input

        # map them to the right interval
        x_start = self._map_to_int_interval(x_start, 0, self.grid_size - 1)
        y_start = self._map_to_int_interval(y_start, 0, self.grid_size - 1)
        x_control = self._map_to_int_interval(x_control, 0, self.grid_size - 1)
        y_control = self._map_to_int_interval(y_control, 0, self.grid_size - 1)
        x_end = self._map_to_int_interval(x_end, 0, self.grid_size - 1)
        y_end = self._map_to_int_interval(y_end, 0, self.grid_size - 1)
        brush_pressure = self._map_to_int_interval(brush_pressure,
                                                   action_spec["pressure"].minimum, action_spec["pressure"].maximum)
        brush_size = self._map_to_int_interval(brush_size,
                                               action_spec["size"].minimum, action_spec["size"].maximum)
        r = self._map_to_int_interval(r,
                                      action_spec["red"].minimum, action_spec["red"].maximum)
        g = self._map_to_int_interval(g,
                                      action_spec["green"].minimum, action_spec["green"].maximum)
        b = self._map_to_int_interval(b,
                                      action_spec["blue"].minimum, action_spec["blue"].maximum)

        print(x_start, y_start, "\n")
        print(x_control, y_control, "\n")
        print(x_end, y_end, "\n")

        if (x_start, y_start) != self.brush_position:
            # go to the strat of the curve
            move = {"control":  np.ravel_multi_index((x_start, y_start),
                                                     (self.grid_size -1, self.grid_size -1)).astype("int32"),
                    "end":      np.ravel_multi_index((x_start, y_start),
                                                     (self.grid_size -1, self.grid_size -1)).astype("int32"),
                    "flag":     np.int32(0),
                    "pressure": np.int32(brush_pressure),
                    "size":     np.int32(brush_size),
                    "red":      np.int32(r),
                    "green":    np.int32(g),
                    "blue":     np.int32(b)}
            self.state = self.env.step(move)
            self.brush_position = x_start, y_start

        # draw the curve
        draw = {"control":  np.ravel_multi_index((x_control, y_control),
                                                 (self.grid_size -1, self.grid_size -1)).astype("int32"),
                "end":      np.ravel_multi_index((x_end, y_end),
                                                 (self.grid_size -1, self.grid_size -1)).astype("int32"),
                "flag":     np.int32(1),
                "pressure": np.int32(brush_pressure),
                "size":     np.int32(brush_size),
                "red":      np.int32(r),
                "green":    np.int32(g),
                "blue":     np.int32(b)}
        self.state = self.env.step(draw)
        self.brush_position = x_end, y_end

    def get_canvas(self):
        return self.state.observation["canvas"]
