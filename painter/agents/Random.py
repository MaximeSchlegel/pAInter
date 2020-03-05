import random

from painter.agents.Agent import Agent


class Random(Agent):

    def __int__(self):
        pass

    def step(self, observation, target):
        x_start, y_start = random.random(), random.random()
        x_control, y_control = random.random(), random.random()
        x_end, y_end = random.random(), random.random()
        brush_pressure, brush_size = random.random(), random.random()
        r, g, b = random.random(), random.random(), random.random()
        return (x_start, y_start,
                x_control, y_control,
                x_end, y_end,
                brush_pressure, brush_size,
                r, g, b)