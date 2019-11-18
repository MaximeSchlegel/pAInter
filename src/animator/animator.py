import imageio
import numpy as np
import math as m
import matplotlib.pyplot as plt
from PIL import image

from src.agents.impala_agent import Agent
from src.environments.libmypaint import LibMyPaint


class Animator:
    def __init__(self, agent, environment):
        assert isinstance(agent, Agent), "Agent is not of type agent"
        assert isinstance(environment, LibMyPaint), "Environment is not of type LibMyPaint"

        self.agent = agent
        self.environment = environment

    def anime(self, target, lenght, fps):
        # Initialize the agent
        # TODO: initialize the agent, get the first observation, take the first action

        initial_state, step = agent_utils.get_module_wrappers(MODULE_PATH)
        state = initial_state()

        time_step = self.environment.reset()

        noise_sample = np.random.normal(size=(10,)).astype(np.float32)

        actions = []
        for t in range(19):
            time_step.observation["noise_sample"] = noise_sample
            action, state = step(time_step.step_type, time_step.observation, state)
            time_step = self.environment.step(action)
            actions.append(action)


