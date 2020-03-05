import imageio
import numpy as np
import math as m
import matplotlib.pyplot as plt
from PIL import Image

from painter.agents.Agent import Agent
from painter.environmentInterfaces.EnvironmentInterface import EnvironmentInterface


########################################################################################################################
class Animator:
    def __init__(self, agent, environment_interface):
        assert isinstance(agent, Agent), "Agent is not of type agent"
        assert isinstance(environment_interface, EnvironmentInterface), "Environment is not of type LibMyPaint"

        self.agent = agent
        self.envInterface = environment_interface

    def anime(self, target, fps):
        # Initialize the agent
        # TODO: initialize the agent, get the first observation, take the first action

        self.envInterface.reset(target)

        for t in range(19):
            time_step.observation["noise_sample"] = noise_sample
            action, state = step(time_step.step_type, time_step.observation, state)
            time_step = self.environment.step(action)
            actions.append(action)


