import imageio
from pygifsicle import optimize

from painter.agents.Agent import Agent
from painter.environments.LibMyPaintInterface import LibMyPaintInterface

########################################################################################################################
class Animator:
    def __init__(self, agent, environment_interface, objectif):
        assert isinstance(agent, Agent), "Agent is not of type agent"
        assert isinstance(environment_interface, LibMyPaintInterface), "Environment is not of type LibMyPaint"

        self.agent = agent
        self.envInterface = environment_interface
        self.objectif = objectif

    def anime(self, target, fps):
        # Initialize the agent
        # TODO: initialize the agent, get the first observation, take the first action

        self.envInterface.reset(target)
        obs, score, ended, info = self.envInterface.getObservable()
        img = self.envInterface.render("rgb")

        writer = imageio.get_writer('out/animation.gif', mode='I', fps=fps)

        while not ended:
            action = self.agent.step(obs, self.objectif)
            obs, score, ended, info = self.envInterface.step(action)
            img = self.envInterface.render("rgb")
            writer.append_data(img)

        optimize("out/animation.gif")
