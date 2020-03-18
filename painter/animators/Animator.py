import imageio
import pathlib
import random

from pygifsicle import optimize


########################################################################################################################
class Animator:
    def __init__(self, agent, environment_interface, out_path):
        self.agent = agent
        self.envInterface = environment_interface
        self.out_path = out_path

    def anime(self, target, fps):
        # Initialize the environment
        obs, ended = self.envInterface.reset(target), False
        path = pathlib.Path.joinpath(self.out_path,
                                     str(hash(self.agent.agent_name + random.randint(0, 1000))) + ".gif").as_posix()
        writer = imageio.get_writer(path,
                                    mode='I',
                                    fps=fps)

        print("output path: " + path)

        # Add the white canvas
        img = self.envInterface.render()
        writer.append_data(img)

        # Play until the sequence has ended
        while not ended:
            action = self.agent.decision(obs)
            obs, score, ended, info = self.envInterface.step(action)
            img = self.envInterface.render()
            writer.append_data(img)
        optimize(path)
