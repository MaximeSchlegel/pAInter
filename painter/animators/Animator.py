import imageio
import pathlib

from pygifsicle import optimize


########################################################################################################################
class Animator:
    def __init__(self, agent, environment_interface, target, out_path):
        self.agent = agent
        self.envInterface = environment_interface
        self.target_img = target
        self.out_path = out_path

    def anime(self, target, fps):
        # Initialize the environment
        self.envInterface.reset(target)
        obs, score, ended, info = self.envInterface.getObservable()
        img = self.envInterface.render()

        writer = imageio.get_writer(pathlib.Path.joinpath(self.out_path, str(hash(self.agent.name())) + ".gif"),
                                    mode='I',
                                    fps=fps)

        while not ended:
            action = self.agent.step(obs)
            obs, score, ended, info = self.envInterface.step(action)
            img = self.envInterface.render()
            writer.append_data(img)
        optimize(pathlib.Path.joinpath(self.out_path, str(hash(self.agent.name()))))
