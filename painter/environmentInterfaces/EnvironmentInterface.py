class EnvironmentInterface:
    def __int__(self):
        pass

    def reset(self, objective):
        pass

    def getObservable(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass

    def render(self, mode):
        pass
