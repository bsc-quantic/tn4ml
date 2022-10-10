import abc


class Optimizer:
    def __init__(self, hyperparameters: dict):
        super().__setattr__("hyperparameters", hyperparameters)

    @abc.abstractmethod
    def __call__(self, tensor, grad):
        pass

    def __getattr__(self, name: str):
        try:
            return self.hyperparameters[name]
        except:
            raise AttributeError(f"{name} hyperparameter not found")

    def __setattr__(self, name: str, value):
        try:
            self.hyperparameters[name] = value
        except KeyError:
            raise AttributeError(f"{name} hyperparameter not found")


class DirectGradientDescent(Optimizer):
    def __init__(self, learning_rate=1e-3):
        super().__init__({"learning_rate": learning_rate})

    def __call__(self, tensor, grad):
        return tensor - self.learning_rate * grad
