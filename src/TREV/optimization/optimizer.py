from typing import Callable, Dict


class Optimizer:
    def __init__(self, optimizer:Callable, args:Dict):
        self.optimizer = optimizer
        self.args = args

    def get_optimizer(self, theta):
        return self.optimizer(theta, **self.args)
    