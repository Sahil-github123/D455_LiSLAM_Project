from .observer_state import ObserverState
from .propagator import Propagator

class Observer:

    def __init__(self):

        self.state = ObserverState()
        self.propagator = Propagator()

    def process(self, frame):
        self.propagator.propagate(self.state, frame)
        
        # print("State after propagation:")
        # print("R:", self.state.R)
        # print("v:", self.state.v)
        # print("x:", self.state.x)
        # print("Prediction")
        # print("Correction")
        # print("State updated")
        
        return self.state



# Testing math_utils functions
# from .math_utils import skew, so3_exp
# import numpy as np
# print(skew(np.array([1.,2.,3.])))
# print(so3_exp(np.zeros(3)))