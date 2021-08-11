import numpy as np
from momentum import Momentum


class Neuron:
    def __init__(self, inputNum: int):
        weights = []
        for i in range(inputNum):
            weights.append(Momentum(np.random.normal(), 0.5))
        self._weights = np.array(weights)
        self._bias = Momentum(np.random.normal(), 0.5)

    @staticmethod
    def _sigmoid(x) -> np.float64:
        return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

    @classmethod
    def _activation(cls, x) -> np.float64:
        return cls._sigmoid(x)

    @classmethod
    def _activation_deriv(cls, x) -> np.float64:
        fx = cls._sigmoid(x)
        return fx * (1 - fx)

    def _basis(self, inputs: np.ndarray) -> np.float64:
        return np.dot(self._weights, inputs) + self._bias

    def feedforward(self, inputs: np.ndarray):
        total = self._basis(inputs)
        return self._activation(total)

    def backpropagation(self, inputs: np.ndarray, learnRate: float, lastDerivBack: np.float64 = 1) -> np.ndarray:
        d = self._activation_deriv(self._basis(inputs)) * lastDerivBack
        back = d * self._weights
        for idx in range(self._weights.size):
            self._weights[idx] << learnRate * d * inputs[idx]
        self._bias << learnRate * d
        return back
