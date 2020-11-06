from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, in_):
        pass

    @abstractmethod
    def backward(self, in_, err):
        pass


class Line(Layer):
    def __init__(self, weight, bias):
        self.weight = np.array(weight)
        self.bias = np.array(bias)

        self.dw = np.zeros(self.weight.shape)
        self.db = np.zeros(self.bias.shape)

    def forward(self, in_):
        in_ = np.array(in_)
        dot = in_.dot(self.weight.T)
        add_bias = dot + self.bias
        return add_bias

    def backward(self, in_, err, lk=0.00001):
        err = np.array(err)
        in_ = np.array(in_)

        error_ones = np.ones(err.shape[0])
        self.weight -= (err.T.dot(in_) / err.shape[0]) * lk
        self.weight -= self.weight * 0.01 * lk

        self.bias -= (err.T.dot(error_ones) / err.shape[0]) * lk
        self.bias -= self.bias * 0.01 * lk

        nabla_prev_layer = err.dot(self.weight)
        return nabla_prev_layer


class Relu(Layer):
    LEAK = 0.0001

    def forward(self, in_):
        in_ = np.array(in_)
        return in_ * (in_ > 0) + in_ * (in_ <= 0) * self.LEAK

    def backward(self, in_, err):
        err = np.array(err)
        in_ = np.array(in_)
        return err * (in_ > 0) + err * (in_ <= 0) * self.LEAK
