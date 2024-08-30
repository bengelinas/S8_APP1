import numpy as np

from dnn_framework.layer import Layer

ESPILON = 1e-7


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()
        self.input_count = input_count
        self.output_count = output_count
        self.w = np.random.normal(0, 2 / (output_count + input_count), (output_count, input_count))
        self.b = np.random.normal(0, 1 / output_count, output_count)

    def get_parameters(self):
        params = {'w': self.w,
                  'b': self.b}
        return params

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = x @ self.w.T + self.b
        cache = {'y': y, 'x': x}
        return y, cache

    def backward(self, output_grad, cache):
        grad_x = output_grad @ self.w
        grad_w = output_grad.T @ cache['x']
        grad_b = np.sum(output_grad, axis=0)
        params_grad = {'w': grad_w,
                       'b': grad_b}
        return grad_x, params_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.input_count = input_count
        self.alpha = alpha
        self.gamma = np.ones(input_count)
        self.beta = np.zeros(input_count)
        self.mean_t = np.zeros(input_count)
        self.variance_t = np.zeros(input_count)

    def get_parameters(self):
        params = {'gamma': self.gamma,
                  'beta': self.beta}
        return params

    def get_buffers(self):
        buffers = {'global_mean': self.mean_t,
                   'global_variance': self.variance_t}
        return buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        mean_batch = np.sum(x, axis=0) / len(x)
        variance_batch = np.sum((x - mean_batch)**2, axis=0) / len(x)

        est_x = (x - mean_batch) / np.sqrt(variance_batch + ESPILON)
        y = self.gamma * est_x + self.beta
        self.mean_t = (1 - self.alpha) * self.mean_t + self.alpha * mean_batch
        self.variance_t = (1 - self.alpha) * self.variance_t + self.alpha * variance_batch

        cache = (x, est_x, mean_batch, variance_batch)
        return y, cache

    def _forward_evaluation(self, x):
        est_x = (x - self.mean_t) / np.sqrt(self.variance_t + ESPILON)
        y = self.gamma * est_x + self.beta

        cache = {'x': x, 'est_x': est_x}
        return y, cache

    def backward(self, output_grad, cache):
        x, est_x, mean_batch, variance_batch = cache
        grad_gamma = np.sum(output_grad * est_x, axis=0)
        grad_beta = np.sum(output_grad, axis=0)

        deriv_est_x = output_grad * self.gamma
        deriv_var = np.sum(deriv_est_x * (x - mean_batch) * -0.5 * (variance_batch + ESPILON) ** -1.5, axis=0)
        deriv_mean = np.sum(deriv_est_x * -1 / np.sqrt(variance_batch + ESPILON), axis=0) + deriv_var * np.mean(
            -2 * (x - mean_batch), axis=0)
        grad_x = deriv_est_x / np.sqrt(variance_batch + ESPILON) + deriv_var * 2 * (x - mean_batch) / x.shape[
            0] + deriv_mean / x.shape[0]

        return grad_x, {"gamma": grad_gamma, "beta": grad_beta}


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        cache = {'y': y}
        return y, cache

    def backward(self, output_grad, cache):
        grad = ((1 - cache['y']) * cache['y']) * output_grad
        cache_ = {'gradient': grad}
        return grad, cache_


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = np.maximum(0, x)
        cache = {'x': x}
        return y, cache

    def backward(self, output_grad, cache):
        grad = np.where(cache['x'] > 0, 1, 0) * output_grad
        return grad, {}
