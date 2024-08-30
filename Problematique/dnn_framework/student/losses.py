import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        soft_x = softmax(x)

        target_one_hot = np.zeros_like(soft_x)
        target_one_hot[np.arange(x.shape[0]), target] = 1

        grad = (soft_x - target_one_hot) / x.shape[0]

        return -np.sum(target_one_hot * np.log(soft_x)) / x.shape[0], grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        grad = 2 * (x - target) / x.size

        return (1 / x.size) * np.sum((x - target) ** 2), grad
