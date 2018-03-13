import numpy as np
from chainer import links as L
from chainercb.policies import Softmax


def setup_softmax_policy(tau=1.0):
    """
    Sets up a softmax policy

    :param tau: Temperature parameter dictating the smoothness of the
                distribution
    :type tau: float
    :return: A softmax policy with 3-dimensional input and 6 actions
    :rtype: chainercb.policies.Softmax
    """
    W = np.array([[0.5, 0.5, 0.5],
                  [1.0, -1.0, 1.0],
                  [0.9, 0.9, 0.1],
                  [-3.0, -3.0, -3.0],
                  [-1.0, 1.0, -1.0],
                  [0.1, 1.1, 0.5]])
    predictor = L.Linear(None, 6, initialW=W)
    return Softmax(predictor, tau=tau)
