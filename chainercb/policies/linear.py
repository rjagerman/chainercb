from chainer import cuda, functions as F, as_variable

from chainercb.policy import Policy
from chainercb.util import RidgeRegression


class LinearPolicy(Policy):
    """
    A strictly linear, finite-arm, policy that uses a per-arm regressor.
    """

    def __init__(self, k, d, alpha=1.0, regularizer=1.0, device=None):
        """
        :param k: The number of arms (actions)
        :type k: int

        :param d: The number of dimensions (features)
        :type d: int

        :param alpha: The variance scaling factor for UCB
        :type alpha: float

        :param regularizer: The ridge regression regularization constant
        :type regularizer: float

        :param device: The GPU device to use or None to use CPU
        :type device: int|None
        """
        super().__init__()
        self.k = k
        self.d = d
        self.regressors = [RidgeRegression(d, alpha, regularizer, device)
                           for _ in range(self.k)]

    def max(self, x):
        xp = cuda.get_array_module(x)
        pred = [self.regressors[a].predict(x).data for a in range(self.k)]
        pred = [xp.reshape(p, (p.shape[0], 1)) for p in pred]
        pred = xp.hstack(pred)
        return F.argmax(pred, axis=1)

    def uniform(self, x):
        xp = cuda.get_array_module(x)
        return as_variable(xp.random.randint(self.k, size=(x.shape[0])))

    def nr_actions(self, x):
        xp = cuda.get_array_module(x)
        return as_variable(xp.ones(x.shape[0]) * self.k)

    def log_nr_actions(self, x):
        return F.log(self.nr_actions(x))

    def update(self, x, actions, log_p, rewards):
        xp = cuda.get_array_module(actions)
        for a in xp.unique(actions.data):
            c_x = x.data[actions.data == a, :]
            c_r = rewards.data[actions.data == a]
            self.regressors[a].update(as_variable(c_x), as_variable(c_r))
