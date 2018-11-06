from chainer import cuda, functions as F, as_variable

from chainercb.policy import Policy
from chainercb.util import RidgeRegression, select_items_per_row


class ADFPolicy(Policy):
    """
    A strictly linear, finite-arm, policy that uses a single regressor and where
    actions are represented by feature-vectors. This is sometimes referred to as
    action-dependent features (ADF).
    """

    def __init__(self, d, alpha=1.0, regularizer=1.0, device=None):
        """
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
        self.d = d
        self.regressor = RidgeRegression(d, alpha, regularizer, device)

    def max(self, x):
        x_r = F.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        out = self.regressor.predict(x_r)
        result = F.reshape(out, (x.shape[0], x.shape[1]))
        return F.argmax(result, axis=1)

    def uniform(self, x):
        xp = cuda.get_array_module(x)
        result = xp.random.random((x.shape[0], x.shape[1]))
        return F.argmax(result, axis=1)

    def nr_actions(self, x):
        xp = cuda.get_array_module(x)
        return as_variable(xp.ones(x.shape[0]) * x.shape[1])

    def log_nr_actions(self, x):
        return F.log(self.nr_actions(x))

    def update(self, x, actions, log_p, rewards):
        xp = cuda.get_array_module(x)
        x_r = F.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        incr = xp.arange(x.shape[0], dtype=actions.dtype) * x.shape[1]
        a = x_r[actions.data + incr, :]
        self.regressor.update(a, rewards)
