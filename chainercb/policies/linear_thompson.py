from math import factorial
from chainer import cuda, functions as F, as_variable

from chainercb.policies.linear import LinearPolicy
from chainercb.util import select_items_per_row


class ThompsonPolicy(LinearPolicy):
    """
    A strictly linear policy that uses thompson sampling to draw actions.
    """
    def draw(self, x):
        xp = cuda.get_array_module(x)
        ts = [self.regressors[a].thompson(x).data for a in range(self.k)]
        ts = [xp.reshape(t, (t.shape[0], 1)) for t in ts]
        ts = xp.hstack(ts)
        return F.argmax(ts, axis=1)

    def propensity(self, x, action):
        xp = cuda.get_array_module(x)
        """: type: numpy"""

        # Compute independent thompson sample distributions
        z_means = xp.zeros((x.shape[0], self.k))
        z_std = xp.zeros((x.shape[0], self.k))
        for a in range(self.k):
            m, s = self.regressors[a].thompson_distribution(x)
            z_means[:, a] = m.data
            z_std[:, a] = s.data

        # Compute the argmax probability
        m_i, m_j = _tiles(z_means)
        s_i, s_j = _tiles(z_std)

        c_m = _cut_diagonals(m_i - m_j).data
        c_s = _cut_diagonals(s_i + s_j).data

        opts = factorial(self.k - 1)

        res = xp.prod(0.5 * (1 + F.erf(c_m / (xp.sqrt(2) * c_s)).data), axis=2)

        a = F.reshape(action, (action.shape[0], 1))
        res = select_items_per_row(as_variable(res), a)
        return F.reshape(res, action.shape)

    def log_propensity(self, x, action):
        return F.log(self.propensity(x, action))


def _tiles(x):
    xp = cuda.get_array_module(x)
    x = x.data
    x_i = xp.reshape(x, (x.shape[0], x.shape[1], 1))
    x_j = xp.reshape(x, (x.shape[0], 1, x.shape[1]))
    x_i = xp.broadcast_to(x_i, (x.shape[0], x.shape[1], x.shape[1]))
    x_j = xp.broadcast_to(x_j, (x.shape[0], x.shape[1], x.shape[1]))
    return as_variable(x_i), as_variable(x_j)


def _cut_diagonals(x):
    xp = cuda.get_array_module(x)
    x = x.data
    e = xp.reshape(xp.eye(x.shape[1]), (1, x.shape[1], x.shape[2]))
    e = xp.broadcast_to(e, x.shape)
    res = xp.reshape(x[e == 0.0], (x.shape[0], x.shape[1], x.shape[2] - 1))
    return as_variable(res)
