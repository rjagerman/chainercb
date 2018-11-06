from chainer import cuda, functions as F, as_variable

from chainercb.policies.linear import LinearPolicy


class LinUCBPolicy(LinearPolicy):
    """
    A strictly linear policy that uses a per-arm Upper Confidence Bound
    estimation of performance.
    """
    def draw(self, x):
        xp = cuda.get_array_module(x)
        ucbs = [self.regressors[a].ucb(x).data for a in range(self.k)]
        ucbs = [xp.reshape(ucb, (ucb.shape[0], 1)) for ucb in ucbs]
        ucbs = xp.hstack(ucbs)
        return F.argmax(ucbs, axis=1)

    def propensity(self, x, action):
        return as_variable(1.0 * (self.draw(x).data == action.data))

    def log_propensity(self, x, action):
        return F.log(self.propensity(x, action))
