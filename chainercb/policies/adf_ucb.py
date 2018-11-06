from chainer import cuda, functions as F, as_variable

from chainercb.policies.adf import ADFPolicy


class ADFUCBPolicy(ADFPolicy):
    """
    A strictly linear policy that uses a per-arm Upper Confidence Bound
    estimation of performance.
    """
    def draw(self, x):
        x_r = F.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        out = self.regressor.ucb(x_r)
        result = F.reshape(out, (x.shape[0], x.shape[1]))
        return F.argmax(result, axis=1)

    def propensity(self, x, action):
        return as_variable(1.0 * (self.draw(x).data == action.data))

    def log_propensity(self, x, action):
        return F.log(self.propensity(x, action))
