from chainer import cuda, as_variable, functions as F
from chainercb.policy import Policy


class Explore(Policy):
    def __init__(self, policy):
        """
        Repurposes a policy by only choosing actions uniformly at random. This
        results in an explore-only policy.

        :param policy: The underlying policy
        :type policy: chainercb.policy.Policy
        """
        super().__init__(policy=policy)

    def draw(self, x):
        return self.uniform(x)

    def max(self, x):
        return self.policy.max(x)

    def uniform(self, x):
        return self.policy.uniform(x)

    def nr_actions(self, x):
        return self.policy.nr_actions(x)

    def propensity(self, x, action):
        xp = cuda.get_array_module(x)
        output = xp.ones(x.shape[0]) / (1.0 * self.nr_actions(x))
        return as_variable(output.data.astype(dtype=x.dtype))

    def log_propensity(self, x, action):
        return F.log(self.propensity(x, action))
