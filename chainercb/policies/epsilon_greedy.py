from chainer import cuda, as_variable, functions as F
from chainercb.policy import Policy


class EpsilonGreedy(Policy):
    def __init__(self, policy, epsilon):
        """
        Repurposes a policy by choosing actions uniformly at random with
        probability epsilon and actions currently considered best with
        probability (1 - epsilon).

        :param policy: The underlying policy
        :type policy: chainercb.policy.Policy

        :param epsilon: The probability of exploring
        :type epsilon: float
        """
        super().__init__(policy=policy)
        self.epsilon = epsilon

    def draw(self, x):
        xp = cuda.get_array_module(x)

        # Get a full mini batch of actions from both max or uniform, these will
        # get mixed later on with a bernoulli selection strategy
        from_max = self.max(x)
        from_uniform = self.uniform(x)

        # Draw a bernoulli sample with probability epsilon for every item in the
        # mini batch
        draw = (self.rng(xp).random((x.shape[0])) < self.epsilon) * 1.0

        # Shape depends on the output from the underlying policy, so we
        # reshape to correctly broadcast
        draw = draw.reshape([draw.shape[0] if i == 0 else 1
                             for i in range(len(from_max.shape))])

        # Return with probability (1.0 - epsilon) from maximum and with
        # probability epsilon from uniform
        return (1.0 - draw) * from_max + draw * from_uniform

    def max(self, x):
        return self.policy.max(x)

    def uniform(self, x):
        return self.policy.uniform(x)

    def nr_actions(self, x):
        return self.policy.nr_actions(x)

    def log_nr_actions(self, x):
        return self.policy.log_nr_actions(x)

    def propensity(self, x, action):
        xp = cuda.get_array_module(x, action)
        max_action = self.max(x)
        if action.ndim > 1:
            p = 1.0 * (xp.all(action.data == max_action.data, axis=1))
        else:
            p = 1.0 * (action.data == max_action.data)
        p *= (1.0 - self.epsilon)
        p += self.epsilon / self.nr_actions(x)
        return as_variable(p.data.astype(dtype=x.dtype))

    def log_propensity(self, x, action):
        xp = cuda.get_array_module(x, action)
        max_action = self.max(x)
        if action.ndim > 1:
            p = 1.0 * (xp.all(action.data == max_action.data, axis=1))
        else:
            p = 1.0 * (action.data == max_action.data)
        p_inv = 1.0 - p
        p *= xp.log(1 - self.epsilon + self.epsilon / self.nr_actions(x).data)
        p_inv *= xp.log(self.epsilon) - self.log_nr_actions(x)
        return p + p_inv
