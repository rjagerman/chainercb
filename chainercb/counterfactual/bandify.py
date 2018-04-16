from chainer import cuda, Chain, Variable


class Bandify(Chain):
    def __init__(self, acting_policy):
        super().__init__(acting_policy=acting_policy)

    def __call__(self, observations, labels):
        actions = self.acting_policy.draw(observations)
        log_propensities = self.acting_policy.log_propensity(observations,
                                                             actions)
        rewards = self.reward(actions, labels, dtype=observations.dtype)
        return observations, actions, log_propensities, rewards

    def reward(self, actions, labels, dtype):
        """
        Computes a reward for given actions and ground-truth labels

        :param actions: The actions to play
        :type actions: chainer.Variable

        :param labels: The ground truth labels
        :type labels: chainer.Variable

        :return: The reward
        :rtype: chainer.Variable
        """
        raise NotImplementedError


class MultiClassBandify(Bandify):
    """
    Chain to turn a multiclass problem into a counter factual bandit problem
    """
    def reward(self, actions, labels, dtype):
        xp = cuda.get_array_module(actions.data)
        array = xp.equal(actions.data, labels) * 1.0
        return Variable(array.astype(dtype))
