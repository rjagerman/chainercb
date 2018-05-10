from chainer import cuda, Chain, Variable


class Bandify(Chain):
    def __init__(self, acting_policy):
        super().__init__(acting_policy=acting_policy)
        self._hooks = []

    def __call__(self, *args):
        if len(args) != 2:
            raise RuntimeError('expecting 2 arguments for bandify: (x, y)')
        observations, labels = args
        actions = self.acting_policy.draw(observations)
        log_propensities = self.acting_policy.log_propensity(observations,
                                                             actions)
        rewards = self.reward(actions, labels, dtype=observations.dtype)
        self.call_hooks(observations, actions, log_propensities, rewards)
        return observations, actions, log_propensities, rewards

    def call_hooks(self, obs, actions, log_p, rewards):
        for hook in self._hooks:
            hook(obs, actions, log_p, rewards)

    def add_hook(self, fn):
        self._hooks.append(fn)

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
