from chainer import cuda, Chain, Variable, as_variable


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
        rewards = self.reward(actions, as_variable(labels),
                              dtype=observations.dtype)

        self._call_hooks(observations, actions, log_propensities, rewards)

        return observations, actions, log_propensities, rewards

    def _call_hooks(self, x, actions, log_p, rewards):
        """
        Calls the internal hooks
        
        :param x: The context vectors
        :type x: chainer.Variable

        :param actions: The actions that were executed
        :type actions: chainer.Variable

        :param log_p: The log propensity score(s) of the given action(s)
        :type log_p: chainer.Variable

        :param rewards: The obtained rewards for the chosen actions
        :type rewards: chainer.Variable
        """
        for hook in self._hooks:
            hook(x, actions, log_p, rewards)

    def update_policy(self, policy):
        """
        Updates the given policy in this bandit chain. That is, whenever an
        action is executed and reward is obtained, this is propagated to the
        policy so it can update its state.

        :param policy:
        :type policy: chainercb.policy.Policy
        """
        self._hooks.append(policy)

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
    Chain to turn a multiclass problem into a contextual bandit problem
    """
    def reward(self, actions, labels, dtype):
        xp = cuda.get_array_module(actions.data)
        array = xp.equal(actions.data, labels.data) * 1.0
        return Variable(array.astype(dtype))
