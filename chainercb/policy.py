from chainer import Chain, functions as F, cuda


class Policy(Chain):
    """
    Abstract class representing a policy
    """

    def draw(self, x):
        """
        Draws actions stochastically for given batch of context vectors x

        :param x: The context vectors
        :type x: chainer.Variable

        :return: The actions drawn stochastically from the policy
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def max(self, x):
        """
        Selects actions deterministically with the highest propensity score for
        given batch of context vectors x

        :param x: The context vectors
        :type x: chainer.Variable

        :return: The best actions according to the policy
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def uniform(self, x):
        """
        Selects actions uniformly at random for given batch of context vectors x

        :param x: The context vectors
        :type x: chainer.Variable

        :return: Actions selected uniformly at random
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def nr_actions(self, x):
        """
        The number of actions that the policy can possibly execute for given
        batch of context vectors x

        :param x: The context vector
        :type x: chainer.Variable

        :return: The number of actions that can be executed for each vector x
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def log_nr_actions(self, x):
        """
        Returns the logarithm of the number of actions that the policy can
        possibly execute for a given batch of context vectors x

        :param x: The context vector
        :type x: chainer.Variable

        :return: The logarithm of the number of actions that can be executed for
                 each vector x
        :rtype: chainer.Variable
        """
        return F.log(self.nr_actions(x))

    def propensity(self, x, action):
        """
        Computes the propensity scores of a batch of actions for a given batch
        of context vectors x

        :param x: The context vectors
        :type x: chainer.Variable

        :param action: The actions to execute
        :type action: chainer.Variable

        :return: The propensity score(s) of the given action(s)
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def log_propensity(self, x, action):
        """
        Computes the logarithm of the propensity score of a batch of actions for
        a given batch of context vectors x

        :param x: The context vectors
        :type x: chainer.Variable

        :param action: The actions to execute
        :type action: chainer.Variable

        :return: The log propensity score(s) of the given action(s)
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def __call__(self, x, actions, log_p, rewards):
        """
        Updates the policy with given batch of observations, actions,
        log_probabilities and rewards. This can also be ignored (e.g. in the
        case where we would backprop through log_p).
        By default this method does nothing.

        :param x: The context vectors
        :type x: chainer.Variable

        :param actions: The actions that were executed
        :type actions: chainer.Variable

        :param log_p: The log propensity score(s) of the given action(s)
        :type log_p: chainer.Variable

        :param rewards: The obtained rewards for the chosen actions
        :type rewards: chainer.Variable
        """
        pass
