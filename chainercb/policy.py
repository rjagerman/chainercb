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

    def rng(self, xp, seed=None):
        """
        Returns the random number generator for this policy

        :param xp: Numpy or cupy module depending on whether we want to generate
                   numbers on cpu or gpu
        :type xp: numpy|cupy

        :param seed: The seed
        :type seed: int|None

        :return: A random number generator
        :rtype: numpy.random.RandomState|cupy.random.RandomState
        """
        if not hasattr(self, '_rng') or self._rng is None:
            self.seed_rng(xp, seed)
        return self._rng

    def seed_rng(self, xp, seed=None):
        """
        (Re)-seeds the random number generator

        :param xp: Numpy or cupy module depending on whether we want to generate
                   numbers on cpu or gpu
        :type xp: numpy|cupy

        :param seed: The seed
        :type seed: int|None
        """
        self._rng = xp.random.RandomState(seed)
