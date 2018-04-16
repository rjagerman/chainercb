from chainercb.counterfactual import CHEstimator, MPeBEstimator
from chainercb.policy import Policy


class SafePolicy(Policy):

    def __init__(self, safe_policy, new_policy, safe_estimator, new_estimator):
        """
        Constructs a SEA-type policy which uses a safe policy and a new policy
        where actions from the new policy are only executed if its estimated
        performance is at least as good as that of the safe policy. Otherwise
        actions from the safe policy are executed.

        :param safe_policy: The safe policy
        :type safe_policy: chainercb.policy.Policy

        :param new_policy: The new policy we wish to optimize
        :type new_policy: chainercb.policy.Policy
        """
        super().__init__(safe_policy=safe_policy, new_policy=new_policy)
        self.safe_estimator = safe_estimator
        self.new_estimator = new_estimator

    def update_bounds(self, observations, actions, propensities, rewards):
        """
        Updates the performance bound estimates with a mini-batch of
        observations

        :param observations: The observations (or feature vectors)
        :type observations: chainer.Variable

        :param actions: The actions that were taken
        :type actions: chainer.Variable

        :param propensities: The propensities
        :type propensities: chainer.Variable

        :param rewards: The observed rewards
        :type rewards: chainer.Variable
        """
        self.safe_estimator.update_bounds(observations, actions, propensities,
                                          rewards)
        self.new_estimator.update_bounds(observations, actions, propensities,
                                         rewards)

    def _select_policy(self):
        """
        :return: The policy with the highst lower bound on performance
        :rtype chainercb.policy.Policy
        """
        if self.new_estimator.lower_bound() > self.safe_estimator.lower_bound():
            return self.new_policy
        else:
            return self.safe_policy

    def draw(self, x):
        self._select_policy().draw(x)

    def max(self, x):
        self._select_policy().max(x)

    def uniform(self, x):
        self._select_policy().uniform(x)

    def propensity(self, x, action):
        self._select_policy().propensity(x, action)

    def log_propensity(self, x, action):
        self._select_policy().log_propensity(x, action)


class CHSafe(SafePolicy):
    def __init__(self, safe_policy, new_policy, decay=0.99):
        """
        Constructs a Chernoff-Hoeffding estimated SEA policy

        :param safe_policy: The safe policy
        :type safe_policy: chainercb.policy.Policy

        :param new_policy: The new policy to learn
        :type new_policy: chainercb.policy.Policy

        :param decay: The decay on the exponential moving mean
        :type decay: float
        """
        super().__init__(safe_policy, new_policy,
                         CHEstimator(safe_policy, decay),
                         CHEstimator(new_policy, decay))


class MPeBSafe(SafePolicy):
    def __init__(self, safe_policy, new_policy, decay=0.99):
        """
        Constructs a Maurer & Pontil Emperical Bernstein estimated SEA policy

        :param safe_policy: The safe policy
        :type safe_policy: chainercb.policy.Policy

        :param new_policy: The new policy to learn
        :type new_policy: chainercb.policy.Policy

        :param decay: The decay on the exponential moving mean
        :type decay: float
        """
        super().__init__(safe_policy, new_policy,
                         MPeBEstimator(safe_policy, decay),
                         MPeBEstimator(new_policy, decay))
