from chainer import functions as F


class PerformanceEstimator():
    def __init__(self, policy):
        """
        Constructs a performance estimator for given policy

        :param policy: The policy whose performance is estimated
        :type policy: chainercb.policy.Policy
        """
        self.policy = policy

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
        raise NotImplementedError

    def lower_bound(self, delta=0.95):
        """
        Returns the lower bound on the performance estimate

        :param delta: The confidence interval coverage in (0.0, 1.0)
        :type delta: float

        :return: Returns the lower bound
        :rtype: float
        """
        raise NotImplementedError

    def estimate(self):
        """
        Returns the performance estimate

        :return: The performance estimate
        :rtype: float
        """
        raise NotImplementedError


class CHEstimator(PerformanceEstimator):
    def __init__(self, policy, decay=0.99):
        """
        Constructs a performance estimator that uses the Chernoff-Hoeffding
        inequality to estimate the lower bound

        :param policy: The policy to evaluate
        :type policy: chainercb.policy.Policy

        :param decay: The exponential moving decay factor
        :type decay: float
        """
        super().__init__(policy)
        self.decay = decay
        self.b = 1.0
        self.mean = 0.0
        self.bias_corrected_mean = 0.0
        self.lower_bound = 0.0
        self.n = 0

    def update_bounds(self, observations, actions, propensities, rewards):
        ips = self.policy.propensity(observations, actions).data / propensities.data
        self.b = max(self.b, float(F.max(ips * rewards).data))
        for i in range(observations.shape[0]):

            # Compute bias-corrected exponential moving average
            self.n += 1
            self.mean = self.mean * self.decay
            self.mean += (1.0 - self.decay) * (ips[i] * rewards[i])
            self.bias_corrected_mean = self.mean / (1 - self.decay**self.n)

    def lower_bound(self, delta=0.95):
        variance = self.b * F.sqrt(F.log(1.0 / delta) / (2 * self.n))
        return self.bias_corrected_mean - variance.data

    def estimate(self):
        return self.bias_corrected_mean


class MPeBEstimator(PerformanceEstimator):
    def __init__(self, policy, decay=0.99):
        """
        Constructs a performance estimator that uses the Maurer & Pontil
        empirical Bernstein inequality to estimate the lower bound

        :param policy: The policy to evaluate
        :type policy: chainercb.policy.Policy

        :param decay: The exponential moving decay factor
        :type decay: float
        """
        super().__init__(policy)
        self.decay = decay
        self.b = 1.0
        self.mean = 0.0
        self.bias_corrected_mean = 0.0
        self.lower_bound = 0.0
        self.n = 0

    def update_bounds(self, observations, actions, propensities, rewards):
        ips = self.policy.propensity(observations, actions).data / propensities.data
        self.b = max(self.b, float(F.max(ips * rewards).data))
        for i in range(observations.shape[0]):

            # Compute bias-corrected exponential moving average
            self.n += 1
            next_observation = ips[i] * rewards[i]
            self.mean = self.mean * self.decay
            self.mean += (1.0 - self.decay) * next_observation
            self.bias_corrected_mean = self.mean / (1 - self.decay**self.n)

            # Compute exponential moving sample variance
            diff = next_observation - self.mean
            self.variance = (1 - self.decay) * self.variance
            self.variance += (self.decay - self.decay**2) * diff**2

    def lower_bound(self, delta=0.95):
        log_term = F.log(2.0 / delta)
        std_1 = (7 * self.b * log_term) / (3*(self.n - 1))
        variance_term = (log_term / (self.n - 1)) * self.variance
        std_2 = (1 / self.n) * F.sqrt(variance_term)
        return self.bias_corrected_mean - std_1 - std_2

    def estimate(self):
        return self.bias_corrected_mean
