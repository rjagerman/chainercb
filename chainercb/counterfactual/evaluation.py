import numpy as np
from chainer import functions as F, cuda, as_variable, report


class PerformanceEstimator():
    def __init__(self, policy):
        """
        Constructs a performance estimator for given policy

        :param policy: The policy whose performance is estimated
        :type policy: chainercb.policy.Policy
        """
        self.policy = policy

    def __call__(self, observations, actions, log_propensities, rewards):
        self.update_bounds(observations, actions, log_propensities, rewards)

    def update_bounds(self, observations, actions, log_propensities, rewards):
        """
        Updates the performance bound estimates with a mini-batch of
        observations

        :param observations: The observations (or feature vectors)
        :type observations: chainer.Variable

        :param actions: The actions that were taken
        :type actions: chainer.Variable

        :param log_propensities: The propensities
        :type log_propensities: chainer.Variable

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
    def __init__(self, policy, decay=0.99, clip=None):
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
        self.n = 0
        self.clip = clip

    def update_bounds(self, observations, actions, log_propensities, rewards):

        # Compute IPS weighted rewards, part of this happens in log-space for
        # numerical stability, the final result is not in log-space.
        pol_log_p = self.policy.log_propensity(observations, actions).data
        log_ips = pol_log_p - log_propensities.data
        xp = cuda.get_array_module(log_ips)
        if self.clip is not None:
            log_ips = F.minimum(log_ips, as_variable(xp.ones(log_ips.shape, dtype=log_ips.dtype) * self.clip))
        ips = F.exp(log_ips)
        ips_r = F.mean(ips * rewards, axis=1)

        # To update exponential moving average we use a closed-form batch-update
        # that can be GPU accelerated
        weights = self.decay ** (xp.arange(observations.shape[0], 0, -1.0,
                                           dtype=log_propensities.dtype) - 1.0)
        self.b = max(self.b, float(F.max(ips_r).data))
        summation = (1 - self.decay) * F.sum(as_variable(weights) * ips_r,
                                             axis=0)
        self.mean = (self.decay ** observations.shape[0]) * self.mean + \
                    summation.data
        self.n += observations.shape[0]
        self.bias_corrected_mean = self.mean / (1 - self.decay ** self.n)

    def lower_bound(self, delta=0.95):
        nr_samples = np.maximum(1, 2 * self.n)
        variance = self.b * np.sqrt(np.log(1.0 / delta) / nr_samples)
        if np.isnan(variance):
            variance = np.infty
        return self.bias_corrected_mean - variance

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
        self.n = 0

    def update_bounds(self, observations, actions, log_propensities, rewards):
        pol_log_p = self.policy.log_propensity(observations, actions).data
        ips = F.exp(pol_log_p - log_propensities.data)
        self.b = max(self.b, float(F.max(ips * rewards).data))
        for i in range(observations.shape[0]):

            # Compute bias-corrected exponential moving average
            self.n += 1
            next_observation = ips[i] * rewards[i]
            self.mean = self.mean * self.decay
            self.mean += (1.0 - self.decay) * next_observation

            # Compute exponential moving sample variance
            diff = next_observation - self.mean
            self.variance = (1 - self.decay) * self.variance
            self.variance += (self.decay - self.decay**2) * diff**2
        self.bias_corrected_mean = self.mean / (1 - self.decay ** self.n)

    def lower_bound(self, delta=0.95):
        log_term = F.log(2.0 / delta)
        std_1 = (7 * self.b * log_term) / (3*(self.n - 1))
        variance_term = (log_term / (self.n - 1)) * self.variance
        std_2 = (1 / self.n) * F.sqrt(variance_term)
        print(self.bias_corrected_mean - std_1 - std_2)
        return self.bias_corrected_mean - std_1 - std_2

    def estimate(self):
        return self.bias_corrected_mean
