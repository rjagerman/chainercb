from chainer import as_variable, cuda, functions as F
from chainercb.policy import Policy


class Softmax(Policy):
    def __init__(self, predictor, tau=1.0):
        """
        A policy that repurposes the output of a prediction function via a
        softmax as a conditional probability distribution from which actions can
        be sampled. Drawing actions from this policy results in Boltzmann
        exploration.

        :param predictor: The predictor function
        :type predictor: chainer.Link

        :param tau: The temperature parameter dictating the smoothness of the
                    softmax
        :type tau: float
        """
        super().__init__(predictor=predictor)
        self.tau = tau

    def draw(self, x):
        # Construct a conditional probability distribution via softmax and then
        # sample a set of actions from that distribution
        p = self._log_propensities(x)
        return self._sample(p, self.rng(cuda.get_array_module(x)))

    def max(self, x):
        # The highest value from our predictor is, by definition, the arg max,
        # so we will use it
        return F.argmax(self.predictor(x), axis=1)

    def uniform(self, x):
        xp = cuda.get_array_module(x)

        # Generate a uniform random sample as if it is a softmax where all
        # values are equal
        p = F.log_softmax(xp.ones(self.predictor(x).shape))
        return self._sample(p, self.rng(xp))

    def nr_actions(self, x):
        xp = cuda.get_array_module(x)
        prediction_shape = self.predictor(x).shape
        return as_variable(xp.ones(prediction_shape[0]) * prediction_shape[1])

    def propensity(self, x, action):
        probabilities = F.softmax(self._predict(x))
        return F.select_item(probabilities, action.data)

    def log_propensity(self, x, action):
        log_probabilities = self._log_propensities(x)
        return F.select_item(log_probabilities, action.data)

    def _log_propensities(self, x):
        """
        Computes the log propensity (or log probability) of executing the
        actions. In the case of a softmax, we can just use the log_softmax
        function to give us these values

        :param x: The context vectors x
        :type x: chainer.Variable

        :return: The log propensities
        :rtype: chainer.Variable
        """
        return F.log_softmax(self._predict(x))

    def _predict(self, x):
        """
        Generates a prediction (and rescales it according to the temperature
        parameter)

        :param x: The context vectors
        :type x: chainer.Variable

        :return: The predictions made by our predictor (read neural network)
        :rtype: chainer.Variable
        """
        return self.predictor(x) / self.tau

    def _sample(self, log_p, rng=None):
        """
        Samples an index with probabilities p. This is a modification to the
        reservoir sampling algorithm made efficient on GPUs and numerically
        stable by operating on log-probabilities

        :param log_p: The log probabilities per row
        :type log_p: chainer.Variable

        :param rng: The random number generator
        :type rng: numpy.random.RandomState|cupy.random.RandomState|None

        :return: For each row, one index, sampled proportional to p
        :rtype: chainer.Variable
        """
        xp = cuda.get_array_module(log_p)

        if rng is None:
            rng = xp.random

        u = rng.uniform(0.0, 1.0, log_p.shape).astype(dtype=log_p.dtype)
        r = F.log(-F.log(u)) - log_p
        return F.argmin(r, axis=1)
