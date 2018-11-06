from collections import deque

import numpy as np
from chainer import as_variable
from chainer.backends import cuda


class RidgeRegression:
    def __init__(self, d, alpha=1.0, regularization=1.0, device=None):
        """
        Initializes the ridge regression estimate

        :param d: The dimensionality
        :type d: int
        
        :param alpha: The variance scaling factor
        :type alpha: float

        :param regularization: The regularization parameter
        :type regularization: float

        :param device: Device on which to perform ridge regression
        :type device: int|None
        """
        self.device = device
        self._set_xp()
        self._d = d
        self._alpha = alpha
        self._regularization = regularization
        self._A = self.xp.identity(self._d,
                                   dtype=np.float32) * self._regularization
        self._A_inv = self.xp.identity(self._d,
                                       dtype=np.float32) / self._regularization
        self._b = self.xp.zeros(self._d, dtype=np.float32)
        self._theta = self._A_inv.dot(self._b)
        self._compute_cholesky = True
        self._cho = None

    def update(self, x, r, sub=False):
        """
        Updates the ridge regression estimate

        :param x: Batch of feature vectors, matrix of shape (n, d)
        :type x: chainer.Variable

        :param r: Batch of targets, vector of shape (n)
        :type r: chainer.Variable

        :param sub: Whether to subtract a data point
        :type sub: bool
        """
        sub = -1.0 if sub is False else 1.0
        x = x.data
        r = r.data
        x_m = self.xp.reshape(x, (x.shape[0], x.shape[1], 1))
        x_m_T = self.xp.reshape(x, (x.shape[0], 1, x.shape[1]))
        to_add = sub * self.xp.sum(self.xp.matmul(x_m, x_m_T), axis=0)
        self._A += to_add
        self._b += sub * self.xp.sum(self.xp.broadcast_to(r[:, None], x.shape) * x,
                                     axis=0)
        if x.shape[0] == 1:
            # Perform Sherman-Morrison fast incremental inversion update
            prev = self._A_inv
            x_m = x_m[0, :, :]
            x_m_T = x_m_T[0, :, :]

            numerator = self.xp.matmul(prev, x_m)
            numerator = self.xp.matmul(numerator, x_m_T)
            numerator = self.xp.matmul(numerator, prev)

            denominator = self.xp.matmul(x_m_T, prev)
            denominator = self.xp.matmul(denominator, x_m)
            denominator = 1 + denominator

            self._A_inv = prev - numerator / denominator
        else:
            # Compute actual matrix inverse
            self._A_inv = self.xp.linalg.inv(self._A)
        self._compute_cholesky = True
        self._theta = self._A_inv.dot(self._b)

    def predict(self, x):
        """
        Predicts target values for given batch of feature vectors x

        :param x: Batch of feature vectors, matrix of shape (n, d)
        :type x: numpy.ndarray

        :return: Predicted target values, vector of shape (n)
        :rtype: numpy.ndarray
        """
        return as_variable(self.xp.dot(self._theta, x.data.T))

    def ucb(self, x):
        """
        Computes the upper confidence bound on predictions for given batch of
        feature vectors x

        :param x: Batch of feature vectors, matrix of shape (n, d)
        :type x: numpy.ndarray

        :return: The predicted target values with an upper confidence bound,
                 vector of shape (n)
        :rtype: numpy.ndarray
        """
        mean = self.predict(x).data
        x = x.data
        dev = self.xp.diag(self.xp.dot(self.xp.dot(x, self._A_inv), x.T))
        return as_variable(mean + self._alpha * self.xp.sqrt(dev))

    def thompson(self, x):
        """
        Computes thompson sampled predictions for given batch of feature
        vectors x

        :param x: Batch of feature vectors, matrix of shape (n, d)
        :type x: chainer.Variable

        :return: The predicted target values via thompson sampling, vector of
                 shape (n)
        :rtype: chainer.Variable
        """

        # This samples a theta from a multivariate normal, we avoid using
        # np.random.multivariate_normal due to numerical stability
        x = x.data
        cho = self._cholesky_decomposition()
        u = self.xp.random.standard_normal(size=self._theta.shape)
        sampled_theta = self._theta + self.xp.matmul(cho, u)

        # Predictions based on the sampled theta
        return as_variable(self.xp.dot(sampled_theta, x.T))

    def thompson_distribution(self, x):
        """
        Computes the distribution of the thompson sampled predictions for given
        batch of feature vectors x

        :param x: Batch of feature vectors, matrix of shape (n, d)
        :type x: chainer.Variable

        :return: A tuple where the first entry contains the means for the batch
                 and the second entry contains the std for the batch
        :rtype: (chainer.Variable, chainer.Variable)
        """
        x = x.data
        mean = self.xp.dot(x, self._theta)
        std = self.xp.diag(self.xp.matmul(self.xp.matmul(x, self._A_inv), x.T))
        std = self.xp.sqrt(std)
        return as_variable(mean), as_variable(std)

    def _cholesky_decomposition(self):
        """
        Computes the cholesky decomposition if it does not exist in cache,
        otherwise this returns the cached version. Any update to the model will
        invalidate the cache since the decomposition will have to be recomputed.

        :return: The cholesky decomposition of A inverse
        :rtype: numpy.ndarray|cupy.ndarray
        """
        if self._compute_cholesky:
            self._cho = self.xp.linalg.cholesky(self._A_inv)
            self._compute_cholesky = False
        return self._cho

    def __getstate__(self):
        # This customizes pickle behavior (to prevent xp from being pickled)
        d = dict(self.__dict__)
        del d['xp']
        return d

    def __setstate__(self, d):
        # This customizes pickle behavior (make sure xp is properly reloaded)
        self.__dict__.update(d)
        self._set_xp()

    def _set_xp(self):
        """
        Sets xp (the numpy or cupy module) depending on whether we execute on
        CPU or GPU.
        """
        if self.device is None:
            self.xp = cuda.get_array_module(np.array([0.0]))
        else:
            self.xp = cuda.get_array_module(cuda.to_gpu(np.array([0.0]),
                                                        device=self.device))

