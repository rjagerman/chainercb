import numpy as np
from chainer import as_variable
from chainercb.util import RidgeRegression
from chainer.testing import assert_allclose

from chainercb.util.ridge import WindowRidgeRegression


def test_predict():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)

    # No observations
    assert_allclose(r.predict(x).data,
                    np.array([0.0, 0.0, 0.0]))

    # Update once
    r.update(x, y)
    assert_allclose(r.predict(x).data,
                    np.array([0.99656349, 1.03779954, -0.90377945]))

    # Update 100 times
    for _ in range(100):
        r.update(x, y)
    assert_allclose(r.predict(x).data,
                    np.array([1.00018692, 1.00122452, -0.99799728]))


def test_predict_vector():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)

    # No observations
    assert_allclose(r.predict(x).data,
                    np.array([0.0, 0.0, 0.0]))

    # Update once
    for i in range(x.shape[0]):
        r.update(x[i, :][None, :], y[i][None])
    assert_allclose(r.predict(x).data,
                    np.array([0.99656349, 1.03779954, -0.90377945]))

    # Update 100 times
    for _ in range(100):
        for i in range(x.shape[0]):
            r.update(x[i, :][None, :], y[i][None])
    assert_allclose(r.predict(x).data,
                    np.array([1.00018692, 1.00122452, -0.99799728]))


def test_ucb():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)

    # No observations
    assert_allclose(r.ucb(x).data,
                    np.array([5.29150262, 5.29150262, 4.12310563]))

    # Update once
    r.update(x, y)
    assert_allclose(r.ucb(x).data,
                    np.array([1.94904291, 1.91711522, -0.13497172]))

    # Update 100 times
    for _ in range(100):
        r.update(x, y)
    assert_allclose(r.ucb(x).data,
                    np.array([1.09963632, 1.1004903, -0.89896227]))


def test_thompson_no_observations():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)
    np.random.seed(42)

    assert_allclose(r.thompson(x).data,
                    np.array([-1.7033947, 0.87402812, -0.28144131]))
    assert_allclose(r.thompson(x).data,
                    np.array([1.47054412, 6.77040797, -3.95803068]))
    assert_allclose(r.thompson(x).data,
                    np.array([-5.36107422, -4.0085478, 2.02298249]))


def test_thompson_1_observation():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)
    np.random.seed(42)

    # Update once
    r.update(x, y)
    assert_allclose(r.thompson(x).data,
                    np.array([0.04086122, 1.62961431, -1.34344107]))
    assert_allclose(r.thompson(x).data,
                    np.array([0.64437957, 2.09125993, -1.14306858]))
    assert_allclose(r.thompson(x).data,
                    np.array([1.12360373, 0.2720241, -1.58045177]))


def test_thompson_100_observations():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)
    np.random.seed(42)

    # Update 100 times
    for _ in range(100):
        r.update(x, y)
    assert_allclose(r.thompson(x).data,
                    np.array([0.8769278, 1.0556106, -1.07069319]))
    assert_allclose(r.thompson(x).data,
                    np.array([0.97438225, 1.04545584, -1.06124991]))
    assert_allclose(r.thompson(x).data,
                    np.array([1.04556294, 0.90065285, -1.06563813]))


def test_thompson_distribution():
    r = RidgeRegression(6)
    x = np.array([[1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
                  [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                  [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    x = as_variable(x)
    y = as_variable(y)
    np.random.seed(42)
    r.update(x, y)

    # Compute statistics according to analytical method
    means, stds = r.thompson_distribution(x)

    # Compute sample statistics
    nr_samples = 10000
    samples = np.zeros((nr_samples, x.shape[0]))
    for i in range(nr_samples):
        samples[i, :] = r.thompson(x).data

    # Assert analytical answers are close to statistical results
    assert_allclose(np.mean(samples, axis=0), means.data, rtol=1e-2, atol=1e-2)
    assert_allclose(np.std(samples, axis=0), stds.data, rtol=1e-2, atol=1e-2)

