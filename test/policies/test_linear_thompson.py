import numpy as np
from chainer import as_variable
from chainer.testing import assert_allclose

from chainercb.bandify import MultiClassBandify
from chainercb.policies import ThompsonPolicy


def test_draw():
    policy = ThompsonPolicy(4, 6)

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = as_variable(np.random.random((32, 6)).astype(np.float32))

    # Assert draw
    expected = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3,
                         3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3])
    assert_allclose(policy.draw(x).data, expected)


def test_update():
    policy = ThompsonPolicy(4, 6)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))

    # Drawing at this point should return the default action
    expected = np.array([0, 1, 2, 3])
    assert_allclose(policy.draw(x).data, expected)

    # Perfect update
    log_p = as_variable(np.zeros(y.shape))
    for _ in range(100):
        a = as_variable(np.random.randint(4, size=y.shape))
        r = (1.0 * (a.data == y.data))
        policy.update(x, a, log_p, as_variable(r))

    # Drawing at this point should be perfect
    expected = np.array([2, 1, 0, 3])
    assert_allclose(policy.draw(x).data, expected)


def test_bandify_update():
    policy = ThompsonPolicy(4, 6)
    mcb = MultiClassBandify(policy)
    mcb.update_policy(policy)

    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))
    np.random.seed(42)

    # Drawing at this point should return the default action
    expected = np.array([0, 1, 2, 3])
    assert_allclose(policy.draw(x).data, expected)

    # Run multi class bandify chain
    for _ in range(100):
        _, a, log_p, r = mcb(x, y)

    # Drawing at this point should be perfect
    expected = np.array([2, 1, 0, 3])
    assert_allclose(policy.draw(x).data, expected)


def test_bandify_propensity_sum():
    k = 4
    policy = ThompsonPolicy(k, 6)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))

    # Drawing at this point should return the default action
    expected = np.array([0, 1, 2, 3])
    assert_allclose(policy.draw(x).data, expected)

    # Do a perfect update for all actions (full information essentially)
    for a in range(k):
        log_p = as_variable(np.zeros(y.shape))
        actions = as_variable(np.ones(4, dtype=np.int32) * a)
        r = (1.0 * (actions.data == y.data))
        policy.update(x, actions, log_p, as_variable(r))

    # Compute propensity scores for all actions, given the contexts
    results = np.zeros((x.shape[0], k))
    for a in range(k):
        actions = as_variable(np.ones(4, dtype=np.int32) * a)
        results[:, a] = policy.propensity(x, actions).data

    assert_allclose(np.sum(results, axis=1), np.ones(4), atol=1e-2, rtol=1e-2)


def test_bandify_propensity_statistic():
    k = 4
    policy = ThompsonPolicy(k, 6)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))

    # Drawing at this point should return the default action
    expected = np.array([0, 1, 2, 3])
    assert_allclose(policy.draw(x).data, expected)

    # Do a perfect update for all actions (full information essentially)
    for a in range(k):
        log_p = as_variable(np.zeros(y.shape))
        actions = as_variable(np.ones(4, dtype=np.int32) * a)
        r = (1.0 * (actions.data == y.data))
        policy.update(x, actions, log_p, as_variable(r))

    # Compute propensity scores for all actions, given the contexts
    results = np.zeros((x.shape[0], k))
    for a in range(k):
        actions = as_variable(np.ones(4, dtype=np.int32) * a)
        results[:, a] = policy.propensity(x, actions).data

    # Compute sampled propensity scores
    nr_samples = 10000
    samples = np.zeros((x.shape[0], k))
    for i in range(nr_samples):
        for a in range(k):
            actions = policy.draw(x).data
            samples[np.arange(samples.shape[0]), actions] += 1.0

    samples /= np.sum(samples, axis=1)
