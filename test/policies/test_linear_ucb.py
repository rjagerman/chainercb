import numpy as np
from chainer import as_variable
from chainer.testing import assert_allclose

from chainercb.bandify import MultiClassBandify
from chainercb.policies import LinUCBPolicy


def test_draw():
    policy = LinUCBPolicy(4, 6)

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = as_variable(np.random.random((32, 6)).astype(np.float32))

    # Assert draw
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert_allclose(policy.draw(x).data, expected)


def test_update():
    policy = LinUCBPolicy(4, 6)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))

    # Drawing at this point should return the default action
    expected = np.array([0, 0, 0, 0])
    assert_allclose(policy.draw(x).data, expected)

    # Perfect update
    log_p = as_variable(np.zeros(y.shape))
    r = as_variable(np.ones((y.shape[0], 1)))
    for _ in range(100):
        a = as_variable(np.random.randint(4, size=y.shape))
        r = (1.0 * (a.data == y.data))
        policy.update(x, a, log_p, as_variable(r))

    # Drawing at this point should be perfect
    expected = np.array([2, 1, 0, 3])
    assert_allclose(policy.draw(x).data, expected)


def test_bandify_update():
    policy = LinUCBPolicy(4, 6)
    mcb = MultiClassBandify(policy)
    mcb.update_policy(policy)

    x = as_variable(np.array([[1.0, 2.0, 3.0, 3.0, -2.0, -1.0],
                              [2.0, 3.0, 1.0, -1.0, -3.0, -2.0],
                              [-1.0, -2.0, -1.0, 1.0, 3.0, 1.0],
                              [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]]))
    y = as_variable(np.array([2, 1, 0, 3]))

    # Drawing at this point should return the default action
    expected = np.array([0, 0, 0, 0])
    assert_allclose(policy.draw(x).data, expected)

    # Run multi class bandify chain
    np.random.seed(42)
    for _ in range(100):
        _, a, log_p, r = mcb(x, y)

    # Drawing at this point should be perfect
    expected = np.array([2, 1, 0, 3])
    assert_allclose(policy.draw(x).data, expected)
