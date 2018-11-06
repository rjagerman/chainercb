import numpy as np
from chainer import as_variable
from chainer.testing import assert_allclose

from chainercb.bandify import MultiClassBandify
from chainercb.policies import ADFUCBPolicy


def test_draw():
    policy = ADFUCBPolicy(6)

    # Generate minibatch of 32 random samples with 4 actions each
    np.random.seed(42)
    x = as_variable(np.random.random((32, 4, 6)).astype(np.float32))

    # Assert draw
    expected = np.array([1, 1, 0, 0, 3, 3, 1, 2, 0, 2, 0, 1, 2, 2, 2, 3, 2, 0,
                         2, 1, 3, 0, 0, 3, 0, 2, 1, 0, 1, 3, 0, 0])
    assert_allclose(policy.draw(x).data, expected)


def test_update():
    policy = ADFUCBPolicy(3)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[[ 2.0, 1.0,   3.0], [-2.0, 4.0, -1.0]],
                              [[-3.0, 0.0,  -1.0], [ 1.0, 0.0,  1.0]],
                              [[-1.0, 0.01, -1.0], [ 0.1, 2.0,  2.0]]]))
    y = as_variable(np.array([0, 1, 1]))

    # Drawing at this point should return the default action
    expected = np.array([1, 0, 1])
    assert_allclose(policy.draw(x).data, expected)

    # Perfect update
    log_p = as_variable(np.zeros(y.shape))
    r = as_variable(np.ones((y.shape[0], 1)))
    for _ in range(100):
        a = as_variable(np.random.randint(2, size=y.shape))
        r = (1.0 * (a.data == y.data))
        policy.update(x, a, log_p, as_variable(r))

    # Drawing at this point should be perfect
    expected = np.array([0, 1, 1])
    assert_allclose(policy.draw(x).data, expected)


def test_bandify_update():
    policy = ADFUCBPolicy(3)
    mcb = MultiClassBandify(policy)
    mcb.update_policy(policy)

    # Generate minibatch
    np.random.seed(42)
    x = as_variable(np.array([[[2.0, 1.0, 3.0], [-2.0, 4.0, -1.0]],
                              [[-3.0, 0.0, -1.0], [1.0, 0.0, 1.0]],
                              [[-1.0, 0.01, -1.0], [0.1, 2.0, 2.0]]]))
    y = as_variable(np.array([0, 1, 1], dtype=np.int32))

    # Drawing at this point should return the default action
    expected = np.array([1, 0, 1])
    assert_allclose(policy.draw(x).data, expected)

    # Run multi class bandify chain
    np.random.seed(42)
    for _ in range(100):
        _, a, log_p, r = mcb(x, y)

    # Drawing at this point should be perfect
    expected = np.array([0, 1, 1])
    assert_allclose(policy.draw(x).data, expected)
