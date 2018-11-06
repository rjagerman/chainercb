import numpy as np
from chainer import Variable
from chainer.testing import assert_allclose
from chainercb.policies import Explore
from test.policy import setup_softmax_policy


def test_draw():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert draw
    expected = np.array([5, 2, 4, 4, 1, 1, 2, 1, 2, 4, 1, 3, 2, 4, 2, 5, 0, 5,
                         3, 4, 3, 4, 0, 1, 1, 1, 4, 3, 0, 3, 4, 2])
    assert_allclose(epsilon_greedy.draw(x).data, expected)


def test_uniform():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert uniform
    expected = np.array([5, 2, 4, 4, 1, 1, 2, 1, 2, 4, 1, 3, 2, 4, 2, 5, 0, 5,
                         3, 4, 3, 4, 0, 1, 1, 1, 4, 3, 0, 3, 4, 2])
    assert_allclose(epsilon_greedy.uniform(x).data, expected)


def test_max():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert max
    expected = np.array([5, 2, 5, 1, 2, 5, 1, 5, 2, 2, 2, 2, 1, 1, 5, 2, 1, 2,
                         2, 1, 1, 2, 5, 2, 5, 2, 1, 2, 5, 2, 5, 2])
    assert_allclose(epsilon_greedy.max(x).data, expected)


def test_nr_actions():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert nr actions
    assert_allclose(epsilon_greedy.nr_actions(x).data, np.ones(32) * 6)


def test_propensity():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype(dtype='float32'))
    action = Variable(
        np.random.random_integers(
            0,
            epsilon_greedy.nr_actions(x)[0].data - 1,
            size=32
        ).astype('int32')
    )
    p = epsilon_greedy.propensity(x, action)
    expected = np.array([0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667, 0.166667, 0.166667, 0.166667,
                         0.166667, 0.166667])
    assert_allclose(p.data, expected)


def test_log_propensity():
    policy = setup_softmax_policy()
    epsilon_greedy = Explore(policy)

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype(dtype='float32'))
    action = Variable(
        np.random.random_integers(
            0,
            epsilon_greedy.nr_actions(x)[0].data - 1,
            size=32
        ).astype('int32')
    )
    p = epsilon_greedy.log_propensity(x, action)
    expected = np.array([-1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759, -1.791759, -1.791759, -1.791759,
                         -1.791759, -1.791759])
    assert_allclose(p.data, expected)
