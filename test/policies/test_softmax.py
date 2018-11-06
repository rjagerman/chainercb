import numpy as np
from chainer import Variable
from chainer.testing import assert_allclose
from test.policy import setup_softmax_policy


def test_draw():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert draw
    expected = np.array([5, 2, 4, 1, 1, 1, 2, 1, 2, 4, 1, 0, 1, 4, 2, 5, 0, 5,
                         0, 1, 1, 4, 0, 1, 5, 2, 0, 2, 0, 2, 0, 2])
    assert_allclose(policy.draw(x).data, expected)


def test_draw_tau_0_001():
    policy = setup_softmax_policy(tau=0.001)

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert draw
    expected = np.array([5, 2, 5, 1, 2, 5, 1, 5, 2, 2, 2, 2, 1, 1, 5, 2, 1, 2,
                         2, 1, 1, 2, 5, 2, 5, 2, 1, 2, 5, 2, 5, 2])
    assert_allclose(policy.draw(x).data, expected)


def test_draw_tau_10000():
    policy = setup_softmax_policy(tau=10000.0)

    # Generate minibatch of 32 random samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert draw
    expected = np.array([5, 2, 4, 4, 1, 1, 2, 1, 2, 4, 1, 3, 2, 4, 2, 5, 0, 5,
                         3, 4, 3, 4, 0, 1, 1, 1, 4, 3, 0, 3, 4, 2])
    assert_allclose(policy.draw(x).data, expected)


def test_uniform():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert uniform
    expected = np.array([5, 2, 4, 4, 1, 1, 2, 1, 2, 4, 1, 3, 2, 4, 2, 5, 0, 5,
                         3, 4, 3, 4, 0, 1, 1, 1, 4, 3, 0, 3, 4, 2])
    assert_allclose(policy.uniform(x).data, expected)


def test_max():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert max
    expected = np.array([5, 2, 5, 1, 2, 5, 1, 5, 2, 2, 2, 2, 1, 1, 5, 2, 1, 2,
                         2, 1, 1, 2, 5, 2, 5, 2, 1, 2, 5, 2, 5, 2])
    assert_allclose(policy.max(x).data, expected)


def test_nr_actions():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype('float32'))

    # Assert nr actions
    assert_allclose(policy.nr_actions(x).data, np.ones(32) * 6)


def test_propensity():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype(dtype='float32'))
    action = Variable(
        np.random.random_integers(
            0,
            policy.nr_actions(x)[0].data - 1,
            size=32
        ).astype('int32')
    )
    p = policy.propensity(x, action)
    expected = np.array([2.2147602e-01, 2.4666181e-01, 8.0026180e-02,
                         5.2394089e-04, 5.1778451e-02, 2.2216703e-01,
                         2.2792980e-01, 1.3349118e-02, 1.1443234e-01,
                         3.6572900e-03, 1.8359013e-01, 2.6039204e-02,
                         5.2841913e-02, 2.2601378e-01, 1.8996757e-01,
                         1.2003987e-03, 2.4849439e-02, 3.1758940e-01,
                         3.4999129e-01, 2.0063059e-01, 1.2531378e-03,
                         2.3336175e-01, 6.2047362e-02, 5.2259181e-02,
                         3.5261405e-01, 2.0531374e-01, 3.7587512e-02,
                         2.0282875e-01, 1.9408551e-03, 2.0037284e-04,
                         8.5751503e-04, 3.7217548e-04])
    assert_allclose(p.data, expected)


def test_log_propensity():
    policy = setup_softmax_policy()

    # Generate minibatch of 32 identical samples
    np.random.seed(42)
    x = Variable(np.random.random((32, 3)).astype(dtype='float32'))
    action = Variable(
        np.random.random_integers(
            0,
            policy.nr_actions(x)[0].data - 1,
            size=32
        ).astype('int32')
    )
    p = policy.log_propensity(x, action)
    expected = np.array([-1.5074409, -1.399737,  -2.5254014, -7.554132,
                         -2.960781,  -1.5043257, -1.4787177, -4.316305,
                         -2.1677716, -5.611033,  -1.6950495, -3.6481519,
                         -2.9404507, -1.4871595, -1.6609019, -6.7251015,
                         -3.69492,   -1.146996,  -1.0498471, -1.6062899,
                         -6.6821046, -1.4551655, -2.7798572, -2.9515395,
                         -1.0423813, -1.583216,  -3.2810836, -1.5953934,
                         -6.2446265, -8.515331,  -7.061472,  -7.896145])
    assert_allclose(p.data, expected)
