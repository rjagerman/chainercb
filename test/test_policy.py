from chainer import Variable
from chainercb.policy import Policy
from nose.tools import raises


@raises(NotImplementedError)
def test_no_impl_draw():
    policy = Policy()
    x = Variable()
    policy.draw(x)


@raises(NotImplementedError)
def test_no_impl_max():
    policy = Policy()
    x = Variable()
    policy.max(x)


@raises(NotImplementedError)
def test_no_impl_uniform():
    policy = Policy()
    x = Variable()
    policy.uniform(x)


@raises(NotImplementedError)
def test_no_impl_nr_actions():
    policy = Policy()
    x = Variable()
    policy.nr_actions(x)


@raises(NotImplementedError)
def test_no_impl_propensity():
    policy = Policy()
    x = Variable()
    a = Variable()
    policy.propensity(x, a)


@raises(NotImplementedError)
def test_no_impl_log_propensity():
    policy = Policy()
    x = Variable()
    a = Variable()
    policy.log_propensity(x, a)
