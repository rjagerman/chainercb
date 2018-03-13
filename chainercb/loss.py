import chainer.functions as F


def ips(observations, actions, propensities, rewards, policy, λ=0.0, clip=None,
        reduce='mean'):
    """
    This is the lambda-translated IPS loss as described in Joachims et al
    (2018), Deep Learning with Logged Bandit Feedback.

    :param observations: The observations for the current batch
    :type observations: chainer.Variable

    :param actions: The actions that were performed by the logging policy
    :type actions: chainer.Variable

    :param propensities: The logging policy propensity scores for those actions
    :type propensities: chainer.Variable

    :param rewards: The rewards (in [0, 1]) that were observed for the chosen
                    actions
    :type rewards: chainer.Variable

    :param policy: The policy we wish to optimize
    :type chainercb.policy.Policy

    :param λ: The λ value for translating the loss
    :type λ: float

    :param clip: The clipping value for propensity scores
    :type clip: float|None

    :param reduce: How to reduce the loss: 'no' means this function returns the
                   per-element loss whereas 'mean' lets this function return the
                   mean of the loss over the entire minibatch
    :type reduce: str

    :return: The loss
    :rtype: chainer.Variable
    """

    # Compute the propensity scores of the policy we wish to optimize (this is
    # capable of backprop)
    policy_propensity = policy.propensity(observations, actions)

    # Compute the λ-translated loss from the rewards
    loss = (1.0 - rewards) - λ

    # Clip logged propensities for numerical stability
    if clip is not None:
        clipped_propensities = F.maximum(propensities.data, 1.0 / clip)
        inverse_propensities = F.minimum(1.0 / clipped_propensities, clip)
    else:
        inverse_propensities = 1.0 / propensities.data

    # Compute the per-element loss
    weight = loss * inverse_propensities
    element_loss = policy_propensity * weight.data

    # Return the loss as a mean or element-wise
    return _reduce_loss(element_loss, reduce)


def policy_gradient(observations, actions, rewards, policy, reduce='mean'):
    """
    This loss is the simple policy-gradient loss

    :param observations: The observations for the current batch
    :type observations: chainer.Variable

    :param actions: The actions that were performed by the policy
    :type actions: chainer.Variable

    :param rewards: The rewards (in [0, 1]) that were observed for the chosen
                    actions
    :type rewards: chainer.Variable

    :param policy: The policy we wish to optimize
    :type chainercb.policy.Policy

    :param reduce: How to reduce the loss: 'no' means this function returns the
                   per-element loss whereas 'mean' lets this function return the
                   mean of the loss over the entire minibatch
    :type reduce: str

    :return: The loss
    :rtype: chainer.Variable
    """

    # Compute the log propensity scores of the policy we wish to optimize (this
    # is capable of backprop)
    policy_log_propensity = policy.log_propensity(observations, actions)

    # Compute the per-element loss
    element_loss = -policy_log_propensity * rewards

    # Return the loss as a mean or element-wise
    return _reduce_loss(element_loss, reduce)


def _reduce_loss(element_loss, reduce):
    """
    Reduces the loss in the per-element loss either by 'no' reduction or by
    'mean' reduction

    :param element_loss: The loss for each element in the mini batch
    :type element_loss: chainer.Variable

    :param reduce: The reduce operation to use
    :type reduce: str

    :return: The reduced loss
    :rtype: chainer.Variable
    """
    if reduce == 'mean':
        return F.mean(element_loss)
    elif reduce == 'no':
        return element_loss
    else:
        raise ValueError(f"only 'mean' and 'no' are valid for 'reduce', but "
                         f"'{reduce}' is given'")
