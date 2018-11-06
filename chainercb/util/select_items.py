from chainer import cuda
import chainer.functions as F


def select_items_per_row(values2d, idx2d):
    """
    Selects items from 2-dimensional tensors (matrices) per row by given indices

    :param values2d: The values to choose from
    :type values2d: chainer.Variable

    :param idx2d: The indices to select
    :type idx2d: chainer.Variable

    :return: A matrix with the same shape as idx2d
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(values2d, idx2d)
    rows = idx2d.shape[0]
    cols_idx = idx2d.shape[1]
    cols_values = values2d.shape[1]

    flattened_idx = xp.ravel(idx2d.data) + xp.repeat(
        xp.arange(0, rows * cols_values, cols_values), cols_idx)

    flattened_values = F.flatten(values2d)
    flattened_values = flattened_values[flattened_idx]
    return F.reshape(flattened_values, idx2d.shape)


def inverse_select_items_per_row(values2d, idx2d):
    """
    Selects items from 2-dimensional tensor (matrix) per row that are not in the
    given indices

    :param values2d: The values to choose from
    :type values2d: chainer.Variable

    :param idx2d: The indices to select
    :type idx2d: chainer.Variable

    :return: A matrix with the other elements selected
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(values2d, idx2d)
    rows = idx2d.shape[0]
    cols_idx = idx2d.shape[1]
    cols_values = values2d.shape[1]

    flattened_idx = xp.ravel(idx2d.data) + xp.repeat(
        xp.arange(0, rows * cols_values, cols_values), cols_idx)

    all_idx = xp.arange(0, rows * cols_values, dtype=idx2d.data.dtype)
    all_idx[flattened_idx.data] = -1
    remaining_idx = all_idx[all_idx >= 0]

    flattened_values = F.flatten(values2d)
    flattened_values = flattened_values[remaining_idx]
    return F.reshape(flattened_values, (rows, cols_values - cols_idx))
