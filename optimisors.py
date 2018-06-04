import numpy as np
from manipulate_data import generate_batch_data


def SDG(data, labels, init_params, loss_func=None,
        batch_size=0, learning_rate=0.1):
    """
    loss_func(data, labels, params) -> loss, d_params
    """
    params = init_params
    if batch_size > 0:
        batch_data, batch_labels = generate_batch_data(data, labels, batch_size)
        loss, d_params = loss_func(batch_data, batch_labels, params)
    else:
        loss, d_params = loss_func(data, labels, params)
    params -= d_params * learning_rate
    return loss, params


