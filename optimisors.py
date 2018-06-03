import numpy as np


def generate_batch_data(data, labels, batch_size):
    batch_indices = np.random.choice(np.arange(data.shape[0]),  # or simply X.shape[0]
                                     batch_size,
                                     replace=False)
    return data[batch_indices, :], labels[batch_indices]


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


