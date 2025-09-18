def to_storage_idx(tensor_idx: tuple, stride: tuple, offset: int):
    """
    Given the indices of an element within a tensor, the function returns the index of
    the element within the storage of the tensor.

    Args:
        tensor_idx (tuple): The index of the element within the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.

    Returns:
        int: The index of the element within the storage of the tensor.

    """
    storage_idx = 0

    for i in range(len(tensor_idx)):
        storage_idx += tensor_idx[i] * stride[i]

    return storage_idx + offset


def to_tensor_idx(storage_idx: int, shape: tuple):
    """
    Given the indices of an element within the storage of a tensor, the function returns the index of
    the element within the tensor based on contiguous layout, regardless of the stride.

    Args:
        storage_idx (int): The index of the element within the storage of the tensor.
        shape (tuple): The shape of the tensor.

    Returns:
        tuple: The index of the element within the tensor based on contiguous layout.

    """
    tensor_idx = [0] * len(shape)

    for i in range(len(shape) - 1, -1, -1):
        tensor_idx[i] = storage_idx % shape[i]
        storage_idx //= shape[i]

    return tuple(tensor_idx)


def from_broadcast_idx(broadcast_idx: tuple, broadcast_shape: tuple, shape: tuple):
    """
    Given the index of an element within a broadcast tensor, this function returns its index within
    the original tensor before broadcasting, which means no new copies need to be made to address the
    distinction in shape, less memory will be used, more efficient.

    Args:
        broadcast_idx: The index of the element within the broadcast tensor.
        broadcast_shape: The shape of the broadcast tensor.
        shape: The shape of the original tensor before broadcasting.

    Returns:
        tuple: The index of the element within the original tensor before broadcasting.

    """
    index = [0] * len(shape)
    for i in range(-1, -len(shape) - 1, -1):
        if shape[i] == 1:
            index[i] = 0
        else:
            index[i] = broadcast_idx[i]
    return tuple(index)