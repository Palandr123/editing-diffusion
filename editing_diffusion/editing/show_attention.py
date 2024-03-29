from collections import defaultdict

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline


def search_sequence_numpy(arr: np.ndarray[np.int64], seq: np.ndarray[np.int64]) -> np.ndarray[np.int64]:
    """
    Search for the occurrence of a sequence within a numpy array.

    Parameters:
    arr: np.ndarray[np.int64] - the input array in which the sequence will be searched
    seq: np.ndarray[np.int64] - the sequence to search for within the input array

    Returns:
    np.ndarray[np.int64] - a numpy array of indices where the sequence is found in the input array.
              If no match is found, an empty list is returned.
    """
    arr_length, seq_length = arr.size, seq.size

    # Create an array of indices for the sequence.
    seq_indices = np.arange(seq_length)

    # Compare the elements in 'arr' with 'seq' to find matching subsequences.
    is_match = (arr[np.arange(arr_length - seq_length + 1)[:, None] + seq_indices] == seq).all(1)
    
    # If any match is found, return the indices where the sequence is found.
    if is_match.any():
        # Use convolution to find the starting indices of matching sequences.
        return np.where(np.convolve(is_match, np.ones(seq_length, dtype=int) > 0))[0]
    return np.array([], dtype=np.int64)  # No match found


def get_aux(model: StableDiffusionXLPipeline, apply_tree_map: bool = True, transpose: bool = True) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get auxiliary data.
        This method iterates through named modules in the UNet and collects auxiliary data from them.
        The `apply_tree_map` and `transpose` parameters control the operations applied to the data.
        If `transpose` is True, it transposes the data. If `apply_tree_map` is True, it applies tree_map to the data.

        Args:
            apply_tree_map: bool - whether to apply tree_map. Default is True.
            transpose: bool - whether to transpose the data. Default is True.

        Returns:
            dict[str, dict[str, torch.Tensor]] - the auxiliary data.
        """
        auxiliary_data = defaultdict(dict)
        for name, aux_module in model.unet.named_modules():
            if hasattr(aux_module, '_aux'):
                module_auxiliary = aux_module._aux
                if transpose:
                    for key, value in module_auxiliary.items():
                        if apply_tree_map:
                            value = torch.utils._pytree.tree_map(
                                lambda vv: vv.chunk(2)[1] if vv is not None else None,
                                value,
                            )
                        auxiliary_data[key][name] = value
                else:
                    auxiliary_data[name] = module_auxiliary
                    if apply_tree_map:
                        auxiliary_data[name] = {
                            k: torch.utils._pytree.tree_map(
                                lambda vv: vv.chunk(2)[1] if vv is not None else None, v
                            )
                            for k, v in auxiliary_data[name].items()
                        }
        return auxiliary_data
