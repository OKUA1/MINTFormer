"""
Numba compiled function to be used by dataloader
See dataloader for exact description of the functions

"""
from typing import Tuple, Dict, List, Iterator
from mintformer_prep.utils import numba_wrapper, dict_type
import numpy as np

@numba_wrapper
def _add_noise_num(slice_: np.ndarray) -> np.ndarray:
    val = np.random.normal(0, 1.0)
    slice_[0] = val
    return slice_

@numba_wrapper
def _add_noise_cat(slice_: np.ndarray) -> np.ndarray:
    n_cat = len(slice_) - 2
    slice_[0:-1] = 0
    i = np.random.randint(0, n_cat)
    slice_[i] = 1
    return slice_

@numba_wrapper
def _gen_noise_mask(size: int, prob: float, mask: np.ndarray) -> np.ndarray:
    mask_pos = mask == 1
    n_masks = len(mask[mask_pos])
    noise_mask = np.zeros(size)
    noise = np.random.binomial(1, prob, size=n_masks)
    noise_mask[mask_pos] = noise
    return noise_mask

@numba_wrapper
def _check_mask(slice_: np.ndarray) -> bool:
    return slice_[-1] == 1.0

@numba_wrapper
def _apply_mask(
        self_metadata: dict_type,
        self_target_ind: List, 
        self_splits: List,
        batch: np.ndarray,
        feature_mask_prob: float = 0.15,
        feature_noise_prob: float = 0.1,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    noise_masks = []
    batch_size = len(batch)
    for i in range(self_metadata["n_features"]):
        is_target = i in self_target_ind
        # do not apply mask if p = 0
        if (not is_target and feature_mask_prob <= 0.0) or (
            is_target and target_mask_prob <= 0.0
        ):
            masks.append(np.asarray([0 for _ in range(batch_size)]))
            continue
        # check if feature is cat
        is_cat = i in self_metadata["cat_features_ind"]
        # select probabilities corresponding to features/targets
        if not is_target:
            mask_p, noise_p = feature_mask_prob, feature_noise_prob
        else:
            mask_p, noise_p = target_mask_prob, target_noise_prob
        # make slice of the feature
        feature = batch[:, self_splits[i][0] : self_splits[i][1]]
        # generate mask
        mask = np.random.binomial(1, mask_p, size=batch_size)
        # generate noise mask (onnly at the positions where mask = 1)
        noise_mask = _gen_noise_mask(batch_size, noise_p, mask)
        # calculate final mask where noise mask is != 1
        mask = mask - noise_mask
        for j in range(batch_size):
            feature_row = feature[j]
            # if mask was already applied before (e.g. missing value) -> do not use for loss
            if _check_mask(feature_row):
                feature_row[0:-1] = 0.0
                mask[j] = 0
                noise_mask[j] = 0
            else:
                # apply mask
                if mask[j] == 1:
                    feature_row[0:-1] = 0.0
                    feature_row[-1] = 1
                # add noise
                elif noise_mask[j] == 1:
                    if is_cat:
                        feature_row = _add_noise_cat(feature_row)
                    else:
                        feature_row = _add_noise_num(feature_row)
            feature[j] = feature_row
        batch[:, self_splits[i][0] : self_splits[i][1]] = feature
        masks.append(mask)
        noise_masks.append(noise_mask)
    final_mask = np.stack(masks, axis=1)
    final_noise_mask = np.stack(noise_masks, axis=1)
    return batch, final_mask, final_noise_mask

@numba_wrapper
def _generate(
        self_memory_ind: List,
        self_splits_no_tokens: List,
        self_metadata: dict_type,
        self_target_ind: List, 
        self_splits: List,
        batch_size: int,
        data: np.ndarray,
        feature_mask_prob: float = 0.15,
        feature_noise_prob: float = 0.1,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
        gen_mem_mask: bool = True,
    ) -> Iterator:
    head = 0
    max_ = len(data) - 1
    tail = head + batch_size
    while True:
        break_ = False
        if tail >= max_:
            tail = max_
            break_ = True

        if gen_mem_mask:
            data_ind = np.expand_dims(np.arange(head, tail), 1)
            mem_mask = ~(data_ind == self_memory_ind)
            mem_mask = np.expand_dims(mem_mask, 0).astype(np.int8)
        else:
            mem_mask = None

        masked_data, mask, noise_mask = _apply_mask(
            self_metadata,
            self_target_ind,
            self_splits_no_tokens,
            data[head:tail].copy(),
            feature_mask_prob=feature_mask_prob,
            feature_noise_prob=feature_noise_prob,
            target_mask_prob=target_mask_prob,
            target_noise_prob=target_noise_prob,
        )
        yield (
            data[head:tail],
            masked_data,
            mask,
            noise_mask,
            mem_mask,
            self_splits,
            self_splits_no_tokens,
        )

        head = tail
        tail = tail + batch_size
        if break_:
            break