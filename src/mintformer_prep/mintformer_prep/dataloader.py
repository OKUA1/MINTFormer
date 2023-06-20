import numpy as np
from copy import deepcopy
from typing import List, Tuple, Iterator
from mintformer_prep.dataset import MintDataset


Split = Tuple[int, int]


class MintDataLoader:
    """
    A reference dataloader without dependencies on DL frameworks.
    This dataloader is not optimized for performance. However, it should not be a bottleneck for most use cases.
    """

    def __init__(self, dataset: MintDataset):
        """
        Parameters
        ----------
        dataset : MintDataset
            dataset instance to be used with the data loader
        """
        self.target_ind = dataset.target_ind
        self._dataset = dataset
        data = dataset.get_data()
        self.train = data[0]
        self.val = data[1]
        self.test = data[2]
        self.memory = data[3]
        self.memory_ind = np.expand_dims(data[4], axis=0)
        self.metadata = data[5]
        self.splits = self._gen_splits_tuples(
            self.metadata["splits_ind"], self.metadata["total_size_with_tokens"]
        )
        self.splits_no_tokens = self._gen_splits_tuples(
            self.metadata["splits_ind_no_tokens"], self.metadata["total_size_no_tokens"]
        )

    def _gen_splits_tuples(self, splits_ind: List[int], total_size: int) -> List[Split]:
        """
        Generates indices in a form [(start, end)] from [start]

        Parameters
        ----------
        splits_ind : List[int]
            starting indices of each column
        total_size : int
            total number of features

        Returns
        -------
        List[Split]
            splits
        """
        ind = 0
        splits = []
        for s in splits_ind:
            splits.append((ind, s))
            ind = s
        splits.append((ind, total_size))
        return splits

    def _add_noise_cat(self, slice_: np.ndarray) -> np.ndarray:
        """
        Sets a random category in one-hot-encoded vector to one and everythin else to 0

        Parameters
        ----------
        slice_ : np.ndarray
            feature in the form [one-hot-encoded, mask_token]

        Returns
        -------
        np.ndarray
            randomized feature
        """
        n_cat = len(slice_) - 2
        slice_[0:-1] = 0
        i = np.random.randint(0, n_cat)
        slice_[i] = 1
        return slice_

    def _add_noise_num(self, slice_: np.ndarray) -> np.ndarray:
        """
        Sets a value of a numerical feature to a random one drawn from normal distribution (0,1)

        Parameters
        ----------
        slice_ : np.ndarray
            feature in the form [value, mask token]

        Returns
        -------
        np.ndarray
            randomized feature
        """
        val = np.random.normal(0, 1.0)
        slice_[0] = val
        return slice_

    def _check_mask(self, slice_: np.ndarray) -> bool:
        """
        Checks whether mask token is set to 1 for a given feature

        Parameters
        ----------
        slice_ : np.ndarray
            feature in a form [feature, mask_token]

        Returns
        -------
        bool
            True if mask token is set to 1 else False
        """
        return slice_[-1] == 1.0

    def _gen_noise_mask(self, size: int, prob: float, mask: np.ndarray) -> np.ndarray:
        """
        Generates a noise mask on top of an existing mask.
        Noise mask can only be 1 where the original mask was also set to 1.

        Parameters
        ----------
        size : int
            size of the mask (usually equal to the size of the current batch)
        prob : float
            probability of setting noise_mask = 1 if mask == 1
            entries with mask == 0 can never become noise_mask = 1
        mask : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        mask_pos = mask == 1
        n_masks = len(mask[mask_pos])
        noise_mask = np.zeros(size)
        noise = np.random.binomial(1, prob, size=n_masks)
        noise_mask[mask_pos] = noise
        return noise_mask

    def _apply_mask(
        self,
        batch: np.ndarray,
        feature_mask_prob: float = 0.15,
        feature_noise_prob: float = 0.1,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates and applies mask according to provided probabilities

        Parameters
        ----------
        batch : np.ndarray
            batch to apply the mask on
        feature_mask_prob : float, optional
            probability of masking the feature, by default 0.15
        feature_noise_prob : float, optional
            probability of randomizing the (masked) feature, by default 0.1
        target_mask_prob : float, optional
            probability of masking the target, by default 1.0
        target_noise_prob : float, optional
            probability of randomizing the (masked) target, by default 0.0

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            masked batch, mask, noise mask
        """
        masks = []
        noise_masks = []
        batch_size = len(batch)
        for i in range(self.metadata["n_features"]):
            is_target = i in self.target_ind
            # do not apply mask if p = 0
            if (not is_target and feature_mask_prob <= 0.0) or (
                is_target and target_mask_prob <= 0.0
            ):
                masks.append(np.asarray([0 for _ in range(batch_size)]))
                continue
            # check if feature is cat
            is_cat = i in self.metadata["cat_features_ind"]
            # select probabilities corresponding to features/targets
            if not is_target:
                mask_p, noise_p = feature_mask_prob, feature_noise_prob
            else:
                mask_p, noise_p = target_mask_prob, target_noise_prob
            # make slice of the feature
            feature = batch[:, self.splits[i][0] : self.splits[i][1]]
            # generate mask
            mask = np.random.binomial(1, mask_p, size=batch_size)
            # generate noise mask (onnly at the positions where mask = 1)
            noise_mask = self._gen_noise_mask(batch_size, noise_p, mask)
            # calculate final mask where noise mask is != 1
            mask = mask - noise_mask
            for j in range(batch_size):
                feature_row = feature[j]
                # if mask was already applied before (e.g. missing value) -> do not use for loss
                if self._check_mask(feature_row):
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
                            feature_row = self._add_noise_cat(feature_row)
                        else:
                            feature_row = self._add_noise_num(feature_row)
                feature[j] = feature_row
            batch[:, self.splits[i][0] : self.splits[i][1]] = feature
            masks.append(mask)
            noise_masks.append(noise_mask)
        final_mask = np.stack(masks, axis=1)
        final_noise_mask = np.stack(noise_masks, axis=1)
        return batch, final_mask, final_noise_mask

    def _generate(
        self,
        batch_size: int,
        data: np.ndarray,
        feature_mask_prob: float = 0.15,
        feature_noise_prob: float = 0.1,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
        gen_mem_mask: bool = True,
    ) -> Iterator:
        """
        Generates data

        Parameters
        ----------
        batch_size : int
            batch size
        data : np.ndarray
            dataset
        feature_mask_prob : float, optional
            probability of masking the feature, by default 0.15
        feature_noise_prob : float, optional
            probability of randomizing the (masked) feature, by default 0.1
        target_mask_prob : float, optional
            probability of masking the target, by default 1.0
        target_noise_prob : float, optional
            probability of randomizing the (masked) target, by default 0.0
        gen_mem_mask : bool, optional
            defines whether a memory masked should be generated, by default True
            memory mask is used as additional input during training to prevent direct lookups

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[Split], List[Split]]]
            <unmasked data>, <masked data>, <mask>, <noise mask>, <memory mask>, <splits (with tokens)>, <splits (no tokens)>
        """
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
                mem_mask = ~(data_ind == self.memory_ind)
                mem_mask = np.expand_dims(mem_mask, 0).astype(np.int8)
            else:
                mem_mask = None

            masked_data, mask, noise_mask = self._apply_mask(
                deepcopy(data[head:tail]),
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
                self.splits,
                self.splits_no_tokens,
            )

            head = tail
            tail = tail + batch_size
            if break_:
                break

    def generate_train(
        self,
        batch_size: int,
        feature_mask_prob: float = 0.15,
        feature_noise_prob: float = 0.1,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
        gen_mem_mask: bool = True,
    ) -> Iterator:
        """
        Train loader

        Parameters
        ----------
        batch_size : int
            batch size
        feature_mask_prob : float, optional
            probability of masking the feature, by default 0.15
        feature_noise_prob : float, optional
            probability of randomizing the (masked) feature, by default 0.1
        target_mask_prob : float, optional
            probability of masking the target, by default 1.0
        target_noise_prob : float, optional
            probability of randomizing the (masked) target, by default 0.0
        gen_mem_mask : bool, optional
            defines whether a memory masked should be generated, by default True
            memory mask is used as additional input during training to prevent direct lookups

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[Split], List[Split]]]
            <unmasked data>, <masked data>, <mask>, <noise mask>, <memory mask>, <splits (with tokens)>, <splits (no tokens)>
        """
        return self._generate(
            batch_size=batch_size,
            data=self.train,
            feature_mask_prob=feature_mask_prob,
            feature_noise_prob=feature_noise_prob,
            target_mask_prob=target_mask_prob,
            target_noise_prob=target_noise_prob,
            gen_mem_mask=gen_mem_mask,
        )

    def generate_val(
        self,
        batch_size: int,
        feature_mask_prob: float = 0.0,
        feature_noise_prob: float = 0.0,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
    ) -> Iterator:
        """
        Validation loader
        Memory mask is always None since validation samples are never used as memory

        Parameters
        ----------
        batch_size : int
            batch size
        feature_mask_prob : float, optional
            probability of masking the feature, by default 0.15
        feature_noise_prob : float, optional
            probability of randomizing the (masked) feature, by default 0.1
        target_mask_prob : float, optional
            probability of masking the target, by default 1.0
        target_noise_prob : float, optional
            probability of randomizing the (masked) target, by default 0.0

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[Split], List[Split]]]
            <unmasked data>, <masked data>, <mask>, <noise mask>, <memory mask = None>, <splits (with tokens)>, <splits (no tokens)>
        """
        if self.val is None: 
            raise RuntimeError("Validation subset is not available")
        return self._generate(
            batch_size=batch_size,
            data=self.val,
            feature_mask_prob=feature_mask_prob,
            feature_noise_prob=feature_noise_prob,
            target_mask_prob=target_mask_prob,
            target_noise_prob=target_noise_prob,
            gen_mem_mask=False,
        )

    def generate_test(
        self,
        batch_size: int,
        feature_mask_prob: float = 0.0,
        feature_noise_prob: float = 0.0,
        target_mask_prob: float = 1.0,
        target_noise_prob: float = 0.0,
    ) -> Iterator:
        """
        Test loader
        Memory mask is always None since test samples are never used as memory

        Parameters
        ----------
        batch_size : int
            batch size
        feature_mask_prob : float, optional
            probability of masking the feature, by default 0.15
        feature_noise_prob : float, optional
            probability of randomizing the (masked) feature, by default 0.1
        target_mask_prob : float, optional
            probability of masking the target, by default 1.0
        target_noise_prob : float, optional
            probability of randomizing the (masked) target, by default 0.0

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[Split], List[Split]]]
            <unmasked data>, <masked data>, <mask>, <noise mask>, <memory mask = None>, <splits (with tokens)>, <splits (no tokens)>
        """
        if self.test is None: 
            raise ValueError("Test subset is not available")
        return self._generate(
            batch_size=batch_size,
            data=self.test,
            feature_mask_prob=feature_mask_prob,
            feature_noise_prob=feature_noise_prob,
            target_mask_prob=target_mask_prob,
            target_noise_prob=target_noise_prob,
            gen_mem_mask=False,
        )
    
    def n_batches(self, batch_size: int) -> int: 
        """
        Calculates the number of batches for a given batch size.

        Parameters
        ----------
        batch_size : int
            batch size

        Returns
        -------
        int
            number of batches
        """
        if batch_size > len(self.train):
            return 1
        return int(np.ceil(len(self.train)/batch_size))
    
    @property
    def is_test_available(self):
        return self.test is not None
    
    @property
    def is_val_available(self): 
        return self.val is not None


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    np.set_printoptions(linewidth=256)
    N_SAMPLES = 70
    num = np.random.random(size=(N_SAMPLES, 3))
    cat = np.random.randint(0, 3, size=(N_SAMPLES, 2))
    data = np.hstack((num, cat))
    ds = MintDataset(data=data, cat_ind=[3, 4], target_ind=[4], autoprep=True)
    ds.prepare_memory(20)

    dl = MintDataLoader(dataset=ds, targets_ind=[4])

    for i in dl.generate_train(16):
        print("Unmasksed")
        print(i[0])
        print("Masked")
        print(i[1])
        print("Mask")
        print(i[2])
        print("Noise mask")
        print(i[3])
        print("Memory mask")
        print(i[4])
        print("Splits (with tokens)")
        print(i[5])
        print("Splits (no tokens)")
        print(i[6])
        break