import numpy as np
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split
from mintformer_prep.samplers.base import BaseMintSampler

class RandomSampler(BaseMintSampler):
    def __init__(self, stratify_if_possible: bool = True) -> None:
        """
        Class to be used for performing of memory random sampling.

        Parameters
        ----------
        stratify_if_possible : bool, optional
            defines whether stratified sampling should be used whenever possible, by default True
        """
        self.stratify_if_possible = stratify_if_possible

    def sample(
        self,
        mem_size: int,
        train_subset: np.ndarray,
        train_subset_prepared: np.ndarray,
        target_ind: List[int],
        strat_column: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Performs random (stratified) sampling of the dataset

        Parameters
        ----------
        mem_size : int
            number of samples to be sampled
        train_subset : np.ndarray
            raw train data
        train_subset_prepared : np.ndarray
            prepared train data
        target_ind : List[int]
            list of target indices (not used, only for compatibility)
        strat_column : Optional[int]
            column to be used for stratification

        Returns
        -------
        Tuple[np.ndarray, List[int], bool]
            selected samples, indices of selected samples, requires_transformation = False
        """
        ind = [i for i in range(len(train_subset_prepared))]
        if self.stratify_if_possible and isinstance(strat_column, int):
            stratify = train_subset[:, strat_column]
        else: 
            stratify = None
        
        selected_ind, _ = train_test_split(ind, train_size=mem_size, random_state=42, stratify=stratify)
        return train_subset_prepared[selected_ind, :], selected_ind, False
