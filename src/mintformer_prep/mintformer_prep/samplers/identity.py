from mintformer_prep.samplers.base import BaseMintSampler
import numpy as np
from typing import Optional, Tuple, List


class IdentitySampler(BaseMintSampler):
    def sample(
        self,
        mem_size: int,
        train_subset: np.ndarray,
        train_subset_prepared: np.ndarray,
        target_ind: List[int],
        strat_column: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Returns the whole training set as memory. 
        Most of the arguments (including the size of the memory) are ignored. 


        Parameters
        ----------
        mem_size : int
            this input is ignored
        train_subset : np.ndarray
            this input is ignored
        train_subset_prepared : np.ndarray
            prepared train data
        target_ind : List[int]
            this input is ignored
        strat_column : Optional[int]
            this input is ignored

        Returns
        -------
        Tuple[np.ndarray, List[int], bool]
            selected samples, indices of selected samples, requires_transformation = False
        """

        return train_subset_prepared, [i for i in range(len(train_subset_prepared))], False
