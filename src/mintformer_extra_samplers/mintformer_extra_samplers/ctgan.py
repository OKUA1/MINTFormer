from mintformer_prep import BaseMintSampler
import numpy as np
from typing import Optional, List, Tuple
from sdv.tabular import CTGAN
import pandas as pd

class CTGANSampler(BaseMintSampler):
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
        model = CTGAN()
        df = pd.DataFrame(train_subset, columns = [f'Column_{i}' for i in range(train_subset.shape[1])])
        model.fit(df)
        gen_data = model.sample(num_rows=mem_size).to_numpy()
        mem_ind = np.full(shape = (mem_size,), fill_value=-1)
        return gen_data, mem_ind, True