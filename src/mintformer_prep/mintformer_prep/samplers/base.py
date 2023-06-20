import numpy as np
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class BaseMintSampler(ABC):
    @abstractmethod
    def sample(
        self,
        mem_size: int,
        train_subset: np.ndarray,
        train_subset_prepared: np.ndarray,
        target_ind: np.ndarray,
        strat_column: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        pass
