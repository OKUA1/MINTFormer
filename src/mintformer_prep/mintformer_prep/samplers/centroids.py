import numpy as np
from typing import Optional, Tuple, List
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from mintformer_prep.samplers.base import BaseMintSampler

class NaiveClusterCentroidsSampler(BaseMintSampler):
    def sample(
        self,
        mem_size: int,
        train_subset: np.ndarray,
        train_subset_prepared: np.ndarray,
        target_ind: List[int],
        strat_column: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Performs subsampling by fitting a K-Means model and using cluster centers as samples.
        This sampler operates on the prepared data and might produce invalid samples.

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
        
        perform_dim_reduction = False
        X = train_subset_prepared
        if train_subset_prepared.shape[-1] > 64:
            perform_dim_reduction = True
        
        if perform_dim_reduction:
            svd = TruncatedSVD(n_components=64, random_state=42)
            X = svd.fit_transform(X)
        
        km = KMeans(n_clusters = mem_size, random_state=42)
        km.fit(X)
        centroids = km.cluster_centers_

        if perform_dim_reduction:
            centroids = svd.inverse_transform(centroids)

        return centroids, [-1 for _ in range(mem_size)], False