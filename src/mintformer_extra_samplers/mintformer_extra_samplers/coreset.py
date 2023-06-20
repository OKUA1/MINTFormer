import numpy as np
from mintformer_prep import BaseMintSampler
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import torch


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def euclidean_dist_pair(x):
    m = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)
    dist = xx + xx.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def euclidean_dist_np(x, y):
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    return np.sqrt(np.clip(x2 + y2 - 2. * xy, 1e-12, None))

def euclidean_dist_pair_np(x):
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))


def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 2000):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class CoreSampler(BaseMintSampler):
    def __init__(self, emb_size = 64) -> None:
        """
        Class to be used for performing of memory random sampling.

        Parameters
        ----------
        emb_size : int
            defines the size of the projected space
        """
        self.emb_size = emb_size
        self.transformer = None

    def inverse_transform(self, core_set):
        if self.transformer is None: 
            return core_set
        return self.transformer.inverse_transform(core_set)

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

        projected = train_subset_prepared
        if train_subset_prepared.shape[-1] > self.emb_size: 
            self.transformer = TruncatedSVD(n_components = self.emb_size)
            projected = self.transformer.fit_transform(train_subset_prepared)
            print(projected.shape)
        
        core_set_ind = k_center_greedy(projected.astype(np.float32), mem_size, euclidean_dist, 'cpu', already_selected = []).tolist()
        
        return train_subset_prepared[core_set_ind], core_set_ind, False
