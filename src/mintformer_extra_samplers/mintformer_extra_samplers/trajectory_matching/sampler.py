import numpy as np
from mintformer_prep import BaseMintSampler
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import torch
from .expert_training import train_experts
from .distill import distill
import os
import re


def count_files(path):
    return len(
        [
            name
            for name in os.listdir(path)
            if os.path.isfile(os.path.join(path, name)) and re.match("^\d+$", name)
        ]
    )


def count_subfolders(path):
    return len(
        [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    )


class TrajectoryMatchSampler(BaseMintSampler):
    def __init__(
        self,
        n_experts=100,
        expert_batch_size=256,
        expert_epochs=1000,
        n_syn_steps=50,
        student_batch_size=256,
        range_mult=3,
        n_syn_iterations=100000,
        expert_path=None,
        distilled_path = None,
    ) -> None:
        """
        Parameters
        ----------
        n_experts : int
            number of experts to train
        expert_batch_size : int
            batch size for expert training
        expert_epochs : int
            number of epochs for expert training
        n_syn_steps : int
            number of synthetic steps
        student_batch_size : int
            batch size for student training
        range_mult : int
            coefficient for long range matching
        n_syn_iterations : int
            number of synthetic iterations
        expert_path : Optional[str]
            path to pretrained experts (optional)
        distilled_path : Optional[str]
            path to distilled dataset (optional)

        Returns
        -------
        None
        """
        self.n_experts = n_experts
        self.expert_batch_size = expert_batch_size
        self.expert_epochs = expert_epochs
        self.n_syn_steps = n_syn_steps
        self.student_batch_size = student_batch_size
        self.range_mult = range_mult
        self.n_syn_iterations = n_syn_iterations

        self.expert_path = expert_path
        self.distilled_path = distilled_path  

    def sample(
        self,
        mem_size: int,
        train_subset: np.ndarray,
        train_subset_prepared: np.ndarray,
        target_ind: List[int],
        strat_column: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Performs dataset distillation based on trajectory matching

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
        mem_id = [-1 for _ in range(mem_size)]

        if self.distilled_path:
            distilled = np.load(self.distilled_path)
            return distilled, mem_id, False

        if not self.expert_path:
            expert_path = train_experts(
                train_subset_prepared,
                self.n_experts,
                self.expert_batch_size,
                self.expert_epochs,
            )
        else:
            expert_path = self.expert_path
            self.n_experts = count_subfolders(expert_path)
            self.expert_epochs = count_files(os.path.join(expert_path, "0"))
            print(expert_path, self.n_experts, self.expert_epochs)

        distilled = distill(
            mem_size,
            train_subset_prepared.shape[1],
            self.n_syn_steps,
            self.expert_epochs,
            self.n_experts,
            expert_path,
            self.student_batch_size,
            self.range_mult,
            self.n_syn_iterations,
        )

        np.save("distilled_.npy", distilled)

        column_sum = np.sum(train_subset_prepared, axis=0)
        print(column_sum)
        binary_vector = np.where(column_sum == 0, 0, 1).astype(np.float32)
        print(binary_vector)
        distilled = distilled * binary_vector
        print("distilled", distilled)

        return distilled, mem_id, False