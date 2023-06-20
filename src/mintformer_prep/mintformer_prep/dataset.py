from typing import Union, List, Optional, Dict, Tuple, Iterator, Type, Any
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from mintformer_prep.preprocessor import MintPreprocessor
from mintformer_prep.samplers import BaseMintSampler, RandomSampler


def load_data_util(data: Union[str, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Either reads data from csv file, converts dataframe to np array or does nothing depending on the type of passed argument.

    Parameters
    ----------
    data : Union[str, np.ndarray, pd.DataFrame]
        data to read

    Returns
    -------
    np.ndarray
        data

    Raises
    ------
    ValueError
        Invalid data format
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    if not isinstance(data, np.ndarray):
        raise ValueError("Unsupported data format was encountered.")
    return data


class MintDataset:
    """
    Wrapper for main dataset preparation operations.
    """

    def __init__(
        self,
        data: Union[str, np.ndarray, pd.DataFrame],
        cat_ind: List[int],
        target_ind: List[int],
        autoprep: bool = True,
        preprocessor: Optional[Type] = None,
    ) -> None:
        """
        Initializes the dataset

        Parameters
        ----------
        data : Union[str, np.ndarray, pd.DataFrame]
            full dataset
        cat_ind : List[int]
            indices of categorical features
        target_ind : List[int]
            indices of target column (can be one or several)
        autoprep : bool, optional
            defines whether data has to be loaded and prepared right right away, by default True
            equivalent to manually calling `.load_data()` and `.prepare_data()`
        preprocessor : Optional[Type], optional
            preprocessor class, should be a child of  `MintPreprocessor`;
            if None, `MintPreprocessor` is used, by default None

        """

        self.data = None
        self._data = data
        self.cat_ind = cat_ind
        self.target_ind = target_ind
        self._data_ready, self._mem_ready = False, False
        (
            self.train_subset_prepared,
            self.test_subset_prepared,
            self.val_subset_prepared,
        ) = (None, None, None)
        self._select_strat_column()
        if autoprep:
            self.load_data()
            self.prepare_data()
        if preprocessor and not issubclass(preprocessor, MintPreprocessor):
            raise ValueError(
                "The preprocessor should be a subclass of `MintPreprocessor`"
            )
        self._preprocessor_class = preprocessor if preprocessor else MintPreprocessor

    @classmethod
    def with_cv(
        cls,
        data: Union[str, np.ndarray, pd.DataFrame],
        stratified: bool,
        strat_column_ind: Optional[int],
        cat_ind: List[int],
        target_ind: List[int],
        n_splits: int = 5,
        autoprep: bool = True,
        val_frac: Optional[float] = None,
        preprocessor: Optional[Type] = None,
    ) -> Iterator:
        """
        Splits the data in CV manner and yields a separate dataset instance per fold.

        Parameters
        ----------
        data : Union[str, np.ndarray, pd.DataFrame]
            full dataset
        stratified : bool
            defines whether stratified CV is used
        strat_column_ind : Optional[int]
            column for stratification
        cat_ind : List[int]
            indices of categorical features
        target_ind : List[int]
            indices of target column (can be one or several)
        n_splits : int, optional
            number of CV splits, by default 5
        autoprep : bool, optional
            defines whether data has to be loaded and prepared right right away, by default True
        val_frac : float, optional
            fraction (of train set) to be left for validation, by default 0.1
        preprocessor : Optional[Type], optional
            preprocessor class, should be a child of  `MintPreprocessor`;
            if None, `MintPreprocessor` is used, by default None

        Yields
        ------
        Iterator
            one instance of dataset per fold

        """
        data = load_data_util(data)
        if stratified:
            if not isinstance(strat_column_ind, int):
                raise ValueError("Invalid stratification, column")
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = [
                s
                for s in kf.split(
                    data,
                    data[
                        :,strat_column_ind,
                    ],
                )
            ]
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = [s for s in kf.split(data)]

        for split in splits:
            train_data = data[split[0]]
            test_data = data[split[1]]
            ds = cls(
                data=data,
                cat_ind=cat_ind,
                target_ind=target_ind,
                autoprep=False,
                preprocessor=preprocessor,
            )
            ds.set_subsets(
                train_data, test_data, val_frac=val_frac, shuffle_train=False
            )
            if autoprep:
                ds.load_data()
                ds.prepare_data(split=False, shuffle_train=False)
            yield ds

    def load_data(self) -> None:
        """
        Loads the data. See `load_data_util` for more details.
        """
        data = load_data_util(self._data)
        self.data = data

    def _select_strat_column(self) -> None:
        """
        Picks first target categorical column (or None) to use for stratified train/test split and stores its index.
        """
        self.strat_column = None
        for i in self.target_ind:
            if i in self.cat_ind:
                self.strat_column = i
                break

    def _stratification(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates a vector to be used for stratification.

        Parameters
        ----------
        x : np.ndarray
            Dataset to split

        Returns
        -------
        Optional[np.ndarray]
            Column to use for stratification or None.
        """
        if not self.strat_column:
            return None
        else:
            return x[:, self.strat_column]

    def prepare_data(
        self,
        split: bool = True,
        shuffle_train: bool = True,
        train_size: Optional[float] = 0.75,
        val_frac: Optional[int] = 0.1,
    ):
        """
        Prepares the datset by first [optionally] splitting the dataset.
        Fits the preprocessor on the train set.
        Applies the preprocessor to all subsets.
        Stores the prepared subsets.

        If the split is not performed, unprocessed train/val/test datasets should be set manually.
        See `.set_subsets()`


        Parameters
        ----------
        split : bool, optional
            Defines if train/test split should be used, by default True
        shuffle_train : bool, optional
            Defined whether the train set should be shuffled, by default True
        train_size : Optional[float], optional
            size of the train size, by default 0.75
        val_frac : Optional[int], optional
            a fraction size (of train set) to be used for validation, by default 0.1
        """
        if split:
            train, test = train_test_split(
                self.data,
                train_size=train_size,
                random_state=42,
                stratify=self._stratification(self.data),
            )
            self.set_subsets(
                train=train, test=test, val_frac=val_frac, shuffle_train=shuffle_train
            )
        self._preprocessor = MintPreprocessor(cat_ind=self.cat_ind)
        self.train_subset_prepared = self._preprocessor.fit_transform(self.train_subset)

        if self.val_subset is not None:
            self.val_subset_prepared = self._preprocessor.transform(self.val_subset)
        if self.test_subset is not None:
            self.test_subset_prepared = self._preprocessor.transform(self.test_subset)
        self._data_ready = True

    def get_metadata(self) -> Dict:
        """
        Returns the metadata produced by the preprocessor.
        See the documentation of preprocessor for details.

        Returns
        -------
        Dict
            dataset metadata
        """
        if not self._data_ready:
            raise RuntimeError(
                "Cannot get metadata of non-prepared dataset. Try calling .prepare_data() first."
            )
        return self._preprocessor.get_metadata()

    def prepare_memory(
        self, mem_size: Union[int, float], sampler: Optional[BaseMintSampler] = None
    ) -> None:
        """
        Prepares the memory by applying the subsampling techique.

        Parameters
        ----------
        mem_size : Union[int, float]
            The size of the memory. Either float (fraction) or int (number of samples)
        sampler : Optional[BaseMintSampler], optional
            An instance of sampler to be used; if None, RandomSampler is used, by default None

        """
        if not self._data_ready:
            raise RuntimeError("Data should be prepared before memory.")
        if isinstance(mem_size, float):
            if mem_size >= 1:
                self.memory = self._train_prepared
                self.memory_ind = [i for i in range(len(self.train_subset_prepared))]
                return
            if mem_size <= 0.0:
                raise ValueError(
                    "If mem_size is float, it should be in the range (0, 1]."
                )
            else:
                mem_size = int(len(self.train_subset_prepared) * mem_size)

        if mem_size == 0:
            raise ValueError("The selected mempory size is too small.")
        if mem_size >= len(self.train_subset_prepared) * mem_size and sampler is None:
            self.memory = self._train_prepared
            self.memory_ind = [i for i in range(len(self.train_subset_prepared))]
            return
        else:
            if sampler is None:
                sampler = RandomSampler()
            memory, self.memory_ind, requires_transformation = sampler.sample(
                mem_size,
                self.train_subset,
                self.train_subset_prepared,
                self.target_ind,
                self.strat_column,
            )
            if requires_transformation:
                self.memory = self._preprocessor.transform(memory)
            else: 
                self.memory = memory
        self._mem_ready = True

    def get_data(
        self,
    ) -> Tuple[
        np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[int], Dict
    ]:
        """
        Returns all data required for training.

        Returns
        -------
        Tuple[ np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[int], Dict ]
            <prepared train set>, <prepared validation set or None>, <prepared test set>, <memory>, <memory ind>, <metadata>
        """
        if not self._data_ready or not self._mem_ready:
            raise RuntimeError("Either data of memory was not prepared.")
        return (
            self.train_subset_prepared,
            self.val_subset_prepared,
            self.test_subset_prepared,
            self.memory,
            self.memory_ind,
            self.get_metadata(),
        )
    
    @property
    def is_test_available(self):
        return self.test_subset_prepared is not None
    
    @property
    def is_val_available(self): 
        return self.val_subset_prepared is not None

    def get_memory(self) -> np.ndarray:
        """
        Returns a copy of data to be used as memory.

        Returns
        -------
        np.ndarray
            memory
        """
        if not self._data_ready or not self._mem_ready:
            raise RuntimeError("Either data of memory was not prepared.")
        return deepcopy(self.memory)

    def set_subsets(
        self,
        train: np.ndarray,
        test: Optional[np.ndarray] = None,
        val_frac: Optional[int] = 0.1,
        shuffle_train=True,
    ):
        """
        Sets the unprocessed train/test subsets and allocates the validation fraction.


        Parameters
        ----------
        train : _type_, optional
            train set
        test : _type_, optional
            test set
        val_frac : Optional[int], optional
            validation fraction (of train set), by default 0.1
        shuffle_train : bool, optional
            defines whether the train set should be shuffled, by default True
        """
        val = None
        if val_frac:
            train, val = train_test_split(
                train,
                test_size=val_frac,
                random_state=42,
                stratify=self._stratification(train),
            )
        if shuffle_train:
            np.random.shuffle(train)
        self.train_subset = train
        self.val_subset = val
        self.test_subset = test
    
    @property
    def model_init_kwargs(self) -> dict:
        metadata = self.get_metadata()
        return {
            "splits": metadata["splits_ind"], 
            "cat_ind": metadata["cat_features_ind"], 
            "feature_sizes": metadata["sizes_with_tokens"], 
            "input_size": metadata["total_size_with_tokens"], 
            "target_ind": self.target_ind,
        }
    
    def inverse_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray: 
        """
        Applies inverse transform of the underlying preprocessor.

        Parameters
        ----------
        X : np.ndarray
            output of NN

        Returns
        -------
        np.ndarray
            transformed data
        """
        return self._preprocessor.inverse_transform(X, **kwargs)


if __name__ == "__main__":

    CV = True
    num = np.random.random(size=(200, 3))
    cat = np.random.randint(0, 3, size=(200, 2))
    data = np.hstack((num, cat))
    if not CV:
        # Example of creating a single dataset instance
        ds = MintDataset(data=data, cat_ind=[3, 4], target_ind=[4], autoprep=True)
        ds.prepare_memory(20)
        print(ds.get_data()[3])
        print(ds.get_data()[3].shape)
    else:
        # Example of generating dataset instances with cv splits
        for ds in MintDataset.with_cv(
            data=data,
            cat_ind=[3, 4],
            target_ind=[4],
            autoprep=True,
            stratified=False,
            strat_column_ind=4,
            n_splits=3,
        ):
            print(ds)
            ds = MintDataset(data=data, cat_ind=[3, 4], target_ind=[4], autoprep=True)
            ds.prepare_memory(20)
            print(ds.get_data()[3])
            print(ds.get_data()[3].shape)
