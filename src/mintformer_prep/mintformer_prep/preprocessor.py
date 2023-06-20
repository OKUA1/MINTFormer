from typing import List, Union, Any, Dict, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


class PadInserter(BaseEstimator, TransformerMixin):
    """Preprocessor that adds constant padding at the end of 1-st axis."""

    def __init__(self, pad_size: int = 1, pad_value: float = 0.0):
        self.pad_size = pad_size
        self.pad_value = pad_value

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """
        Does nothing. Exists purely for sklearn api compat.

        Parameters
        ----------
        X : np.ndarray
        """
        pass

    def transform(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Adds padding to the end of the vectors.

        Parameters
        ----------
        X : np.ndarray
            inputs

        Returns
        -------
        np.ndarray
            inputs with appended paddings
        """
        if len(X.shape) < 2:
            X = np.expand_dims(X, 0)
        return np.pad(
            X,
            ((0, 0), (0, self.pad_size)),
            mode="constant",
            constant_values=self.pad_value,
        )

    def fit_transform(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Equivalent to `transform` since fit does nothing.

        Parameters
        ----------
        X : np.ndarray
            inputs

        Returns
        -------
        np.ndarray
            inputs with appended paddings
        """
        self.fit(X)
        return self.transform(X)


class MintPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms the dataset into required format.
    All categorical features are one-hot encoded and numerical features are scaled.
    A single element zero-padding is added to the end of feature (to be used as mask token later on).
    """

    def __init__(self, cat_ind: List[int]) -> None:
        self.cat_ind = cat_ind

    def _make_branch(self, col: int) -> Pipeline:
        """
        Makes a sub-pipeline with standard scaler for numerical and one-hot-encoder for categorical features.

        Parameters
        ----------
        col : int
            column index

        Returns
        -------
        Pipeline
            branch to be used in column transformer
        """
        is_cat = col in self.cat_ind
        if is_cat:
            steps = [
                ("Encode", OneHotEncoder(sparse=False, handle_unknown='ignore')),
                ("Insert0MaskToken", PadInserter(pad_size=1, pad_value=0.0)),
            ]
        else:
            steps = [
                ("Standardize", StandardScaler()),
                ("Insert0MaskToken", PadInserter(pad_size=1, pad_value=0.0)),
            ]
        return Pipeline(steps=steps)

    def _get_np_arr(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """

        Converts pd.Dataframe to np array.
        Ensures that np array always has batch dimension.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            input

        Returns
        -------
        np.ndarray
            converted array

        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if not isinstance(X, np.ndarray):
            raise ValueError("X should either be a dataframe or a numpy array")
        if len(X.shape) < 2:
            X = np.expand_dims(X, 0)
        return X

    def _store_feature_sizes(self, X: np.ndarray, i: int) -> None:
        """
        Stores the sizes (including mask token) of the encoded features
        Numerical features have a constant size of 2 (value + mask token)
        Categorical features have the size n_unique+1


        Parameters
        ----------
        X : np.ndarray
            dataset
        i : int
            column
        """
        if not (i in self.cat_ind):
            self.f_sizes.append(2)
        else:
            self.f_sizes.append(len(np.unique(X[:, i])) + 1)

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Fits the preprocessor

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            training data
        """
        X = self._get_np_arr(X)
        self.f_sizes = []
        branches = []
        for i in range(X.shape[-1]):
            branch = self._make_branch(i)
            branches.append((f"col{i}", branch, [i]))
            self._store_feature_sizes(X, i)
        self._pipeline = ColumnTransformer(branches)
        self._pipeline.fit(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms the dataset

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            input

        Returns
        -------
        np.ndarray
            transformed input
        """
        X = self._get_np_arr(X)
        return self._pipeline.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fits preprocessor and transforms the data

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            training data

        Returns
        -------
        np.ndarray
            transformed training data
        """
        self.fit(X)
        return self.transform(X)

    def _calc_splits(self, with_tokens: bool = True) -> List[int]:
        """
        Calculates the begining indices of each (encoded) feature

        Parameters
        ----------
        with_tokens : bool, optional
            if True, accounts for additional mask token element, by default True

        Returns
        -------
        List[int]
            _description_
        """
        csum = 0
        offset = 0 if with_tokens else 1
        splits = []
        for i in range(len(self.f_sizes) - 1):
            csum += self.f_sizes[i] - offset
            splits.append(csum)
        return splits

    def get_metadata(self) -> Dict:
        """
        Returns metadata about the dataset

        Returns
        -------
        Dict
            `n_features`: number of features;
            `sizes_with_tokens`: sizes of each feature including the mask token;
            `total_size_with_tokens`: number of features after transformation (including mask tokens);
            `total_size_no_tokens`: number of features after transformation (excluding tokens);
            `num_features_ind`: indices of numerical features;
            `cat_features_ind`: indices of categorical features;
            `splits_ind`: beggining indices of each feature (tokens are accounted for);
            `splits_ind_no_tokens`: beggining indices of each feature (tokens are not accounted for);
        """
        no_tokens = (lambda x: [i - 1 for i in x])(self.f_sizes)
        n_features = len(self.f_sizes)
        num_ind = [i for i in range(n_features) if i not in self.cat_ind]
        return {
            "n_features": n_features,
            "sizes_with_tokens": self.f_sizes,
            "sizes_no_tokens": no_tokens,
            "total_size_with_tokens": sum(self.f_sizes),
            "total_size_no_tokens": sum(self.f_sizes) - n_features,
            "num_features_ind": num_ind,
            "cat_features_ind": self.cat_ind,
            "splits_ind": self._calc_splits(with_tokens=True),
            "splits_ind_no_tokens": self._calc_splits(with_tokens=False),
        }

    def _extract_scaler(self, name: str) -> Callable:
        """
        Extracts the relevant StandardScaler from column transformer given the branch name.

        Parameters
        ----------
        name : str
            name of the column transformer branch

        Returns
        -------
        Callable
            standard scaler object
        """
        return self._pipeline.named_transformers_[name].named_steps["Standardize"]

    def inverse_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Applies reverse_transform of StandardScaler on numerical columns.
        Note: this method ignores all categorical columns.

        Parameters
        ----------
        X : np.ndarray
            output of the NN

        Returns
        -------
        np.ndarray
            X with numerical targets rescaled
        """

        splits = self._calc_splits(with_tokens=False)
        # print(splits)
        for i in range(len(self.f_sizes)):
            if i not in self.cat_ind:
                if i == 0:
                    col_x = 0
                else:
                    col_x = splits[i - 1]
                branch_name = f"col{i}"
                scaler = self._extract_scaler(branch_name)
                X[:, col_x] = np.squeeze(
                    scaler.inverse_transform(X[:, col_x].reshape(-1, 1))
                )

        return X


if __name__ == "__main__":
    # Example usage
    x = np.random.random(size=(5, 3))
    cat = np.asarray([["A"], ["B"], ["C"], ["A"], ["B"]])
    x = np.hstack((x, cat))
    print(x)
    prp = MintPreprocessor([3])
    prp.fit(x)
    print(prp.transform(x[0]))
    print(prp.get_metadata())
