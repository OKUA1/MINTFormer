import os
import numpy as np 
import pandas as pd
from copy import deepcopy
import os

from mintformer_prep import MintDataLoader, MintDataset # type: ignore
from mintformer_prep.samplers import RandomSampler, IdentitySampler, NaiveClusterCentroidsSampler # type: ignore


def read_clf():
    test_clf_csv = os.path.join(os.path.dirname(__file__), 'test_clf.csv')
    df = pd.read_csv(test_clf_csv)
    df.pop('RowNumber')
    df.pop('Surname')
    df.pop('CustomerId')
    df['Exited'] = df['Exited'].astype(str)
    cat_ind, target_ind = [1,2,8,10], [10]
    N = 2
    return df, cat_ind, target_ind, N

def test_target_masking():
    df, cat_ind, target_ind, N = read_clf()
    ds = MintDataset(data=df, cat_ind=cat_ind, target_ind=target_ind, autoprep=True)
    ds.prepare_memory(mem_size = 0.999, sampler = IdentitySampler())
    dl = MintDataLoader(ds)

    for batch in dl.generate_train(
                        batch_size=256,
                        feature_mask_prob=0.15,
                        feature_noise_prob=0.1,
                        target_mask_prob=1.0,
                        target_noise_prob=0.0,
                    ):
        unmasked, masked, mask, noise_mask, mem_mask, _, _ = batch

        assert np.all(unmasked[:, -1] == 0.), 'Found masked token in an unmasked dataset'
        assert np.any(unmasked[:, -2] != 0.), 'At least one value is supposed to be non zero'
        assert np.any(unmasked[:, -3] != 0.), 'At least one value is supposed to be non zero'
        assert np.all(masked[:, -1] == 1.), 'Missing masked token'
        assert np.all(masked[:, -2] == 0.), 'Found unmasked value'
        assert np.all(masked[:, -3] == 0.), 'Found unmasked value'
        assert np.all(ds.memory[:, -1] == 0.), 'Found masked memory'

def test_feature_masking_noise():
    df, cat_ind, target_ind, N = read_clf()
    ds = MintDataset(data=df, cat_ind=cat_ind, target_ind=target_ind, autoprep=True)
    ds.prepare_memory(mem_size = 0.999, sampler = IdentitySampler())
    dl = MintDataLoader(ds)

    feature_value_column = -5
    feature_token_column = -4

    for batch in dl.generate_train(
                        batch_size=256,
                        feature_mask_prob=1.0,
                        feature_noise_prob=0.5,
                        target_mask_prob=1.0,
                        target_noise_prob=0.0,
                    ):
        unmasked, masked, mask, noise_mask, mem_mask, _, _ = batch

        # when masking all the features and adding noise, we should encounter samples with and without mask token
        assert np.any(masked[:, feature_token_column] == 0.)
        assert np.any(masked[:, feature_token_column] == 1.)
        # in this case some of the values should be exactly 0.0 and some features != 0.0
        assert np.any(masked[:, feature_value_column] == 0.)
        assert np.any(masked[:, feature_value_column] != 0.)