"""Utility functions."""

import numpy as np
from aif360.datasets import StructuredDataset


def get_unprotected_attributes(dataset: StructuredDataset):
    """Returns unprotected features from data set.

    Args:
        dataset (StructuredDataset): data set with features, protected features
            and labels.

    Returns:
        (np.ndarray) of unprotected features only
    """
    unprotected_feature_names = np.setdiff1d(
        dataset.feature_names, dataset.protected_attribute_names
    )

    unprotected_feature_indexes = np.in1d(
        dataset.feature_names, unprotected_feature_names
    )
    return dataset.features[:, unprotected_feature_indexes]
