"""Decision boundary plots for 2D data sets.

The code for the scatter function is adopted from fair-classification:
    https://github.com/mbilalzafar/fair-classification.

Usage and interpretation is explained in the example jupyter notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.decomposition import PCA

from ..fairensics_utils import get_unprotected_attributes


class DecisionBoundary:
    """Class for plotting decision boundaries against two axes.

    The data may be down sampled to two dimensions before plotting.
    The decision boundary plots are generated using a mesh grid and the
    following procedure:

        1. If necessary, the data is down-sampled to two dimensions
        2. Min and maximum values for each axis are extracted
        3. A mesh grid is created
        4. If necessary, the mesh grid is up-sampled again
        5. Predictions are made on the mesh grid
        6. Predictions are plotted against the maybe down sampled axis

    TODO: add option to scale data to [0,1]
    """

    _UNPRIVILEGED_GROUP_NEGATIVE_LABEL = (
        "Negative label and unprivileged group"
    )
    _UNPRIVILEGED_GROUP_POSITIVE_LABEL = (
        "Positive label and unprivileged group"
    )
    _PRIVILEGED_GROUP_NEGATIVE_LABEL = "Negative label and privileged group"
    _PRIVILEGED_GROUP_POSITIVE_LABEL = "Positive label and privileged group"

    def __init__(
        self,
        colors=("k", "c", "m", "b", "g", "r", "y"),
        downsampler=PCA(n_components=2),
    ):
        """

        Args:
            colors: iterator over possible colors for the decision boundaries
            downsampler: function to down sample data points
                must implement 'fit_transform' and 'inverse_transform' methods
        """
        self._colors = iter(colors)
        self._downsampler = downsampler
        self._downsampled = False

    def _maybe_downsample(self, dataset, only_unprotected):
        """Downsample data to 2D for plotting.

        If only_unprotected is true the protected features are removed for both
        down sampling or when the raw data set is returned.

        Args:
            dataset (StructuredDataset): aif dataset with features and labels.
            only_unprotected (bool): protected features are ignored if true.

        Returns:
            (np.ndarray): 2D array of maybe down-sampled features.
        """
        if only_unprotected:
            unprotected_features = get_unprotected_attributes(dataset)

            if unprotected_features.shape[1] > 2:
                self._downsampled = True
                return self._downsampler.fit_transform(unprotected_features)

            return unprotected_features

        if dataset.features.shape[1] > 2:
            self._downsampled = True
            return self._downsampler.fit_transform(dataset.features)

        return dataset.features

    def _maybe_upsample(self, mesh):
        """Up-samples mesh for prediction, if down-sampler was called earlier.

        Args:
            mesh (np.array): 2D mesh grid.

        Returns:
            (np.ndarray): Up-sampled mesh grid.
        """
        if self._downsampled:
            return self._downsampler.inverse_transform(mesh)
        return mesh

    # noinspection Duplicates
    def scatter(
        self,
        dataset: BinaryLabelDataset,
        protected_attribute_ind=0,
        only_unprotected=True,
        num_to_draw=100,
    ):
        """ Scatter plot the points in dataset.

        Protected and unprotected individuals and positive and negative label
        are distinguished. Only one protected attribute is considered for
        plotting.

        Args:
            dataset (BinaryLabelDataset): data set to plot.
            protected_attribute_ind (int): index of the protected attribute
                to consider.
            only_unprotected (bool): if true, the classifier only uses the
                unprotected attributes.
            num_to_draw (int): number of points to draw.

        """
        x_draw = self._maybe_downsample(dataset, only_unprotected)[
            :num_to_draw, :
        ]
        y_draw = dataset.labels[:num_to_draw, 0]
        x_protected_draw = dataset.protected_attributes[
            :num_to_draw, protected_attribute_ind
        ]

        unprivileged_group = dataset.unprivileged_protected_attributes[
            protected_attribute_ind
        ]
        unprivileged_mask = x_protected_draw == unprivileged_group

        privileged_group = dataset.privileged_protected_attributes[
            protected_attribute_ind
        ]
        privileged_mask = x_protected_draw == privileged_group

        X_unprivileged = x_draw[unprivileged_mask]
        X_privileged = x_draw[privileged_mask]
        y_unprivileged = y_draw[unprivileged_mask]
        y_privileged = y_draw[privileged_mask]

        plt.scatter(
            X_unprivileged[y_unprivileged == dataset.favorable_label][:, 0],
            X_unprivileged[y_unprivileged == dataset.favorable_label][:, 1],
            color="green",
            marker="x",
            label=self._UNPRIVILEGED_GROUP_POSITIVE_LABEL,
        )

        plt.scatter(
            X_unprivileged[y_unprivileged == dataset.unfavorable_label][:, 0],
            X_unprivileged[y_unprivileged == dataset.unfavorable_label][:, 1],
            color="red",
            marker="x",
            label=self._UNPRIVILEGED_GROUP_NEGATIVE_LABEL,
        )

        plt.scatter(
            X_privileged[y_privileged == dataset.favorable_label][:, 0],
            X_privileged[y_privileged == dataset.favorable_label][:, 1],
            color="green",
            facecolors="none",
            label=self._PRIVILEGED_GROUP_POSITIVE_LABEL,
        )

        plt.scatter(
            X_privileged[y_privileged == dataset.unfavorable_label][:, 0],
            X_privileged[y_privileged == dataset.unfavorable_label][:, 1],
            color="red",
            facecolors="none",
            label=self._PRIVILEGED_GROUP_NEGATIVE_LABEL,
        )

    def add_boundary(
        self,
        dataset: BinaryLabelDataset,
        clf,
        label="",
        only_unprotected=True,
        num_points=100,
        cmap=None,
    ):
        """Adds decision boundary to the current plot.

        If the data set is two dimensional, the boundary is directly plotted
        using a mesh grid. Otherwise, a mesh gird is generated on the
        down-sampled points and up-sampled again for prediction.

        Args:
            dataset (BinaryLabelDataset): the labeled data set.
            clf (object): the classifier object (must implement a predict
                function).
            label (str): the label for the decision boundary.
            only_unprotected (bool): if true, the classifier only uses the
                unprotected attributes.
            num_points (int): number of points in mesh grid.
            cmap (str): colormap from matplotlib. If provided background of the
                plot is colored.

        """
        dataset = self._maybe_downsample(dataset, only_unprotected)

        x1_min, x1_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
        x2_min, x2_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1

        x1_step = (x1_max - x1_min) / num_points
        x2_step = (x2_max - x2_min) / num_points

        xx, yy = np.meshgrid(
            np.arange(x1_min, x1_max, x1_step),
            np.arange(x2_min, x2_max, x2_step),
        )

        mesh = self._maybe_upsample(np.c_[xx.ravel(), yy.ravel()])
        Z = clf.predict(mesh)
        Z = Z.reshape(xx.shape)

        CS = plt.contour(xx, yy, Z, colors=next(self._colors))
        CS.collections[0].set_label(label)

        if cmap is not None:
            plt.contourf(xx, yy, Z, cmap=cmap)

    @staticmethod
    def show(title="", xlabel="", ylabel=""):
        """Shows the plot"""
        plt.legend()
        plt.title = title
        plt.xlabel = xlabel
        plt.ylabel = ylabel
        plt.show()
