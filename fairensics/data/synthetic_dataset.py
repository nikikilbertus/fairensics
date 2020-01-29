"""A synthetic 2D data set with two features and one protected attribute
implemented as AIF360 BinaryLabelDataset.

The code is adopted from: https://github.com/mbilalzafar/fair-classification.

Additionally, a function to scatter plot the points is available.

TODO: pass labels, colors etc. to the plot function
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from scipy.stats import multivariate_normal  # generating synthetic data


class SyntheticDataset(BinaryLabelDataset):
    """Synthetic data set with two features and one protected attribute.

    The data set is randomly generated from two gaussians each time.
    Both protected attribute and label are binary and features are numerical.
    """

    _UNPRIVILEGED_GROUP_NEGATIVE_LABEL = "Prot. -ve"
    _UNPRIVILEGED_GROUP_POSITIVE_LABEL = "Prot. +ve"
    _PRIVILEGED_GROUP_NEGATIVE_LABEL = "Non-prot. -ve"
    _PRIVILEGED_GROUP_POSITIVE_LABEL = "Non-prot. +ve"

    def __init__(
        self,
        n_samples=1000,
        label_name="label",
        feature_one_name="feature_1",
        feature_two_name="feature_2",
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_name="protected_attribute",
        privileged_class=1,
        unprivileged_class=0,
        sd=1122334455,
        mu_1=(2, 2),
        sigma_1=((5, 1), (1, 5)),
        mu_2=(-2, -2),
        sigma_2=((10, 1), (1, 3)),
        initial_discrimination=4.0,
    ):
        """
        Args:
            n_samples (int) : the number of samples to generate
            label_name (str): name of the column storing the target variable
            feature_one_name (str): name of the first unprotected feature
            feature_two_name (str): name of the second unprotected feature
            favorable_label (int): label considered positive
            unfavorable_label (int):  label considered negative
            protected_attribute_name (str): the name of the protected attribute
            privileged_class (int): class of protected attribute considered
                positive
            unprivileged_class (int): class of protected attribute considered
                negative
            sd (int): seed for random generator
            mu_1 (float, float): mean of positive group cluster
            sigma_1 ((float, float), (float, float)): covariance of positive
                group cluster
            mu_2 (float, float): mean of negative group cluster
            sigma_2 ((float, float), (float, float)):  covariance of negative
                group cluster
            initial_discrimination (float): initial discrimination factor
        """
        np.random.seed(sd)

        self.n_samples = n_samples
        self._mu_1 = np.array(mu_1)
        self._sigma_1 = np.array(sigma_1)
        self._mu_2 = np.array(mu_2)
        self._sigma_2 = np.array(sigma_2)
        self._disc_factor = np.pi / initial_discrimination

        X1, X2, X_s, y = self._gen_data(
            favorable_label,
            unfavorable_label,
            privileged_class,
            unprivileged_class,
        )

        df = pd.DataFrame(
            {
                feature_one_name: X1,
                feature_two_name: X2,
                protected_attribute_name: X_s,
                label_name: y,
            }
        )

        super(SyntheticDataset, self).__init__(
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
            df=df,
            label_names=[label_name],
            protected_attribute_names=[protected_attribute_name],
            privileged_protected_attributes=[[privileged_class]],
            unprivileged_protected_attributes=[[unprivileged_class]],
        )

    def _gen_gaussian(self, mean_in, cov_in, class_label):
        """Generates n_samples from gaussian distribution"""
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(self.n_samples)
        y = np.ones(self.n_samples, dtype=float) * class_label
        return nv, X, y

    def _gen_data(
        self,
        favorable_label,
        unfavorable_label,
        privileged_class,
        unprivileged_class,
    ):
        """Code for generating the synthetic data.

        We will have two non-sensitive features and one sensitive feature.

        Args:
            favorable_label (int): the label considered positive.
            unfavorable_label (int): the label considered negative.
            privileged_class (int): the class in protected attribute considered
                privileged.
            unprivileged_class (int): the class in protected attribute
                considered unprivileged.

        Returns:
            X_0 (np.ndarary): 1D array, the first unprotected feature.
            X_1 (np.ndarary): 1D array, the second unprotected feature.
            X_s (np.ndarary): 1D array, binary, the protected attribute.
            y (np.ndarary): 1D array, binary, the labels.

        """
        nv1, X1, y1 = self._gen_gaussian(
            self._mu_1, self._sigma_1, favorable_label
        )
        nv2, X2, y2 = self._gen_gaussian(
            self._mu_2, self._sigma_2, unfavorable_label
        )

        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))

        # shuffle the data
        perm = list(range(0, self.n_samples * 2))
        np.random.shuffle(perm)
        X = X[perm]
        y = y[perm]

        rotation_mult = np.array(
            [
                [np.cos(self._disc_factor), -np.sin(self._disc_factor)],
                [np.sin(self._disc_factor), np.cos(self._disc_factor)],
            ]
        )

        X_aux = np.dot(X, rotation_mult)

        # Generate the sensitive feature here
        x_sensitive = []
        for i in range(0, len(X)):
            x = X_aux[i]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)

            # normalize the probabilities
            s = p1 + p2
            p1 = p1 / s

            r = np.random.uniform()

            if r < p1:  # the first cluster is the positive class
                x_sensitive.append(privileged_class)
            else:
                x_sensitive.append(unprivileged_class)

        x_sensitive = np.array(x_sensitive)

        return X[:, 0], X[:, 1], x_sensitive, y

    # noinspection Duplicates
    def plot(self, num_to_draw=200):
        """Plot subsample of data with unprotected features on x and y axis."""
        x_draw = self.features[:num_to_draw, :2]  # ignore the protected column
        y_draw = self.labels[:num_to_draw, 0]
        x_sensitive_draw = self.protected_attributes[:num_to_draw, 0]

        idx = x_sensitive_draw == self.unprivileged_protected_attributes[0]
        X_unprivileged = x_draw[idx]
        idx = x_sensitive_draw == self.privileged_protected_attributes[0]
        X_privileged = x_draw[idx]

        idx = x_sensitive_draw == self.unprivileged_protected_attributes[0]
        y_unprivileged = y_draw[idx]
        idx = x_sensitive_draw == self.privileged_protected_attributes[0]
        y_privileged = y_draw[idx]

        # pylint: disable=duplicate-code
        plt.scatter(
            X_unprivileged[y_unprivileged == self.favorable_label][:, 0],
            X_unprivileged[y_unprivileged == self.favorable_label][:, 1],
            color="green",
            marker="x",
            label=self._UNPRIVILEGED_GROUP_POSITIVE_LABEL,
        )

        plt.scatter(
            X_unprivileged[y_unprivileged == self.unfavorable_label][:, 0],
            X_unprivileged[y_unprivileged == self.unfavorable_label][:, 1],
            color="red",
            marker="x",
            label=self._UNPRIVILEGED_GROUP_NEGATIVE_LABEL,
        )

        plt.scatter(
            X_privileged[y_privileged == self.favorable_label][:, 0],
            X_privileged[y_privileged == self.favorable_label][:, 1],
            color="green",
            facecolors="none",
            label=self._PRIVILEGED_GROUP_POSITIVE_LABEL,
        )

        plt.scatter(
            X_privileged[y_privileged == self.unfavorable_label][:, 0],
            X_privileged[y_privileged == self.unfavorable_label][:, 1],
            color="red",
            facecolors="none",
            label=self._PRIVILEGED_GROUP_NEGATIVE_LABEL,
        )

        plt.legend()
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])

        plt.xlim((np.min(x_draw[:, 0]) - 2, np.max(x_draw[:, 0]) + 2))
        plt.ylim((np.min(x_draw[:, 1]) - 2, np.max(x_draw[:, 1]) + 2))

        plt.show()
