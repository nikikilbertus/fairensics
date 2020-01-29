"""Utilities for fair-classification:

    - Loss functions and their names
    - get_one_hot_encoding()
    - add_intercept()
    - get_protected_attributes_dict()

Most functions adopted from https://github.com/mbilalzafar/fair-classification.
"""
import cvxpy
import numpy as np


class LossFunctions:
    """Loss functions for fair-classification.

    This class stores implementations of loss functions used in
    fair-classification. The functions can be accessed using the
    get_loss_function() methods passing loss function names either as numpy or
    cvxpy implementation.
    """

    NAME_SVM_LOSS = "svm_linear"
    NAME_LOG_REG = "logreg"
    NAME_LOG_REG_L1 = "logreg_l1"
    NAME_LOG_REG_L2 = "logreg_l2"
    LOSS_NAMES = [
        NAME_LOG_REG,
        NAME_LOG_REG_L1,
        NAME_LOG_REG_L2,
        NAME_SVM_LOSS,
    ]

    @staticmethod
    def get_loss_function(loss_name):
        """Return loss function for loss_name."""

        assert loss_name in LossFunctions.LOSS_NAMES

        if loss_name == LossFunctions.NAME_LOG_REG:
            return LossFunctions.logistic_loss

        if loss_name == LossFunctions.NAME_LOG_REG_L2:
            return LossFunctions.logistic_loss_l1_reg

        if loss_name == LossFunctions.NAME_LOG_REG_L2:
            return LossFunctions.logistic_loss_l2_reg

        if loss_name == LossFunctions.NAME_SVM_LOSS:
            return LossFunctions.hinge_loss

    @staticmethod
    def get_cvxpy_loss_function(loss_name):
        """Return cvxpy loss function for loss_name."""

        assert loss_name in LossFunctions.LOSS_NAMES

        if loss_name == LossFunctions.NAME_LOG_REG:
            return LossFunctions.cvxpy_logistic_loss

        if loss_name == LossFunctions.NAME_LOG_REG_L1:
            return LossFunctions.cvxpy_logistic_loss_l1

        if loss_name == LossFunctions.NAME_LOG_REG_L2:
            return LossFunctions.cvxpy_logistic_loss_l2

        if loss_name == LossFunctions.NAME_SVM_LOSS:
            return LossFunctions.cvxpy_hinge_loss

    @staticmethod
    def cvxpy_logistic_loss(w, X, y, num_points=None):
        """CVXPY implementation of logistic loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            num_points (int): number of points in X
                (first dimension of X "n_samples",
                but some methods pass a different value for scaling).

        Returns:
            (float): the loss.
        """
        if num_points is None:
            num_points = X.shape[0]
        return (
            cvxpy.sum(cvxpy.logistic(cvxpy.multiply(-y, X * w))) / num_points
        )

    @staticmethod
    def cvxpy_logistic_loss_l1(w, X, y, lam=None, num_points=None):
        """CVXPY implementation of L1 regularized logistic loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            lam (float): regularization parameter.
            num_points (int): number of points in X (corresponds to the first
                dimension of X "n",
                but some methods pass a different value for scaling).

        Returns:
            (float): the loss.
        """
        if num_points is None:
            num_points = X.shape[0]

        if lam is None:
            lam = 1.0

        yz = cvxpy.multiply(-y, X * w)

        logistic_loss = cvxpy.sum(cvxpy.logistic(yz))
        l1_reg = (float(lam) / 2.0) * cvxpy.norm1(w)
        out = logistic_loss + l1_reg

        return out / num_points

    @staticmethod
    def cvxpy_logistic_loss_l2(w, X, y, lam=None, num_points=None):
        """CVXPY implementation of L2 regularized logistic loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            lam (float): regularization parameter.
            num_points (int): number of points in X (corresponds to the first
                dimension of X "n",
                but some methods pass a different value for scaling).

        Returns:
            (float): the loss.
        """
        if lam is None:
            lam = 1.0

        if num_points is None:
            num_points = X.shape[0]

        yz = cvxpy.multiply(-y, X * w)

        logistic_loss = cvxpy.sum(cvxpy.logistic(yz))
        l2_reg = (float(lam) / 2.0) * cvxpy.pnorm(w, p=2) ** 2
        out = logistic_loss + l2_reg

        return out / num_points

    @staticmethod
    def cvxpy_hinge_loss(w, X, y, num_points=None):
        """CVXPY implementation of hinge loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            num_points (int): number of points in X (corresponds to the first
                dimension of X "n",
                but some methods pass a different value for scaling).

        Returns:
            (float): the loss.
        """
        if num_points is None:
            num_points = X.shape[0]

        res = cvxpy.sum(cvxpy.max(0, 1 - cvxpy.multiply(y, X * w)))
        return res / num_points

    @staticmethod
    def hinge_loss(w, X, y):
        """Numpy implementation of hinge loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).

        Returns:
            (float): the loss.
        """
        yz = y * np.dot(X, w)  # y * (x.w)
        yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

        return sum(yz)

    @staticmethod
    def logistic_loss(w, X, y, return_arr=False):
        """Numpy implementation of logistic loss.

        This function is used from scikit-learn source code

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            return_arr (bool): if true, an array is returned otherwise the sum
                of the array

        Returns:
            (float or list(float)): the loss.
        """

        yz = y * np.dot(X, w)
        # Logistic loss is the negative of the log of the logistic function.
        if return_arr:
            return -LossFunctions.log_logistic(yz)

        return -np.sum(LossFunctions.log_logistic(yz))

    @staticmethod
    def logistic_loss_l1_reg(w, X, y, lam=None):
        """Numpy implementation of L1 regularized logistic loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            lam (float): regularization parameter.

        Returns:
            (float): the loss.
        """
        if lam is None:
            lam = 1.0

        yz = y * np.dot(X, w)
        # Logistic loss is the negative of the log of the logistic function.
        logistic_loss = -np.sum(LossFunctions.log_logistic(yz))
        l1_reg = (float(lam) / 2.0) * np.sum(abs(w))
        out = logistic_loss + l1_reg
        return out

    @staticmethod
    def logistic_loss_l2_reg(w, X, y, lam=None):
        """Numpy implementation of L2 regularized logistic loss.

        Args:
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features)
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            lam (float): regularization parameter.

        Returns:
            (float): the loss.
        """
        if lam is None:
            lam = 1.0

        yz = y * np.dot(X, w)
        # Logistic loss is the negative of the log of the logistic function.
        logistic_loss = -np.sum(LossFunctions.log_logistic(yz))
        l2_reg = (float(lam) / 2.0) * np.sum([elem * elem for elem in w])
        out = logistic_loss + l2_reg
        return out

    @staticmethod
    def log_logistic(X):
        """Log_logistic from  scikit-learn source code. Source link below.

        Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
        Source code at:
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py

        Args:
            X (array-like): shape (M, N) Argument to the logistic function

        Returns:
            out (np.ndarray): shape (M, N) Log of the logistic function at
                every point in x
        """
        if X.ndim > 1:
            raise Exception("Array of samples cannot be more than 1-D!")
        out = np.empty_like(X)  # same dimensions and data types

        idx = X > 0
        out[idx] = -np.log(1.0 + np.exp(-X[idx]))
        out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
        return out


def get_one_hot_encoding(arr):
    """Returns one hot encoding of array arr.

    Args:
        arr (np.ndarray): 1D array with int values.

    Returns:
        Tuple consisting of out_arr (np.ndarray) one-hot encoded matrix and
        index_dict (dict) dictionary original_val -> column in encoded matrix.
    """
    arr = np.array(arr, dtype=int)
    assert len(arr.shape) == 1  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (
        num_uniq_vals == 2
        and attr_vals_uniq_sorted[0] == 0
        and attr_vals_uniq_sorted[1] == 1
    ):
        return arr, None

    index_dict = {}  # value to the column number
    for i in enumerate(len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in enumerate(arr):
        tup = np.zeros(num_uniq_vals)
        val = arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


def add_intercept(x):
    """Adds intercept (column of ones) to X."""
    m, _ = x.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, x), axis=1)


def get_protected_attributes_dict(names, attributes):
    """Returns dictionary of protected attributes.

    The dictionary has the form: {"s1": [...], "s2": [...], ... }
    Key "sI" is the sensitive feature name, and [...] the 1D array holding the
    sensitive feature.

    Args:
        names (list(str)): names of the attributes in attributes.
        attributes (np.ndarray): 2D array of the sensitive features.

    Returns:
        (dict): {"s1": [attributes[:, 1]], "s2":[attributes[:, 2]], ... }
    """
    protected_attributes_dict = {}
    for i in range(len(names)):
        name = names[i]
        data = attributes[:, i]
        protected_attributes_dict[name] = data
    return protected_attributes_dict
