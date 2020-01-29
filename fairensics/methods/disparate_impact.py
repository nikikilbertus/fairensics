"""Wrapper and functions for DisparateImpact remover from fair-classification.

The base class _DisparateImpact implements predict function for both methods.

The classes AccurateDisparateImpact and FairDisparateImpact inherit from
_DisparateImpact and implement fit() functions with different input signatures
and algorithms for minimization.

Original code:
    https://github.com/mbilalzafar/fair-classification

"""
import warnings
from copy import deepcopy

import numpy as np
from aif360.algorithms import Transformer
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from scipy.optimize import minimize

from .fairness_warnings import FairnessBoundsWarning, DataSetSkewedWarning
from .utils import (
    add_intercept,
    get_protected_attributes_dict,
    get_one_hot_encoding,
    LossFunctions,
)
from ..fairensics_utils import get_unprotected_attributes


class _DisparateImpact(Transformer):
    """Base class for the two methods removing disparate impact.

    Example:
        https://github.com/nikikilbertus/fairensics/blob/master/examples/2_1_fair-classification-disparate-impact-example.ipynb
    """

    def __init__(self, loss_function, warn):
        """
        Args:
            loss_function (str): loss function string from utils.LossFunctions.
            warn (bool): if true, warnings are raised on certain bounds.

        """
        super(_DisparateImpact, self).__init__()

        self._warn = warn
        self._params = {}
        self._initialized = False
        self._loss_function = LossFunctions.get_loss_function(loss_function)

    def predict(self, dataset: BinaryLabelDataset):
        """Make predictions.

        Args:
            dataset: either AIF360 data set or np.ndarray.
                For AIF360 data sets, protected features will be ignored.
                For np.ndarray, only unprotected features should be included.

        Returns:
            Either AIF360 data set or np.ndarray if dataset is np.ndarray.
        """

        if not self._initialized:
            raise ValueError("Model not initialized. Run `fit` first.")

        # TODO: ok?
        if isinstance(dataset, np.ndarray):
            return np.sign(np.dot(add_intercept(dataset), self._params["w"]))

        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.labels = np.sign(
            np.dot(
                add_intercept(get_unprotected_attributes(dataset)),
                self._params["w"],
            )
        )

        # Map the dataset labels to back to their original values.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset_new.labels == 1.0)] = dataset.favorable_label
        temp_labels[(dataset_new.labels == -1.0)] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        if self._warn:
            bound_warnings = FairnessBoundsWarning(dataset, dataset_new)
            bound_warnings.check_bounds()

        return dataset_new

    @staticmethod
    def _get_cov_thresh_dict(cov_thresh, protected_attribute_names):
        """Return dict with covariance threshold for each protected attribute.

        Each attribute gets the same threshold (cov_thresh).

        Args:
            cov_thresh (float): the covariance threshold.
            protected_attribute_names (list(str)): list of protected attribute
                names.

        Returns:
            sensitive_attrs_to_cov_thresh (dict):
            dict of form {"sensitive_attribute_name_1":cov_thresh, ...}.
        """
        sensitive_attrs_to_cov_thresh = {}
        for sens_attr_name in protected_attribute_names:
            sensitive_attrs_to_cov_thresh[sens_attr_name] = cov_thresh

        return sensitive_attrs_to_cov_thresh


class AccurateDisparateImpact(_DisparateImpact):
    """Minimize loss subject to fairness constraints.

    Loss "L" defines whether a logistic regression or a liner SVM is trained.

    Minimize
        L(w)

    Subject to
        cov(sensitive_attributes, true_labels, predictions) <
        sensitive_attrs_to_cov_thresh

    Where:
        predictions: the distance to the decision boundary
    """

    def __init__(self, loss_function=LossFunctions.NAME_LOG_REG, warn=True):
        super(AccurateDisparateImpact, self).__init__(
            loss_function=loss_function, warn=warn
        )

    def fit(
        self,
        dataset: BinaryLabelDataset,
        sensitive_attrs_to_cov_thresh=0,
        sensitive_attributes=None,
    ):
        """Fit the model.

        Args:
            dataset: AIF360 data set
            sensitive_attrs_to_cov_thresh (float or dict): dictionary as
                returned by _get_cov_thresh_dict(). If a single float is passed
                the dict is generated using the _get_cov_thresh_dict() method.
            sensitive_attributes (list(str)): names of protected attributes to
                apply constraints to.
        """
        if self._warn:
            dataset_warning = DataSetSkewedWarning(dataset)
            dataset_warning.check_dataset()

        # constraints are only applied to the selected sensitive attributes
        # if no list is provided, constraints are applied to all protected
        if sensitive_attributes is None:
            sensitive_attributes = dataset.protected_attribute_names

        # if sensitive_attrs_to_cov_thresh is not a dict, each sensitive
        # attribute gets the same threshold
        if not isinstance(sensitive_attrs_to_cov_thresh, dict):
            sensitive_attrs_to_cov_thresh = self._get_cov_thresh_dict(
                sensitive_attrs_to_cov_thresh,
                dataset.protected_attribute_names,
            )

        # fair-classification takes the protected attributes as dict
        protected_attributes_dict = get_protected_attributes_dict(
            dataset.protected_attribute_names, dataset.protected_attributes
        )

        # map labels to -1 and 1
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label)] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label)] = -1.0

        self._params["w"] = self._train_model_sub_to_fairness(
            add_intercept(get_unprotected_attributes(dataset)),
            temp_labels.ravel(),
            protected_attributes_dict,
            sensitive_attributes,
            sensitive_attrs_to_cov_thresh,
        )

        self._initialized = True

        return self

    def _train_model_sub_to_fairness(
        self,
        x,
        y,
        x_control,
        sensitive_attrs,
        sensitive_attrs_to_cov_thresh,
        max_iter=10000,
    ):
        """ Optimize the loss function under fairness constraints.

        Args:
            x (np.ndarray): 2D array of unprotected features and intercept.
            y (np.ndarray): 1D array of labels.
            x_control (dict): dict of protected attributes as returned by
                get_protected_attributes_dict().
            max_iter (int): maximum iterations for solver.
            sensitive_attrs, sensitive_attrs_to_cov_thresh: see fit() method.

        Returns:
            w (np.ndarray): 1D array of the learned weights.

        TODO: sensitive_attrs is redundant. sensitive_attrs_to_cov_thresh
            should only contain features for which constraints are applied.
        """
        constraints = self._get_fairness_constraint_list(
            x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh
        )
        x0 = np.random.rand(x.shape[1])

        w = minimize(
            fun=self._loss_function,
            x0=x0,
            args=(x, y),
            method="SLSQP",
            options={"maxiter": max_iter},
            constraints=constraints,
        )

        if not w.success:
            warnings.warn(
                "Optimization problem did not converge. "
                "Check the solution returned by the optimizer:"
            )
            print(w)

        return w.x

    def _get_fairness_constraint_list(
        self, x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh
    ):
        """Get list of constraints for fairness. See fit method for details.

        Returns:
            constraints (list(str)): fairness constraints in cvxpy format.
                https://www.cvxpy.org/api_reference/cvxpy.constraints.html#
        """
        constraints = []

        for attr in sensitive_attrs:
            attr_arr = x_control[attr]
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

            if index_dict is None:  # binary attribute
                thresh = sensitive_attrs_to_cov_thresh[attr]
                c = {
                    "type": "ineq",
                    "fun": self._test_sensitive_attr_constraint_cov,
                    "args": (x, y, attr_arr_transformed, thresh, False),
                }
                constraints.append(c)
            else:  # categorical attribute, need to set the cov threshs
                for attr_val, ind in index_dict.items():
                    attr_name = attr_val
                    thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                    t = attr_arr_transformed[:, ind]
                    c = {
                        "type": "ineq",
                        "fun": self._test_sensitive_attr_constraint_cov,
                        "args": (x, y, t, thresh, False),
                    }
                    constraints.append(c)

        return constraints

    @staticmethod
    def _test_sensitive_attr_constraint_cov(
        model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose
    ):
        """
        The covariance is computed b/w the sensitive attr val and the distance
        from the boundary. If the model is None, we assume that the
        y_arr_dist_boundary contains the distance from the decision boundary.
        If the model is not None, we just compute a dot product or model and
        x_arr for the case of SVM, we pass the distance from boundary because
        the intercept in internalized for the class and we have compute the
        distance using the project function.

        This function will return -1 if the constraint specified by thresh
        parameter is not satisfied otherwise it will return +1 if the return
        value is >=0, then the constraint is satisfied.
        """
        assert x_arr.shape[0] == x_control.shape[0]
        if len(x_control.shape) > 1:  # make sure we just have one column
            assert x_control.shape[1] == 1

        if model is None:
            arr = y_arr_dist_boundary  # simply the output labels
        else:
            arr = np.dot(
                model, x_arr.T
            )  # the sign of this is the output label

        arr = np.array(arr, dtype=np.float64)
        cov = np.dot(x_control - np.mean(x_control), arr) / len(x_control)

        # <0 if covariance > thresh -- condition is not satisfied
        ans = thresh - abs(cov)
        if verbose is True:
            print("Covariance is", cov)
            print("Diff is:", ans)
            print()
        return ans


class FairDisparateImpact(_DisparateImpact):
    """Minimize disparate impact subject to accuracy constraints.

    Loss "L" defines whether a logistic regression or a liner svm is trained.

    Minimize
        cov(sensitive_attributes, predictions)

    Subject to
        L(w) <= (1-gamma)L(w*)

    Where
        L(w*): is the loss of the unconstrained classifier
        predictions: the distance to the decision boundary
    """

    def __init__(self, loss_function=LossFunctions.NAME_LOG_REG, warn=True):
        super(FairDisparateImpact, self).__init__(
            loss_function=loss_function, warn=warn
        )

    def fit(
        self,
        dataset: BinaryLabelDataset,
        sensitive_attributes=None,
        sep_constraint=False,
        gamma=0,
    ):
        """Fits the model.

        Args:
            dataset: AIF360 data set.
            sensitive_attributes (list(str)): names of protected attributes to
                apply constraints to.
            sep_constraint (bool): apply fine grained accuracy constraint.
            gamma (float): trade off for accuracy for sep_constraint.
        """

        if self._warn:
            dataset_warning = DataSetSkewedWarning(dataset)
            dataset_warning.check_dataset()

        # constraints are only applied to the selected sensitive attributes
        # if no list is provided, constraints for all protected attributes
        if sensitive_attributes is None:
            sensitive_attributes = dataset.protected_attribute_names

        # fair-classification takes the protected attributes as dict
        protected_attributes_dict = get_protected_attributes_dict(
            dataset.protected_attribute_names, dataset.protected_attributes
        )

        # map labels to -1 and 1
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label)] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label)] = -1.0

        self._params["w"] = self._train_model_sub_to_acc(
            add_intercept(get_unprotected_attributes(dataset)),
            temp_labels.ravel(),
            protected_attributes_dict,
            sensitive_attributes,
            sep_constraint,
            gamma,
        )

        self._initialized = True
        return self

    def _train_model_sub_to_acc(
        self,
        x,
        y,
        x_control,
        sensitive_attrs,
        sep_constraint,
        gamma=None,
        max_iter=10000,
    ):
        """Optimize fairness subject to accuracy constraints.

        WARNING: Only first protected attribute is considered as constraint.
        All others are ignored.

        Args:
            x (np.ndarray): 2D array of unprotected features and intercept.
            y (np.ndarray): 1D array of labels.
            x_control (dict): dict of protected attributes as returned by
                get_protected_attributes_dict().
            max_iter (int): maximum number of iterations for solver
            sep_constraint, sensitive_attrs, gamma: see fit() method

        Returns:
            w (np.ndarray): 1D, the learned weight vector for the classifier.

        """

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = x_control_in_arr - np.mean(x_control_in_arr)
            cross_cov *= np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])

        x0 = np.random.rand(x.shape[1])

        # get the initial loss without constraints
        w = minimize(
            fun=self._loss_function,
            x0=x0,
            args=(x, y),
            method="SLSQP",
            options={"maxiter": max_iter},
        )

        old_w = deepcopy(w.x)
        constraints = self._get_accuracy_constraint_list(
            x, y, x_control, w, gamma, sep_constraint, sensitive_attrs
        )

        if len(x_control) > 1:
            warnings.warn(
                "Only the first protected attribute is considered "
                "with this constraint."
            )

        # TODO: only the first protected attribute is passed
        # optimize for fairness under the unconstrained accuracy loss
        w = minimize(
            fun=cross_cov_abs_optm_func,
            x0=old_w,
            args=(x, x_control[sensitive_attrs[0]]),
            method="SLSQP",
            options={"maxiter": max_iter},
            constraints=constraints,
        )

        if not w.success:
            warnings.warn(
                "Optimization problem did not converge. "
                "Check the solution returned by the optimizer:"
            )
            print(w)

        return w.x

    def _get_accuracy_constraint_list(
        self, x, y, x_control, w, gamma, sep_constraint, sensitive_attrs
    ):
        """Constraint list for accuracy constraint.

        Args:
            x, y, x_control, gamma, sep_constraint, sensitive_attrs: see
                _train_model_sub_to_acc method.
            w (scipy.optimize.OptimizeResult): the learned weights of the
                unconstrained classifier.

        Returns:
            constraints (list(str)): accuracy constraints in cvxpy format.
                https://www.cvxpy.org/api_reference/cvxpy.constraints.html#

        TODO: Currently, only the first protected attribute is considered.
            The code should be extended to more than one sensitive_attr.
        """

        def constraint_gamma_all(w_, x_, y_, initial_loss_arr):

            # gamma_arr = np.ones_like(y) * gamma  # set gamma for everyone
            new_loss = self._loss_function(w_, x_, y_)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w_, x_, _y):
            # don't confuse the protected here with the sensitive feature
            # protected/non-protected values protected here means that these
            # points should not be misclassified to negative class
            return np.dot(w_, x_.T)  # if positive, constraint is satisfied

        def constraint_unprotected_people(w_, _ind, old_loss, x_, y_):

            new_loss = self._loss_function(w_, np.array([x_]), np.array(y_))
            return ((1.0 + gamma) * old_loss) - new_loss

        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = self._loss_function(
            w.x, x, y, return_arr=True
        )

        constraints = []
        if sep_constraint:  # separate gamma for different people

            for i in range(0, len(predicted_labels)):
                # TODO: use favorable_label instead of 1.0
                # TODO: extend here to allow more than one sensitive attribute
                if (
                    predicted_labels[i] == 1.0
                    and x_control[sensitive_attrs[0]][i] == 1.0
                ):
                    c = {
                        "type": "ineq",
                        "fun": constraint_protected_people,
                        "args": (x[i], y[i]),
                    }
                    constraints.append(c)

                else:
                    c = {
                        "type": "ineq",
                        "fun": constraint_unprotected_people,
                        "args": (i, unconstrained_loss_arr[i], x[i], y[i]),
                    }
                    constraints.append(c)

        else:  # same gamma for everyone
            c = {
                "type": "ineq",
                "fun": constraint_gamma_all,
                "args": (x, y, unconstrained_loss_arr),
            }
            constraints.append(c)

        return constraints
