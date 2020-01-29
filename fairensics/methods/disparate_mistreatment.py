"""Wrapper for the function "DisparateMistreatment" from fair-classification.

Besides fit() and predict() the class contains three functions:

Two functions taken from fair-classification performing the actual training:
    _train_model_disp_mist():
    _get_constraint_list_cov():

And a function to transform protected attributes into a dict:
    _get_cov_thresh_dict():

Original code:
    https://github.com/mbilalzafar/fair-classification

"""
import warnings

import cvxpy
import numpy as np
from aif360.algorithms import Transformer
from aif360.datasets.binary_label_dataset import BinaryLabelDataset

from .fairness_warnings import FairnessBoundsWarning, DataSetSkewedWarning
from .utils import (
    add_intercept,
    get_protected_attributes_dict,
    get_one_hot_encoding,
    LossFunctions,
)
from ..fairensics_utils import get_unprotected_attributes


class DisparateMistreatment(Transformer):
    """Disparate mistreatment free classifier.
    Loss "L" defines whether a logistic regression or a liner svm is trained.

    Minimize
        L(w)

    Subject to
        cov(sensitive_attributes, predictions) < sensitive_attrs_to_cov_thresh

    Where
        predictions: the distance to the decision boundary

    Example:
        https://github.com/nikikilbertus/fairensics/blob/master/examples/2_2_fair-classification-mistreatment-example.ipynb

    """

    # The original function takes an integer for the desired constraint.
    # The index in the list corresponds to the integer value of each constraint
    _CONS_TYPES = ["all", "fpr", "fnr", None, "fprfnr"]

    def __init__(
        self,
        loss_function=LossFunctions.NAME_LOG_REG,
        constraint_type=None,
        take_initial_sol=True,
        warn=True,
        tau=0.005,
        mu=1.2,
        EPS=1e-6,
        max_iter=100,
        max_iter_dccp=50,
    ):
        """
        Args:
            loss_function (str): name of loss function defined in utils
            constraint_type (str): one of the values in _CONS_TYPE
            take_initial_sol (bool):
            warn (bool): if true, warnings are raised on certain bounds
            tau, mu, EPS, max_iter, max_iter_dccp: solver related parameters
        """

        assert constraint_type in self._CONS_TYPES

        super(DisparateMistreatment, self).__init__()

        self._params = {}
        self._initialized = False
        self._loss_function = LossFunctions.get_cvxpy_loss_function(
            loss_function
        )
        self._constraint_type = self._CONS_TYPES.index(constraint_type)
        self._take_initial_sol = take_initial_sol
        self._warn = warn
        self._tau = tau
        self._mu = mu
        self._EPS = EPS
        self._max_iter = max_iter  # for the convex program
        self._max_iter_dccp = max_iter_dccp  # for the dccp algo

    def fit(
        self, dataset: BinaryLabelDataset, sensitive_attrs_to_cov_thresh=0
    ):
        """Fits the model.

        Args:
            dataset: AIF360 data set
            sensitive_attrs_to_cov_thresh (dict or float): covariance between
                sensitive attribute and decision boundary
        """
        if self._warn:
            dataset_warning = DataSetSkewedWarning(dataset)
            dataset_warning.check_dataset()

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

        self._params["w"] = self._train_model_disp_mist(
            add_intercept(get_unprotected_attributes(dataset)),
            temp_labels.ravel(),
            protected_attributes_dict,
            sensitive_attrs_to_cov_thresh,
        )
        self._initialized = True
        return self

    def predict(self, dataset: BinaryLabelDataset):
        """Make predictions.

        Args:
            dataset: AIF360 data set

        Returns:
            either AIF360 data set or np.array if dataset is also np.array
        """
        if not self._initialized:
            raise ValueError("Model not initialized. Run fit() first.")

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

    def _train_model_disp_mist(
        self, x, y, x_control, sensitive_attrs_to_cov_thresh
    ):
        """Trains model.

        Args:
            x (np.ndarray): 2D array of unprotected features with intercept
            y (np.ndarray): 1D numpy array of targets
            x_control (dict):  {"s": [...]}, key "s" is the sensitive feature
                name and value a 1D vector of values
            sensitive_attrs_to_cov_thresh: covariance threshold for each
                cons_type, eg, key 1 for FPR

        Returns:
            w (np.ndarray): 1D array of the learned weights

        """
        _num_points, num_features = x.shape

        w = cvxpy.Variable(num_features)  # this is the weight vector
        w.value = np.random.rand(num_features)

        constraints = []
        if self._constraint_type != self._CONS_TYPES.index(None):
            constraints = self._get_constraint_list(
                x,
                y,
                x_control,
                sensitive_attrs_to_cov_thresh,
                self._constraint_type,
                w,
            )

        loss = self._loss_function(w, x, y)

        # take the solution of the unconstrained classifier as starting point
        if self._take_initial_sol:
            p = cvxpy.Problem(cvxpy.Minimize(loss), [])
            p.solve()

        # construct the constraint cvxpy problem
        prob = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

        prob.solve(
            method="dccp",
            tau=self._tau,
            mu=self._mu,
            tau_max=1e10,
            solver=cvxpy.ECOS,
            verbose=False,
            feastol=self._EPS,
            abstol=self._EPS,
            reltol=self._EPS,
            feastol_inacc=self._EPS,
            abstol_inacc=self._EPS,
            reltol_inacc=self._EPS,
            max_iters=self._max_iter,
            max_iter=self._max_iter_dccp,
        )

        # check whether constraints are satisfied
        for f_c in constraints:
            if not f_c.value():
                warnings.warn(
                    "Solver hasn't satisfied constraint."
                    " Make sure that constraints are satisfied empirically."
                    " Alternatively, consider increasing tau parameter"
                )

        w = np.array(w.value).flatten()  # flatten converts it to a 1d array
        return w

    @staticmethod
    def _get_constraint_list(
        x_train,
        y_train,
        x_control_train,
        sensitive_attrs_to_cov_thresh,
        cons_type,
        w,
    ):
        """Get the list of constraints to be fed to the minimizer.

        cons_type == 0: means the whole combined miss classification constraint
            (without FNR or FPR)
        cons_type == 1: FPR constraint
        cons_type == 2: FNR constraint
        cons_type == 4: both FPR as well as FNR constraints

        Args:
            sensitive_attrs_to_cov_thresh (dict): {s: {cov_type: val}},
                s is the sensitive attr cov_type. covariance type. contains the
                covariance for all miss classifications, FPR and for FNR etc

        Returns:
            constraints (list(str)): accuracy constraints in cvxpy format.
                https://www.cvxpy.org/api_reference/cvxpy.constraints.html#
        """
        constraints = []
        for attr in sensitive_attrs_to_cov_thresh.keys():

            attr_arr = x_control_train[attr]
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

            # binary attr, attr_arr_transformed = the attr_arr
            if index_dict is None:
                # constrain type -> sens_attr_val -> total number
                s_val_to_total = {ct: {} for ct in [0, 1, 2]}
                s_val_to_avg = {ct: {} for ct in [0, 1, 2]}

                # sum of entities (females and males) in constraints
                cons_sum_dict = {ct: {} for ct in [0, 1, 2]}

                for v in set(attr_arr):
                    s_val_to_total[0][v] = np.sum(x_control_train[attr] == v)
                    # FPR constraint so we only consider the ground truth
                    # negative dataset for computing the covariance
                    where = np.logical_and(
                        x_control_train[attr] == v, y_train == -1
                    )
                    s_val_to_total[1][v] = np.sum(where)
                    where = np.logical_and(
                        x_control_train[attr] == v, y_train == +1
                    )
                    s_val_to_total[2][v] = np.sum(where)

                for ct in [0, 1, 2]:
                    # N1/N in our formulation, differs from one constraint type
                    # to another
                    denom = s_val_to_total[ct][0] + s_val_to_total[ct][1]
                    s_val_to_avg[ct][0] = s_val_to_total[ct][1] / denom
                    s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0]  # N0/N

                for v in set(attr_arr):
                    idx = x_control_train[attr] == v

                    ###########################################################
                    # #DCCP constraints
                    dist_bound_prod = cvxpy.multiply(
                        y_train[idx], x_train[idx] * w
                    )  # y.f(x)

                    # avg miss classification distance from boundary
                    cons_sum_dict[0][v] = cvxpy.sum(
                        cvxpy.minimum(0, dist_bound_prod)
                    ) * (s_val_to_avg[0][v] / len(x_train))

                    # avg false positive distance from boundary
                    # (only operates on the ground truth neg dataset)
                    cons_sum_dict[1][v] = cvxpy.sum(
                        cvxpy.minimum(
                            0,
                            cvxpy.multiply(
                                (1 - y_train[idx]) / 2.0, dist_bound_prod
                            ),
                        )
                    ) * (s_val_to_avg[1][v] / sum(y_train == -1))

                    # avg false negative distance from boundary
                    cons_sum_dict[2][v] = cvxpy.sum(
                        cvxpy.minimum(
                            0,
                            cvxpy.multiply(
                                (1 + y_train[idx]) / 2.0, dist_bound_prod
                            ),
                        )
                    ) * (s_val_to_avg[2][v] / sum(y_train == +1))
                    ###########################################################

                if cons_type == 4:
                    cts = [1, 2]
                elif cons_type in [0, 1, 2]:
                    cts = [cons_type]

                else:
                    raise Exception("Invalid constraint type")

                ###############################################################
                # DCCP constraints
                for ct in cts:
                    thresh = abs(
                        sensitive_attrs_to_cov_thresh[attr][ct][1]
                        - sensitive_attrs_to_cov_thresh[attr][ct][0]
                    )

                    constraints.append(
                        cons_sum_dict[ct][1] <= cons_sum_dict[ct][0] + thresh
                    )
                    constraints.append(
                        cons_sum_dict[ct][1] >= cons_sum_dict[ct][0] - thresh
                    )

                ###############################################################

            else:  # its categorical, need to set the cov thresh for each value
                # TODO: need to fill up this part
                raise NotImplementedError(
                    "Fill constraint code for categorical"
                    " sensitive features... Exit."
                )
        return constraints

    @staticmethod
    def _get_cov_thresh_dict(cov_thresh, protected_attribute_names):
        """Return dict with covariance thresh for each attribute and error type

        The dict has the form:

            {"sensitive_attribute_name_1":{ 0:{0:cov_thresh, 1:cov_thresh},
                                            1:{0:cov_thresh, 1:cov_thresh},
                                            2:{0:cov_thresh, 1:cov_thresh}},
            "sensitive_attribute_name_2":{  0:{0:cov_thresh, 1:cov_thresh},
                                            1:{0:cov_thresh, 1:cov_thresh},
                                            2:{0:cov_thresh, 1:cov_thresh}},
                ...
            }

        Args:
            cov_thresh (float): covariance threshold (same for each attribute)
            protected_attribute_names (list): list of protected attributes

        Returns:
            sensitive_attrs_to_cov_thresh (dict):
        """
        sensitive_attrs_to_cov_thresh = {}
        for sens_attr_name in protected_attribute_names:
            sensitive_attrs_to_cov_thresh[sens_attr_name] = {
                0: {0: cov_thresh, 1: cov_thresh},
                1: {0: cov_thresh, 1: cov_thresh},
                2: {0: cov_thresh, 1: cov_thresh},
            }

        return sensitive_attrs_to_cov_thresh
