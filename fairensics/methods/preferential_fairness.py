"""Wrapper for PreferentialFairness from fair-classification.

Besides fit() and predict() the class contains three functions:

All three functions are part of the original code:
    _train_model_pref_fairness()
    _get_di_cons_single_boundary()
    _get_preferred_cons()

Original code:
    https://github.com/mbilalzafar/fair-classification

"""
import warnings

import cvxpy
import numpy as np
from aif360.algorithms import Transformer
from aif360.datasets.binary_label_dataset import BinaryLabelDataset

from .fairness_warnings import FairnessBoundsWarning, DataSetSkewedWarning
from .utils import add_intercept, LossFunctions
from ..fairensics_utils import get_unprotected_attributes


class PreferentialFairness(Transformer):
    """Train separate classifier clf_z for each group of protected attribute z.
    Loss "L" defines whether a logistic regression or a liner svm is trained.

    Minimize
        L(w)

    Subject to
        sum(predictions_z) > sum(predictions_z')

    Where
        predictions_z are the predictions using group zs classifier clf_z
        predictions_z' are the predictions using group z's classifier clf_z'

    Example:
        https://github.com/nikikilbertus/fairensics/blob/master/examples/2_3_fair-classification-preferential-fairness-example.ipynb
    """

    # original function takes an integer for the constraint. Index in the list
    # corresponds to the integer value of each constraint.
    _CONS_TYPES = ["parity", "impact", "treatment", "both", None]

    def __init__(
        self,
        loss_function=LossFunctions.NAME_LOG_REG,
        constraint_type=None,
        train_multiple=False,
        lam=None,
        warn=True,
        tau=0.5,
        mu=1.2,
        EPS=1e-4,
        max_iter=100,
        max_iter_dccp=50,
    ):
        """
        Args:
            loss_function (str): name of loss function defined in utils.
            constraint_type (str): one of the values in _CONS_TYPE.
            train_multiple (bool): if true, a classifier for each group of
                protected attribute is trained
            lam (dict, optional): ...
            warn (bool): if true, warnings are raised on certain bounds.
            tau, mu, EPS, max_iter, max_iter_dccp: solver related parameters.
        """
        assert constraint_type in self._CONS_TYPES

        if lam is None:
            if not train_multiple:
                lam = 0.0
            else:
                lam = {0: 0.0, 1: 0.0}

        else:
            if train_multiple:
                assert isinstance(lam, dict)

        super(PreferentialFairness, self).__init__()

        self._loss_function = LossFunctions.get_cvxpy_loss_function(
            loss_function
        )
        self._lam = lam
        self._train_multiple = train_multiple
        self._warn = warn
        self._tau = tau
        self._mu = mu
        self._EPS = EPS
        self._max_iter = max_iter
        self._max_iter_dccp = max_iter_dccp
        self._initialized = False
        self._params = {}
        self._constraint_type = self._CONS_TYPES.index(constraint_type)

    def fit(
        self,
        dataset: BinaryLabelDataset,
        s_val_to_cons_sum=None,
        prot_attr_ind=0,
    ):
        """Fits the model.

        Args:
            dataset: AIF360 data set.
            s_val_to_cons_sum (dict): the ramp approximation, only needed for
                _constraint_type 1 and 3.
            prot_attr_ind (int): index of the protected feature to apply
                constraints to.
        """
        if self._warn:
            dataset_warning = DataSetSkewedWarning(dataset)
            dataset_warning.check_dataset()

        # 1 and 3 are the constraints
        if s_val_to_cons_sum is None and self._constraint_type in [1, 3]:
            parity_clf = PreferentialFairness(
                constraint_type="parity", lam=0.01
            )
            parity_clf.fit(dataset)

            _, dist_dict = parity_clf._get_distance_boundary(
                add_intercept(get_unprotected_attributes(dataset)),
                dataset.protected_attributes[:, prot_attr_ind],
            )

            s_val_to_cons_sum = parity_clf._get_sensitive_attr_cov(dist_dict)

        # map labels to -1 and 1
        temp_labels = dataset.labels.copy()

        idx = np.array(dataset.labels == dataset.favorable_label).ravel()
        temp_labels[idx, 0] = 1.0
        idx = np.array(dataset.labels == dataset.unfavorable_label).ravel()
        temp_labels[idx, 0] = -1.0

        if self._train_multiple:
            obj_function = self._train_multi_model_pref_fairness
        else:
            obj_function = self._train_single_model_pref_fairness

        self._params["w"] = obj_function(
            add_intercept(get_unprotected_attributes(dataset)),
            temp_labels.ravel(),
            dataset.protected_attributes[:, 0],
            s_val_to_cons_sum,
        )

        self._initialized = True
        return self

    def predict(self, dataset: BinaryLabelDataset):
        """Make predictions.

        Args:
            dataset: either AIF360 data set or np.ndarray.

        Returns:
            either AIF360 data set or np.ndarray if dataset is a np.ndarray.
        """
        if not self._initialized:
            raise ValueError("Model not initialized. Run fit() first.")

        # TODO: ok?
        if isinstance(dataset, np.ndarray):
            raise NotImplementedError

        dataset_new = dataset.copy(deepcopy=True)
        dist_arr, _ = self._get_distance_boundary(
            add_intercept(get_unprotected_attributes(dataset)),
            dataset.protected_attributes[:, 0],
        )

        dataset_new.labels = np.sign(dist_arr)

        # Map the dataset labels to back to their original values.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset_new.labels == 1.0)] = dataset.favorable_label
        temp_labels[(dataset_new.labels == -1.0)] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        if self._warn:
            bound_warnings = FairnessBoundsWarning(dataset, dataset_new)
            bound_warnings.check_bounds()

        return dataset_new

    def _train_single_model_pref_fairness(
        self, X, y, x_sensitive, s_val_to_cons_sum
    ):
        """Train a single model for all groups.

        Args:
            X (np.ndarray): 2D n x d array, the unprotected features.
            y (np.ndarray): 1D, n length vector, the labels.
            x_sensitive (np.ndarray): 1D, (n, ), the protected attribute.
            s_val_to_cons_sum: The ramp approximation, only needed for
                _constraint_type 1 and 3.

        Returns:
            w (np.ndarray): 1D Array, the learned weights.
        """

        w = cvxpy.Variable(X.shape[1])  # this is the weight vector
        w.value = np.random.rand(X.shape[1])

        obj = 0
        # regularizer -- first term in w is intercept -> no regularization
        obj += cvxpy.sum_squares(w[1:]) * self._lam
        obj += self._loss_function(w, X, y)

        constraints = self._get_constraint_list(
            w, X, y, x_sensitive, s_val_to_cons_sum
        )
        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)

        prob.solve(
            method="dccp",
            tau=self._tau,
            mu=self._mu,
            tau_max=1e10,
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

        # check that the fairness constraint is satisfied
        for f_c in constraints:
            if not f_c.value():
                warnings.warn(
                    "Solver hasn't satisfied constraint."
                    " Make sure that constraints are satisfied empirically."
                    " Alternatively, consider increasing tau parameter"
                )

        return np.array(w.value).flatten()  # flatten converts it to a 1d array

    def _train_multi_model_pref_fairness(
        self, X, y, x_sensitive, s_val_to_cons_sum
    ):
        """Train a separate classifier for each group k in protected attribute.

        Args:
            X (np.ndarray): 2D n x d array, the unprotected features.
            y (np.ndarray): 1D, n length vector, the labels.
            x_sensitive (np.ndarray): 1D, (n, ), the protected attribute.
            s_val_to_cons_sum: The ramp approximation, only needed for
                _constraint_type 1 and 3.

        Returns:
            w (dict): dict of 1D weight arrays learned for each group k in
                protected attribute.
        """
        w = {}
        for k in set(x_sensitive):
            w[k] = cvxpy.Variable(X.shape[1])  # this is the weight vector
            w[k].value = np.random.rand(
                X.shape[1]
            )  # initialize the value of w

        num_all = X.shape[0]  # set of all data points

        obj = 0
        for k in set(x_sensitive):
            idx = x_sensitive == k
            X_k = X[idx]
            y_k = y[idx]

            # first term in w is the intercept, so no need to regularize that
            obj += cvxpy.sum_squares(w[k][1:]) * self._lam[k]
            obj += self._loss_function(w[k], X_k, y_k, num_all)
            # notice that we are dividing by the length of the whole dataset,
            # and not just of this group.
            # the group that has more people contributes more to the loss

        constraints = self._get_constraint_list(
            w, X, y, x_sensitive, s_val_to_cons_sum
        )
        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)

        prob.solve(
            method="dccp",
            tau=self._tau,
            mu=self._mu,
            tau_max=1e10,
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

        # check that the fairness constraint is satisfied
        for f_c in constraints:
            if not f_c.value():
                warnings.warn(
                    "Solver hasn't satisfied constraint."
                    " Make sure that constraints are satisfied empirically."
                    " Alternatively, consider increasing tau parameter"
                )

        w_return = {}
        for k in set(x_sensitive):
            w_return[k] = np.array(w[k].value).flatten()

        return w_return

    def _get_constraint_list(self, w, X, y, x_sensitive, s_val_to_cons_sum):
        """Return list of constraints.

        Args:
            w: the learned weights for each group k in protected attribute.
            X, y, x_sensitive, s_val_to_cons_sum: see
                _train_multi_model_pref_fairness

        Returns:

        """
        if self._constraint_type == 0:  # disp imp with single boundary
            cov_thresh = np.abs(0.0)  # perfect fairness
            return self._get_parity_cons(X, y, x_sensitive, w, cov_thresh)

        # preferred imp, pref imp + pref treat
        elif self._constraint_type in [1, 3]:
            return self._get_preferred_cons(
                X, x_sensitive, w, s_val_to_cons_sum
            )

        elif self._constraint_type == 2:
            return self._get_preferred_cons(X, x_sensitive, w)

        return []

    def _decision_function(self, X, k=None):
        """Makes predictions for all samples in X.

        Args:
            X (np.ndarary): [n_samples, n_features], the input samples.
            k: the group whose decision boundary should be used.
                None means that we trained one clf for the whole dataset.

        Returns:
            (np.ndarray), 1D, the predictions
        """

        if k is None:
            ret = np.dot(X, self._params["w"])
        else:
            ret = np.dot(X, self._params["w"][k])

        return ret

    def _get_distance_boundary(self, X, x_sensitive):
        """Returns the distance boundary for each sample.

        Args:
            X: (np.ndarary): [n_samples, n_features], the input samples.
            x_sensitive (np.ndarary): 1D, the protected feature

        Returns:
            distance_boundary_arr (np.ndarary): distance to boundary, each
                groups owns w is applied on it
            distance_boundary_dict (dict): dict of the form s_attr_group
                (points from group 0/1)
                -> w_group (boundary of group 0/1)
                -> distances for this group with this boundary
        """
        # s_attr_group (0/1) -> w_group (0/1) -> distances
        distances_boundary_dict = {}
        # we have one model for the whole data
        if not isinstance(self._params["w"], dict):
            distance_boundary_arr = self._decision_function(X)

            # there is only one boundary, so the results with this_group and
            # other_group boundaries are the same
            for attr in set(x_sensitive):
                distances_boundary_dict[attr] = {}
                idx = x_sensitive == attr

                for k in set(x_sensitive):
                    # apply same decision function for all the sensitive attrs
                    # because same w is trained for everyone
                    distances_boundary_dict[attr][k] = self._decision_function(
                        X[idx]
                    )

        else:  # w is a dict
            distance_boundary_arr = np.zeros(X.shape[0])

            for attr in set(x_sensitive):

                distances_boundary_dict[attr] = {}
                idx = x_sensitive == attr
                X_g = X[idx]

                # each group gets decision with their own boundary
                distance_boundary_arr[idx] = self._decision_function(X_g, attr)

                for k in self._params["w"].keys():
                    # each group gets a decision with both boundaries
                    distances_boundary_dict[attr][k] = self._decision_function(
                        X_g, k
                    )

        return distance_boundary_arr, distances_boundary_dict

    @staticmethod
    def _get_sensitive_attr_cov(dist_dict):
        """Compute ramp function for each group to estimate acceptance rate.

        Args:
            dist_dict (dict): distance to decision boundary as returned by
                _get_distance_boundary().

        Returns:
            s_val_to_cons_sum (dict): ...
        """
        # {0: {}, 1: {}}  # s_attr_group (0/1) -> w_group (0/1) -> ramp approx
        s_val_to_cons_sum = {}

        for s_val in dist_dict.keys():
            s_val_to_cons_sum[s_val] = {}

            for w_group in dist_dict[s_val].keys():
                fx = dist_dict[s_val][w_group]
                s_val_to_cons_sum[s_val][w_group] = (
                    np.sum(np.maximum(0, fx)) / fx.shape[0]
                )

        return s_val_to_cons_sum

    def _get_preferred_cons(self, X, x_sensitive, w, s_val_to_cons_sum=None):
        """Get constraint list.

        No need to pass s_val_to_cons_sum for preferred treatment (envy free)
        constraints. For details on cons_type, see the documentation of fit()
        function.

        Args:
            X, x_sensitive, w, s_val_to_cons_sum: see _get_constraint_list()

        Returns:
            constraints (list(str)): constraints in cvxpy format.
                https://www.cvxpy.org/api_reference/cvxpy.constraints.html#
        """
        constraints = []
        # 1 - pref imp, 2 - EF, 3 - pref imp & EF
        if self._constraint_type in [1, 2, 3]:

            prod_dict = {}  # s_attr_group (0/1) -> w_group (0/1) -> val
            for val in set(x_sensitive):
                prod_dict[val] = {}
                idx = x_sensitive == val
                X_g = X[idx]
                num_g = X_g.shape[0]

                for k in w.keys():  # get the distance with each group's w
                    prod_dict[val][k] = (
                        cvxpy.sum(cvxpy.multiply(0, X_g * w[k])) / num_g
                    )
        else:
            raise Exception("Invalid constraint type")

        # 1 for preferred impact -- 3 for preferred impact and envy free
        if self._constraint_type == 1 or self._constraint_type == 3:
            constraints.append(prod_dict[0][0] >= s_val_to_cons_sum[0][0])
            constraints.append(prod_dict[1][1] >= s_val_to_cons_sum[1][1])
        # envy free
        if self._constraint_type == 2 or self._constraint_type == 3:
            constraints.append(prod_dict[0][0] >= prod_dict[0][1])
            constraints.append(prod_dict[1][1] >= prod_dict[1][0])

        return constraints

    @staticmethod
    def _get_parity_cons(X, _y, x_sensitive, w, cov_thresh):
        """Get parity impact constraint list.

        Args:
            X, x_sensitive, w, cov_thresh: see _get_constraint_list() method.

        Returns:
            constraints (list(str)): constraints in cvxpy format.
                https://www.cvxpy.org/api_reference/cvxpy.constraints.html#
        """
        # covariance thresh has to be a small positive number
        assert cov_thresh >= 0

        constraints = []
        z_i_z_bar = x_sensitive - np.mean(x_sensitive)

        fx = X * w

        prod = cvxpy.sum(cvxpy.multiply(z_i_z_bar, fx)) / X.shape[0]

        constraints.append(prod <= cov_thresh)
        constraints.append(prod >= -cov_thresh)

        return constraints
