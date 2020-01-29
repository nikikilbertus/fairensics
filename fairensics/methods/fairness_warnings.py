"""Yield warnings when fairness boundaries are violated or data is skewed."""
import warnings

import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


class FairnessBoundsWarning:
    """Raise warnings if classifier misses specified fairness bounds.

    Bounds are checked using AIF360s classification metric if the specified
    bound is not None.
    """

    DISPARATE_IMPACT_RATIO_BOUND = 0.8
    FPR_RATIO_BOUND = 0.8
    FNR_RATIO_BOUND = 0.8
    ERROR_RATIO_BOUND = 0.8

    EO_DIFFERENCE_BOUND = 0.1

    FPR_DIFFERENCE_BOUND = None
    FNR_DIFFERENCE_BOUND = None
    ERROR_DIFFERENCE_BOUND = None

    def __init__(
        self,
        raw_dataset: BinaryLabelDataset,
        predicted_dataset: BinaryLabelDataset,
        privileged_groups=None,
        unprivileged_groups=None,
    ):
        """
        Args:
            raw_dataset (BinaryLabelDataset): Dataset with ground-truth labels.
            predicted_dataset (BinaryLabelDataset): Dataset after predictions.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group.
            unprivileged_groups (list(dict)): Unprivileged groups. Same format
                as privileged_groups.
        """
        self._raw_dataset = raw_dataset
        self._predicted_dataset = predicted_dataset

        if privileged_groups is None:
            privileged_groups = [
                dict(
                    zip(
                        predicted_dataset.protected_attribute_names,
                        predicted_dataset.privileged_protected_attributes,
                    )
                )
            ]

        if unprivileged_groups is None:
            unprivileged_groups = [
                dict(
                    zip(
                        predicted_dataset.protected_attribute_names,
                        predicted_dataset.unprivileged_protected_attributes,
                    )
                )
            ]

        self._classification_metric = ClassificationMetric(
            raw_dataset,
            predicted_dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

    def check_bounds(self):
        """Run methods checking each bound."""
        self._check_disparate_impact()
        self._check_fpr_bound()
        self._check_fnr_bound()
        self._check_all_errors_bound()
        self._check_eo_bound()

    @staticmethod
    def _warn_bound(metric_name, computed_ratio, tolerated_ratio):
        """Raise warning with default message."""
        warning_msg = (
            "Classifier has "
            + metric_name
            + " of : "
            + str(computed_ratio)
            + " above threshold of "
            + str(tolerated_ratio)
        )

        warnings.warn(warning_msg)

    @staticmethod
    def _maybe_scale(num):
        """Return inverse of num if num is larger than one."""
        return num if num < 1 else 1 / num

    def _check_disparate_impact(self):
        """Raise warning if disparate impact bound is breached."""
        if self.DISPARATE_IMPACT_RATIO_BOUND is not None:
            dsp_im = self._maybe_scale(
                self._classification_metric.disparate_impact()
            )

            if dsp_im > self.DISPARATE_IMPACT_RATIO_BOUND:
                self._warn_bound(
                    "disparate impact",
                    dsp_im,
                    self.DISPARATE_IMPACT_RATIO_BOUND,
                )

    def _check_fpr_bound(self):
        """Raise warning if false positive bound is breached."""
        if self.FPR_RATIO_BOUND is not None:
            fprr = self._maybe_scale(
                self._classification_metric.false_positive_rate_ratio()
            )

            if fprr > self.FPR_RATIO_BOUND:
                self._warn_bound(
                    "false positive ratio", fprr, self.FPR_RATIO_BOUND
                )

        if self.FPR_DIFFERENCE_BOUND is not None:
            fprd = self._classification_metric.false_positive_rate_difference()

            if fprd > self.FPR_DIFFERENCE_BOUND:
                self._warn_bound(
                    "false positive rate difference",
                    fprd,
                    self.FPR_DIFFERENCE_BOUND,
                )

    def _check_fnr_bound(self):
        """Raise warning if false negative bound is breached."""
        if self.FNR_RATIO_BOUND is not None:
            fnrr = self._maybe_scale(
                self._classification_metric.false_negative_rate_ratio()
            )

            if fnrr > self.FNR_RATIO_BOUND:
                self._warn_bound(
                    "false negative ratio", fnrr, self.FNR_RATIO_BOUND
                )

        if self.FNR_DIFFERENCE_BOUND is not None:
            fnrd = self._classification_metric.false_positive_rate_difference()
            if fnrd > self.FNR_DIFFERENCE_BOUND:
                self._warn_bound(
                    "false negative rate difference",
                    fnrd,
                    self.FNR_DIFFERENCE_BOUND,
                )

    def _check_all_errors_bound(self):
        """Raise warning if overall error bound is breached."""
        if self.ERROR_RATIO_BOUND is not None:
            err = self._maybe_scale(
                self._classification_metric.error_rate_ratio()
            )

            if err > self.ERROR_RATIO_BOUND:
                self._warn_bound("error ratio", err, self.ERROR_RATIO_BOUND)

        if self.ERROR_DIFFERENCE_BOUND is not None:
            errd = self._classification_metric.error_rate_difference()

            if errd > self.ERROR_DIFFERENCE_BOUND:
                self._warn_bound(
                    "error rate difference", errd, self.ERROR_DIFFERENCE_BOUND
                )

    def _check_eo_bound(self):
        """Raise warning if equalized odds difference is breached."""
        if self.EO_DIFFERENCE_BOUND is not None:
            eo = self._classification_metric.equal_opportunity_difference()

            if eo > self.EO_DIFFERENCE_BOUND:
                self._warn_bound(
                    "true positive rate", eo, self.EO_DIFFERENCE_BOUND
                )


class DataSetSkewedWarning:
    """Raise warning if dataset is skewed with respect to protected attributes.

    Checks are only executed, if the specified bounds are not None.
    """

    POSITIVE_NEGATIVE_CLASS_FRACTION = 0.4
    POSITIVE_NEGATIVE_LABEL_FRACTION = 0.4

    CLASS_LABEL_FRACTION = 0.4

    def __init__(self, dataset: BinaryLabelDataset):
        """
        Args:
            dataset (BinaryLabelDataset): the ground truth data set.
        """
        self._dataset = dataset

    def check_dataset(self):
        """Call methods checking bounds if bounds are specified."""
        if self.POSITIVE_NEGATIVE_CLASS_FRACTION is not None:
            self._check_all_positive_negative_class_ratios()

        if self.POSITIVE_NEGATIVE_LABEL_FRACTION is not None:
            self._check_positive_negative_label_ratio()

        if self.CLASS_LABEL_FRACTION is not None:
            self._check_all_class_label_ratios()

    @staticmethod
    def _check_balance(data_vector, threshold_ratio):
        """Return whether ratio between classes in data_vector is in threshold.

        Args:
            data_vector: 1d vector of categorical features.
            threshold_ratio: tolerated ratio of imbalance between classes.

        Returns:
            True, if ratios of classes in data_vector are within the threshold.
        """
        _groups, class_count = np.unique(data_vector, return_counts=True)

        perfect_balance = 1 / len(class_count)
        tolerated_balance = perfect_balance * threshold_ratio

        class_ratios = class_count / np.sum(class_count)
        balance_diff = abs(class_ratios - perfect_balance)

        if not np.all(balance_diff <= tolerated_balance):
            return False

        return True

    def _check_positive_negative_label_ratio(self):
        """Raise warning if balance of the labels is not within threshold."""
        if not self._check_balance(
            self._dataset.labels, self.POSITIVE_NEGATIVE_LABEL_FRACTION
        ):
            warning_msg = "Ratio between labels is above tolerance of " + str(
                self.POSITIVE_NEGATIVE_LABEL_FRACTION
            )

            warnings.warn(warning_msg)

    def _warn_imbalance(self, prot_attr_idx, tolerated_ratio, msg):
        """Raise warning with standard message."""
        attr_name = self._dataset.protected_attribute_names[prot_attr_idx]

        warning_msg = (
            "Ratio "
            + msg
            + " for protected attribute: "
            + attr_name
            + " is "
            + " above threshold of "
            + str(tolerated_ratio)
        )

        warnings.warn(warning_msg)

    def _check_all_positive_negative_class_ratios(self):
        """Raise warning for ratio between classes of a protected feature.

        For instance, 100 people belong to class 1 but only 10 to class 0.
        """
        for i in range(len(self._dataset.protected_attribute_names)):
            prot_attr = self._dataset.protected_attributes[:, i]

            if not self._check_balance(
                prot_attr, self.POSITIVE_NEGATIVE_CLASS_FRACTION
            ):
                self._warn_imbalance(
                    i,
                    self.POSITIVE_NEGATIVE_CLASS_FRACTION,
                    " between groups ",
                )

    def _check_all_class_label_ratios(self):
        """Check combinations of protected attributes and labels."""

        label_classes = np.unique(self._dataset.labels)

        for i in range(len(self._dataset.protected_attribute_names)):
            prot_attr = self._dataset.protected_attributes[:, i]
            self._check_class_label_ratio(i, prot_attr, label_classes)

    def _check_class_label_ratio(self, i, prot_attr, label_classes):
        """ Raise warning if ratio between feature and labels is too high.

        Args:
            i (int): the index of the protected attribute.
            prot_attr (np.ndarray): 1D array of protected feature.
            label_classes (np.ndarray): 1D array of classes of labels.
        """
        attr_classes = np.unique(prot_attr)
        num_label_class_combinations = len(label_classes) * len(attr_classes)

        perfect_balance = 1 / num_label_class_combinations
        tolerated_imbalance = perfect_balance * self.CLASS_LABEL_FRACTION

        for group in attr_classes:
            for label in label_classes:
                lbl_mask = self._dataset.labels == label
                grp_mask = prot_attr == group
                ratio = np.sum(lbl_mask == grp_mask)
                ratio /= num_label_class_combinations

                if ratio > tolerated_imbalance:
                    self._warn_imbalance(
                        i,
                        self.CLASS_LABEL_FRACTION,
                        " between label and attribute ",
                    )
