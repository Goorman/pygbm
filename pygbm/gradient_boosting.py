"""
Gradient Boosting decision trees for classification and regression.
"""
from abc import ABC, abstractmethod

import numpy as np
from time import time
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import check_scoring
from sklearn.preprocessing import LabelEncoder

from pygbm.plain.grower import TreeGrower as PlainTreeGrower
from pygbm.pwl.grower import TreeGrower as PWLTreeGrower
from pygbm.loss import _LOSSES
from pygbm.dataset import Dataset
from pygbm import options as O
from typing import Union, Dict


class BaseGradientBoostingMachine(BaseEstimator, ABC):
    """Base class for gradient boosting estimators."""
    train_validation_subsample_size = 10000

    @abstractmethod
    def __init__(self, loss=None, tree_type=None, learning_rate=None, max_iter=None, max_leaf_nodes=None,
                 max_depth=None, min_samples_leaf=None, w_l2_reg=None, b_l2_reg=None, max_bins=None,
                 scoring=None, n_iter_no_change=None, tol=None, verbose=None, random_state=None):
        self.loss = loss
        self.tree_type = tree_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.w_l2_reg = w_l2_reg
        self.b_l2_reg = b_l2_reg
        self.max_bins = max_bins
        self.n_iter_no_change = n_iter_no_change
        self.scoring = scoring
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    @property
    def parameter_dict(self) -> Dict[str, O.Option]:
        pass

    def _validate_parameters(self):
        self.options = O.OptionSet(self.parameter_dict)
        self.options.update_from_estimator(self)

    def fit(self, X: Union[np.array, Dataset], y: np.array = None):

        self._validate_parameters()

        rng = check_random_state(self.random_state)

        if not isinstance(X, Dataset):
            dataset = Dataset(X, y, max_bins=self.max_bins, verbose=self.options['verbose'], random_state=self.options['random_state'])
            self.n_features_ = dataset.shape[1]
        else:
            dataset = X
            self.n_features_ = dataset.shape[1]

        y = self._encode_y(dataset.y)

        self.loss_ = self._get_loss()

        do_early_stopping = (self.options['n_iter_no_change'] is not None and
                             self.options['n_iter_no_change'] > 0)

        X_binned_train, y_train, X_train = dataset.X_binned, y, dataset.X
        X_binned_val, y_val, X_val = None, None, None

        # Subsample the training set for score-based monitoring.
        if do_early_stopping:

            n_samples_train = X_binned_train.shape[0]
            if n_samples_train > self.train_validation_subsample_size:
                indices = rng.choice(X_binned_train.shape[0], self.train_validation_subsample_size)
                X_small_train = X_train[indices]
                y_small_train = y_train[indices]
            else:
                X_small_train = X_train
                y_small_train = y_train
            # Predicting is faster of C-contiguous arrays.
            X_small_train = np.ascontiguousarray(X_small_train)

        if self.verbose:
            print("Fitting gradient boosted rounds:")

        n_samples = X_binned_train.shape[0]
        self.baseline_prediction_ = self.loss_.get_baseline_prediction(
            y_train, self.n_trees_per_iteration_)
        # raw_predictions are the accumulated values predicted by the trees
        # for the training data.
        raw_predictions = np.zeros(
            shape=(n_samples, self.n_trees_per_iteration_),
            dtype=self.baseline_prediction_.dtype
        )
        raw_predictions += self.baseline_prediction_

        # gradients and hessians are 1D arrays of size
        # n_samples * n_trees_per_iteration
        gradients, hessians = self.loss_.init_gradients_and_hessians(
            n_samples=n_samples,
            prediction_dim=self.n_trees_per_iteration_
        )

        # predictors_ is a matrix of TreePredictor objects with shape
        # (n_iter_, n_trees_per_iteration)
        self.predictors_ = predictors = []

        # scorer_ is a callable with signature (est, X, y) and calls
        # est.predict() or est.predict_proba() depending on its nature.
        self.scorer_ = check_scoring(self, self.options['scoring'])
        self.train_scores_ = []
        self.validation_scores_ = []
        if do_early_stopping:
            # Add predictions of the initial model (before the first tree)
            self.train_scores_.append(
                self._get_scores(X_train, y_train))

# TODO: ADD VALIDATION
#            if self.validation_split is not None:
#                self.validation_scores_.append(
#                    self._get_scores(X_val, y_val))

        fit_start_time = time()
        acc_find_split_time = 0.  # time spent finding the best splits
        acc_apply_split_time = 0.  # time spent splitting nodes
        acc_prediction_time = 0. # time spent predicting X for gradient and hessians update

        for iteration in range(self.options['max_iter']):

            if self.options['verbose']:
                iteration_start_time = time()
                print(f"[{iteration + 1}/{self.options['max_iter']}] ", end='',
                      flush=True)

            # Update gradients and hessians, inplace
            self.loss_.update_gradients_and_hessians(gradients, hessians,
                                                     y_train, raw_predictions)

            predictors.append([])
            # Build `n_trees_per_iteration` trees.
            for k, (gradients_at_k, hessians_at_k) in enumerate(zip(
                    np.array_split(gradients, self.n_trees_per_iteration_),
                    np.array_split(hessians, self.n_trees_per_iteration_))):
                # the xxxx_at_k arrays are **views** on the original arrays.
                # Note that for binary classif and regressions,
                # n_trees_per_iteration is 1 and xxxx_at_k is equivalent to the
                # whole array.

                if self.options['tree_type'] == 'pwl':
                    tree_grower_cls = PWLTreeGrower
                elif self.options['tree_type'] == 'plain':
                    tree_grower_cls = PlainTreeGrower
                else:
                    raise NotImplementedError

                grower = tree_grower_cls(
                    X_binned_train, X_train, gradients_at_k, hessians_at_k, self.options
                )
                grower.grow()

                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time

                predictor = grower.make_predictor(dataset.numerical_thresholds)
                predictors[-1].append(predictor)

                tic_pred = time()

                grower.update_raw_predictions(raw_predictions[:, k])

                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred

            should_early_stop = False
            if do_early_stopping:
                should_early_stop = self._check_early_stopping(
                    X_small_train, y_small_train,
                    X_val, y_val)

            if self.options['verbose']:
                self._print_iteration_stats(iteration_start_time,
                                            do_early_stopping)

            if should_early_stop:
                break

        if self.options['verbose']:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self.predictors_
                for predictor in predictors_at_ith_iteration)
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self.predictors_)
            print(f"Fit {n_predictors} trees in {duration:.3f} s, "
                  f"({n_total_leaves} total leaves)")
            print(f"{'Time spent finding best splits:':<32} "
                  f"{acc_find_split_time:.3f}s")
            print(f"{'Time spent applying splits:':<32} "
                  f"{acc_apply_split_time:.3f}s")
            print(f"{'Time spent predicting:':<32} "
                  f"{acc_prediction_time:.3f}s")

        self.train_scores_ = np.asarray(self.train_scores_)
        self.validation_scores_ = np.asarray(self.validation_scores_)
        return self

    def _check_early_stopping(self, X_train, y_train,
                              X_val, y_val):
        """Check if fitting should be early-stopped.

        Scores are computed on validation data or on training data.
        """

        self.train_scores_.append(
            self._get_scores(X_train, y_train))
# TODO: ADD VALIDATION
#        if self.validation_split is not None:
#            self.validation_scores_.append(
#                self._get_scores(X_val, y_val))
#            return self._should_stop(self.validation_scores_)

        return self._should_stop(self.train_scores_)

    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # A higher score is always better. Higher tol means that it will be
        # harder for subsequent iteration to be considered an improvement upon
        # the reference score, and therefore it is more likely to early stop
        # because of the lack of significant improvement.
        tol = 0 if self.tol is None else self.tol
        reference_score = scores[-reference_position] + tol
        recent_scores = scores[-reference_position + 1:]
        recent_improvements = [score > reference_score
                               for score in recent_scores]
        return not any(recent_improvements)

    def _get_scores(self, X, y):
        """Compute scores on data X with target y.

        Scores are either computed with a scorer if scoring parameter is not
        None, else with the loss. As higher is always better, we return
        -loss_value.
        """
        if self.scoring is not None:
            return self.scorer_(self, X, y)

        # Else, use the negative loss as score.
        raw_predictions = self._raw_predict(X)
        return -self.loss_(y, raw_predictions)

    def _print_iteration_stats(self, iteration_start_time, do_early_stopping):
        """Print info about the current fitting iteration."""
        log_msg = ''

        predictors_of_ith_iteration = [
            predictors_list for predictors_list in self.predictors_[-1]
            if predictors_list
        ]
        n_trees = len(predictors_of_ith_iteration)
        max_depth = max(predictor.get_max_depth()
                        for predictor in predictors_of_ith_iteration)
        n_leaves = sum(predictor.get_n_leaf_nodes()
                       for predictor in predictors_of_ith_iteration)

        if n_trees == 1:
            log_msg += (f"{n_trees} tree, {n_leaves} leaves, ")
        else:
            log_msg += (f"{n_trees} trees, {n_leaves} leaves ")
            log_msg += (f"({int(n_leaves / n_trees)} on avg), ")

        log_msg += f"max depth = {max_depth}, "

        if do_early_stopping:
            log_msg += f"{self.scoring} train: {self.train_scores_[-1]:.5f}, "
# TODO: ADD VALIDATION
#            if self.validation_split is not None:
#                log_msg += (f"{self.scoring} val: "
#                            f"{self.validation_scores_[-1]:.5f}, ")

        iteration_time = time() - iteration_start_time
        log_msg += f"in {iteration_time:0.3f}s"

        print(log_msg)

    def _raw_predict(self, X, n_trees=None):
        """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples. If ``X.dtype == np.uint8``, the data is assumed
            to be pre-binned and the estimator must have been fitted with
            pre-binned data.

        Returns
        -------
        raw_predictions : array, shape (n_samples * n_trees_per_iteration,)
            The raw predicted values.
        """
        X = check_array(X)
        check_is_fitted(self, 'predictors_')
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f'X has {X.shape[1]} features but this estimator was '
                f'trained with {self.n_features_} features.'
            )

        n_samples = X.shape[0]
        raw_predictions = np.zeros(
            shape=(n_samples, self.n_trees_per_iteration_),
            dtype=self.baseline_prediction_.dtype
        )
        raw_predictions += self.baseline_prediction_

        if n_trees is None or n_trees <= 0 or n_trees > len(self.predictors_) or not isinstance(n_trees, int):
            predictors = self.predictors_
        else:
            predictors = self.predictors_[:n_trees]
        # Should we parallelize this?
        for predictors_of_ith_iteration in predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                raw_predictions[:, k] += predictor.predict(X)

        return raw_predictions

    @abstractmethod
    def _get_loss(self):
        pass

    @abstractmethod
    def _encode_y(self, y):
        pass

    @property
    def n_iter_(self):
        check_is_fitted(self, 'predictors_')
        return len(self.predictors_)


class GradientBoostingRegressor(BaseGradientBoostingMachine, RegressorMixin):
    """Scikit-learn compatible Gradient Boosting Tree for regression.

    Parameters
    ----------
    loss : {'least_squares'}, optional(default='least_squares')
        The loss function to use in the boosting process.
    learning_rate : float, optional(default=0.1)
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, optional(default=100)
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int or None, optional(default=None)
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
    max_depth : int or None, optional(default=None)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf.
    min_samples_leaf : int, optional(default=20)
        The minimum number of samples per leaf.
    max_bins : int, optional(default=256)
        The maximum number of bins to use. Before training, each feature of
        the input array ``X`` is binned into at most ``max_bins`` bins, which
        allows for a much faster training stage. Features with a small
        number of unique values may use less than ``max_bins`` bins. Must be no
        larger than 256.
    scoring : str or callable or None, \
        optional (default=None)
        Scoring parameter to use for early stopping (see sklearn.metrics for
        available options). If None, early stopping is check w.r.t the loss
        value.
    n_iter_no_change : int or None, optional (default=5)
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1``th-to-last one, up to some
        tolerance. If None or 0, no early-stopping is done.
    tol : float or None optional (default=1e-7)
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    verbose: int, optional (default=0)
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, np.random.RandomStateInstance or None, \
        optional (default=None)
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled. See
        `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_.


    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from pygbm import GradientBoostingRegressor
    >>> X, y = load_boston(return_X_y=True)
    >>> est = GradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.92...
    """

    def parameter_dict(self):
        return {
            'learning_rate': O.PositiveFloatOption(default_value=1.0),
            'tree_type': O.StringOption(
                default_value="pwl",
                available_options=["plain", "pwl"]
            ),
            'max_iter': O.PositiveIntegerOption(default_value=100),
            'max_leaf_nodes': O.PositiveIntegerOption(default_value=31, none_value=-1),
            'max_depth': O.PositiveIntegerOption(default_value=10, none_value=-1),
            'min_samples_leaf': O.PositiveIntegerOption(default_value=20),
            'w_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'b_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'max_bins': O.PositiveIntegerOption(default_value=255, max_value=255),
            'n_iter_no_change': O.PositiveIntegerOption(default_value=5, none_value=-1),
            'tol': O.PositiveFloatOption(default_value=1e-7),
            'random_state': O.PositiveIntegerOption(default_value=None),
            'verbose': O.BooleanOption(default_value=False),
            'scoring': O.StringOption(default_value=None),
            'loss': O.StringOption(
                default_value="least_squares",
                available_options=['least_squares']
            )
        }

    def predict(self, X, n_trees=None):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples. If ``X.dtype == np.uint8``, the data is assumed
            to be pre-binned and the estimator must have been fitted with
            pre-binned data.
        n_trees : Optional[int]
            Number of decision trees to use for prediction.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        # Return raw predictions after converting shape
        # (n_samples, 1) to (n_samples,)
        return self._raw_predict(X, n_trees).ravel()

    def _encode_y(self, y):
        # Just convert y to float32
        self.n_trees_per_iteration_ = 1
        y = y.astype(np.float32, copy=False)
        return y

    def _get_loss(self):
        return _LOSSES[self.loss]()


class GradientBoostingClassifier(BaseGradientBoostingMachine, ClassifierMixin):
    """Scikit-learn compatible Gradient Boosting Tree for classification.

    Parameters
    ----------
    loss : {'auto', 'binary_crossentropy', 'categorical_crossentropy'}, \
        optional(default='auto')
        The loss function to use in the boosting process. 'binary_crossentropy'
        (also known as logistic loss) is used for binary classification and
        generalizes to 'categorical_crossentropy' for multiclass
        classification. 'auto' will automatically choose either loss depending
        on the nature of the problem.
    learning_rate : float, optional(default=1)
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, optional(default=100)
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    max_leaf_nodes : int or None, optional(default=None)
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
    max_depth : int or None, optional(default=None)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf.
    min_samples_leaf : int, optional(default=20)
        The minimum number of samples per leaf.
    max_bins : int, optional(default=256)
        The maximum number of bins to use. Before training, each feature of
        the input array ``X`` is binned into at most ``max_bins`` bins, which
        allows for a much faster training stage. Features with a small
        number of unique values may use less than ``max_bins`` bins. Must be no
        larger than 256.
    scoring : str or callable or None, optional (default=None)
        Scoring parameter to use for early stopping (see sklearn.metrics for
        available options). If None, early stopping is check w.r.t the loss
        value.
    n_iter_no_change : int or None, optional (default=5)
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1``th-to-last one, up to some
        tolerance. If None or 0, no early-stopping is done.
    tol : float or None optional (default=1e-7)
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    verbose: int, optional(default=0)
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, np.random.RandomStateInstance or None, \
        optional(default=None)
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled. See `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from pygbm import GradientBoostingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = GradientBoostingClassifier().fit(X, y)
    >>> clf.score(X, y)
    0.97...
    """

    @property
    def parameter_dict(self):
        return {
            'learning_rate': O.PositiveFloatOption(default_value=1.0),
            'tree_type': O.StringOption(
                default_value="pwl",
                available_options=["plain", "pwl"]
            ),
            'max_iter': O.PositiveIntegerOption(default_value=100),
            'max_leaf_nodes': O.PositiveIntegerOption(default_value=31, none_value=-1),
            'max_depth': O.PositiveIntegerOption(default_value=10, none_value=-1),
            'min_samples_leaf': O.PositiveIntegerOption(default_value=20),
            'w_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'b_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'max_bins': O.PositiveIntegerOption(default_value=255, max_value=255),
            'n_iter_no_change': O.PositiveIntegerOption(default_value=5, none_value=-1),
            'tol': O.PositiveFloatOption(default_value=1e-7),
            'random_state': O.PositiveIntegerOption(default_value=None),
            'verbose': O.BooleanOption(default_value=False),
            'scoring': O.StringOption(default_value=None),
            'loss': O.StringOption(
                default_value="auto",
                available_options=[
                    'binary_crossentropy', 'categorical_crossentropy', 'auto']
            )
        }

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples. If ``X.dtype == np.uint8``, the data is assumed
            to be pre-binned and the estimator must have been fitted with
            pre-binned data.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted classes.
        """
        # This could be done in parallel
        encoded_classes = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded_classes]

    def predict_proba(self, X, n_trees=None):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples. If ``X.dtype == np.uint8``, the data is assumed
            to be pre-binned and the estimator must have been fitted with
            pre-binned data.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        raw_predictions = self._raw_predict(X, n_trees)
        return self.loss_.predict_proba(raw_predictions)

    def _encode_y(self, y):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y.astype(np.float32, copy=False)
        return encoded_y

    def _get_loss(self):
        if self.loss == 'auto':
            if self.n_trees_per_iteration_ == 1:
                return _LOSSES['binary_crossentropy']()
            else:
                return _LOSSES['categorical_crossentropy']()

        return _LOSSES[self.loss]()
