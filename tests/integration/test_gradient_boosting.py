import os
import warnings

import pytest
from sklearn.utils.testing import assert_raises_regex
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from pygbm import GradientBoostingClassifier
from pygbm import GradientBoostingRegressor


X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(random_state=0)


@pytest.mark.parametrize('GradientBoosting, X, y', [
    (GradientBoostingClassifier, X_classification, y_classification),
    (GradientBoostingRegressor, X_regression, y_regression)
])
def test_init_parameters_validation(GradientBoosting, X, y):

    assert_raises_regex(
        ValueError,
        'Incorrect value "blah" for parameter "loss": value must be one of .*',
        GradientBoosting(loss='blah').fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "blah" for parameter "tree_type": value must be one of \[plain, pwl\]',
        GradientBoosting(tree_type='blah').fit, X, y
    )

    for learning_rate in (-1, 0):
        assert_raises_regex(
            ValueError,
            f'Incorrect value "{learning_rate}" for parameter "learning_rate": value must be positive',
            GradientBoosting(learning_rate=learning_rate).fit, X, y
        )

    assert_raises_regex(
        ValueError,
        'Incorrect value "0" for parameter "max_iter": value must not be lesser than 1',
        GradientBoosting(max_iter=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "0" for parameter "max_leaf_nodes": value must not be lesser than 1',
        GradientBoosting(max_leaf_nodes=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "0" for parameter "max_depth": value must not be lesser than 1',
        GradientBoosting(max_depth=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "0" for parameter "min_samples_leaf": value must not be lesser than 1',
        GradientBoosting(min_samples_leaf=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "-1" for parameter "b_l2_reg": value must be positive',
        GradientBoosting(b_l2_reg=-1).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "1" for parameter "max_bins": value must not be lesser than 2',
        GradientBoosting(max_bins=1).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "257" for parameter "max_bins": value must not be larger than 256',
        GradientBoosting(max_bins=257).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "-2" for parameter "n_iter_no_change": value must not be lesser than 1',
        GradientBoosting(n_iter_no_change=-2).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        'Incorrect value "-1" for parameter "tol": value must be positive',
        GradientBoosting(tol=-1).fit, X, y
    )


def test_one_sample_one_feature():
    # Until numba issue #3569 is fixed, we raise an informative error message
    # when X is only one sample or one feature in fit (it's OK in predict).
    # The array is both F and C contiguous, and numba can't compile.
    gb = GradientBoostingClassifier()
    for X, y in (([[1, 2]], [0]), ([[1], [2]], [0, 1])):
        assert_raises_regex(
            ValueError,
            'Passing only one sample or one feature is not supported yet.',
            gb.fit, X, y
        )


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Travis times out without numba")
@pytest.mark.parametrize('scoring, validation_split, n_iter_no_change, tol', [
    ('neg_mean_squared_error', .1, 5, 1e-7),  # use scorer
    ('neg_mean_squared_error', None, 5, 1e-1),  # use scorer on training data
    (None, .1, 5, 1e-7),  # use loss
    (None, None, 5, 1e-1),  # use loss on training data
    (None, None, -1, None),  # no early stopping
])
def test_early_stopping_regression(scoring, validation_split,
                                   n_iter_no_change, tol):

    max_iter = 500

    X, y = make_regression(random_state=0)
    if validation_split is not None:
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42)
        eval_set = (X_test, y_test)
    else:
        eval_set = None

    gb = GradientBoostingRegressor(verbose=True,  # just for coverage
                                   scoring=scoring,
                                   tol=tol,
                                   max_iter=max_iter,
                                   n_iter_no_change=n_iter_no_change,
                                   random_state=0)
    gb.fit(X, y, eval_set=eval_set)

    if n_iter_no_change != - 1:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Travis times out without numba")
@pytest.mark.parametrize('data', (
    make_classification(random_state=0),
    make_classification(n_classes=3, n_clusters_per_class=1, random_state=0)
))
@pytest.mark.parametrize('scoring, validation_split, n_iter_no_change, tol', [
    ('accuracy', .1, 5, 1e-7),  # use scorer
    ('accuracy', None, 5, 1e-1),  # use scorer on training data
    (None, .1, 5, 1e-7),  # use loss
    (None, None, 5, 1e-1),  # use loss on training data
    (None, None, -1, None),  # no early stopping
])
def test_early_stopping_classification(data, scoring, validation_split,
                                       n_iter_no_change, tol):

    max_iter = 500

    X, y = data
    if validation_split is not None:
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42)
        eval_set = (X_test, y_test)
    else:
        eval_set = None

    gb = GradientBoostingClassifier(verbose=True,  # just for coverage
                                    scoring=scoring,
                                    tol=tol,
                                    max_iter=max_iter,
                                    n_iter_no_change=n_iter_no_change,
                                    random_state=0)
    gb.fit(X, y, eval_set=eval_set)

    if n_iter_no_change != -1:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize("n_samples, max_iter, n_iter_no_change, tree_type", [
    (int(1e3), 100, 5, 'pwl'),
    (int(1e3), 100, 5, 'plain')
])
def test_early_stopping_loss(n_samples, max_iter, n_iter_no_change, tree_type):
    # Make sure that when scoring is None, the early stopping is done w.r.t to
    # the loss. Using scoring='neg_log_loss' and scoring=None should be
    # equivalent since the loss is precisely the negative log likelihood

    X, y = make_classification(n_samples, random_state=0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    clf_scoring = GradientBoostingClassifier(max_iter=max_iter,
                                             scoring='neg_log_loss',
                                             n_iter_no_change=n_iter_no_change,
                                             tol=1e-4,
                                             verbose=True,
                                             random_state=0,
                                             tree_type=tree_type)
    clf_scoring.fit(X, y, eval_set=(X_val, y_val))

    clf_loss = GradientBoostingClassifier(max_iter=max_iter,
                                          scoring=None,
                                          n_iter_no_change=n_iter_no_change,
                                          tol=1e-4,
                                          verbose=True,
                                          random_state=0,
                                          tree_type=tree_type)
    clf_loss.fit(X, y, eval_set=(X_val, y_val))

    assert n_iter_no_change < clf_loss.n_iter_ < max_iter
    assert clf_loss.n_iter_ == clf_scoring.n_iter_


def test_should_stop():

    def should_stop(scores, n_iter_no_change, tol, tree_type):
        gbdt = GradientBoostingClassifier(n_iter_no_change=n_iter_no_change,
                                          tol=tol, tree_type=tree_type)
        gbdt._validate_parameters()
        return gbdt._should_stop(scores)

    # not enough iterations
    assert not should_stop([], n_iter_no_change=1, tol=0.001, tree_type='plain')
    assert not should_stop([], n_iter_no_change=1, tol=0.001, tree_type='pwl')

    assert not should_stop([1, 1, 1], n_iter_no_change=5, tol=0.001, tree_type='pwl')
    assert not should_stop([1] * 5, n_iter_no_change=5, tol=0.001, tree_type='pwl')

    # still making significant progress up to tol
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=0.001, tree_type='plain')
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=0.999, tree_type='plain')
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5,
                           tol=5 - 1e-5, tree_type='plain')
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=0.001, tree_type='pwl')
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=0.999, tree_type='pwl')
    assert not should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5,
                           tol=5 - 1e-5, tree_type='pwl')

    # no significant progress according to tol
    assert should_stop([1] * 6, n_iter_no_change=5, tol=0.001, tree_type='plain')
    assert should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=5, tree_type='plain')

    assert should_stop([1] * 6, n_iter_no_change=5, tol=0.001, tree_type='pwl')
    assert should_stop([1, 2, 3, 4, 5, 6], n_iter_no_change=5, tol=5, tree_type='pwl')


# TODO: Remove if / when numba issue 3569 is fixed and check_classifiers_train
# is less strict
def custom_check_estimator(Estimator):
    # Same as sklearn.check_estimator, skipping tests that can't succeed.

    from sklearn.utils.estimator_checks import _yield_all_checks
    from sklearn.utils.testing import SkipTest
    from sklearn.exceptions import SkipTestWarning
    from sklearn.utils import estimator_checks

    estimator = Estimator
    name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        if (check is estimator_checks.check_fit2d_1feature or
                check is estimator_checks.check_fit2d_1sample):
            # X is both Fortran and C aligned and numba can't compile.
            # Opened numba issue 3569
            continue
        if check is estimator_checks.check_classifiers_train:
            continue  # probas don't exactly sum to 1 (very close though)
        if (hasattr(check, 'func') and
                check.func is estimator_checks.check_classifiers_train):
            continue  # same, wrapped in a functools.partial object.

        try:
            check(name, estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Potentially long")
@pytest.mark.parametrize('Estimator', (
    GradientBoostingRegressor(),
    GradientBoostingClassifier(n_iter_no_change=None, min_samples_leaf=5),))
def test_estimator_checks(Estimator):
    # Run the check_estimator() test suite on GBRegressor and GBClassifier.

    # Notes:
    # - Can't do early stopping with classifier because often
    #   validation_split=.1 leads to test_size=2 < n_classes and
    #   train_test_split raises an error.
    # - Also, need to set a low min_samples_leaf for
    #   check_classifiers_classes() to pass: with only 30 samples on the
    #   dataset, the root is never split with min_samples_leaf=20 and only the
    #   majority class is predicted.
    custom_check_estimator(Estimator)
