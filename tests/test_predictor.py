import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils.testing import assert_raises_regex

from pygbm.binning import BinMapper
from pygbm.plain.grower import TreeGrower
from pygbm import options as O


def make_options():
    return O.OptionSet({
        'learning_rate': O.PositiveFloatOption(default_value=1.0),
        'tree_type': O.StringOption(
            default_value="plain",
            available_options=["plain", "pwl"]
        ),
        'max_iter': O.PositiveIntegerOption(default_value=100),
        'max_leaf_nodes': O.PositiveIntegerOption(default_value=31, none_value=-1),
        'max_depth': O.PositiveIntegerOption(default_value=10, none_value=-1),
        'min_samples_leaf': O.PositiveIntegerOption(default_value=20),
        'w_l2_reg': O.PositiveFloatOption(default_value=1.0),
        'b_l2_reg': O.PositiveFloatOption(default_value=1.0),
        'max_bins': O.PositiveIntegerOption(default_value=255, max_value=256),
        'n_iter_no_change': O.PositiveIntegerOption(default_value=5, none_value=-1),
        'min_gain_to_split': O.PositiveFloatOption(default_value=1e-8),
        'min_hessian_to_split': O.PositiveFloatOption(default_value=1e-8),
        'tol': O.PositiveFloatOption(default_value=1e-7),
        'random_state': O.PositiveIntegerOption(default_value=None),
        'verbose': O.BooleanOption(default_value=False),
        'scoring': O.StringOption(default_value=None),
        'loss': O.StringOption(
            default_value="auto",
            available_options=[
                'binary_crossentropy', 'categorical_crossentropy', 'auto']
        )
    })


def test_pre_binned_data():
    # Make sure ValueError is raised when predictor.predict() is called while
    # the predictor does not have any numerical thresholds.

    X, y = make_regression()

    # Init gradients and hessians to that of least squares loss
    gradients = -y.astype(np.float32)
    hessians = np.ones(1, dtype=np.float32)

    options = make_options()

    mapper = BinMapper(random_state=0)
    X_binned = mapper.fit_transform(X)
    grower = TreeGrower(X_binned, X_binned, gradients, hessians,
                        mapper.n_bins_per_feature_, options)

    grower.grow()
    predictor = grower.make_predictor(
        numerical_thresholds=None
    )

    assert_raises_regex(
        ValueError,
        'This predictor does not have numerical thresholds',
        predictor.predict, X
    )

    assert_raises_regex(
        ValueError,
        'binned_data dtype should be uint8',
        predictor.predict_binned, X
    )

    predictor.predict_binned(X_binned)  # No error

    predictor = grower.make_predictor(
        numerical_thresholds=mapper.numerical_thresholds_
    )
    assert_raises_regex(
        ValueError,
        'X has uint8 dtype',
        predictor.predict, X_binned
    )
