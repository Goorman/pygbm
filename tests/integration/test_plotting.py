import numpy as np
from sklearn.datasets import make_classification
import pytest
from pygbm.binning import BinMapper
from pygbm.dataset import Dataset
from pygbm.plain.grower import TreeGrower
from pygbm import GradientBoostingRegressor
from pygbm import GradientBoostingClassifier
from pygbm.loss import BinaryCrossEntropy
from pygbm.options import OptionSet


@pytest.fixture
def classification_data():
    return make_classification(n_samples=150, n_classes=2, n_features=5,
                           n_informative=3, n_redundant=0,
                           random_state=0)


def test_plot_grower(tmpdir, classification_data):
    pytest.importorskip('graphviz')
    from pygbm.plotting import plot_tree

    dataset = Dataset(classification_data[0], classification_data[1])
    n_trees_per_iteration = 1
    loss = BinaryCrossEntropy()

    clf = GradientBoostingClassifier()

    gradients, hessians = loss.init_gradients_and_hessians(
        n_samples=dataset.shape[0],
        prediction_dim=n_trees_per_iteration
    )
    y = clf._encode_y(dataset.y)
    baseline_prediction_ = loss.get_baseline_prediction(
        y, 1)
    raw_predictions = np.zeros(
        shape=(dataset.shape[0], n_trees_per_iteration),
        dtype=baseline_prediction_.dtype
    )
    raw_predictions += baseline_prediction_

    loss.update_gradients_and_hessians(
        gradients, hessians, y, raw_predictions
    )

    options = OptionSet(clf.parameter_dict)
    options['max_leaf_nodes'] = 5

    grower = TreeGrower(dataset, gradients, hessians, options)
    grower.grow()
    filename = tmpdir.join('plot_grower.pdf')
    plot_tree(grower, view=False, filename=filename)
    assert filename.exists()


def test_plot_estimator(tmpdir, classification_data):
    pytest.importorskip('graphviz')
    from pygbm.plotting import plot_tree

    n_trees = 3
    est = GradientBoostingRegressor(max_iter=n_trees)
    est.fit(classification_data[0], classification_data[1])
    for i in range(n_trees):
        filename = tmpdir.join('plot_predictor.pdf')
        plot_tree(est, tree_index=i, view=False, filename=filename)
        assert filename.exists()


def test_plot_estimator_and_lightgbm(tmpdir):
    pytest.importorskip('graphviz')
    lightgbm = pytest.importorskip('lightgbm')
    from pygbm.plotting import plot_tree

    n_classes = 3
    X, y = make_classification(n_samples=150, n_classes=n_classes,
                               n_features=5, n_informative=3, n_redundant=0,
                               random_state=0)

    n_trees = 3
    est_pygbm = GradientBoostingClassifier(max_iter=n_trees,
                                           n_iter_no_change=None)
    est_pygbm.fit(X, y)
    est_lightgbm = lightgbm.LGBMClassifier(n_estimators=n_trees)
    est_lightgbm.fit(X, y)

    n_total_trees = n_trees * n_classes
    for i in range(n_total_trees):
        filename = tmpdir.join('plot_mixed_predictors.pdf')
        plot_tree(est_pygbm, est_lightgbm=est_lightgbm, tree_index=i,
                  view=False, filename=filename)
        assert filename.exists()
