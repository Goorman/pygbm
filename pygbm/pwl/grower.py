"""
This module contains the TreeGrower class.

TreeGrowee builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""
from heapq import heappush, heappop
import numpy as np
from numba import njit, prange
from time import time

from .splitting import (SplittingContext, split_indices, find_node_split, update_prediction_values)
from .predictor import TreePredictor, PREDICTOR_RECORD_DTYPE

from pygbm.options import OptionSet


class TreeNode:
    """Tree Node class used in TreeGrower.

    This isn't used for prediction purposes, only for training (see
    TreePredictor).

    Parameters
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root
    samples_indices : array of int
        The indices of the samples at the node
    sum_gradients : float
        The sum of the gradients of the samples at the node
    sum_hessians : float
        The sum of the hessians of the samples at the node
    parent : TreeNode or None, optional(default=None)
        The parent of the node. None for root.

    Attributes
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root
    samples_indices : array of int
        The indices of the samples at the node
    sum_g_hf, sum_gx_hfx, sum_h, sum_hx, sum_hx2 : float
        Difference sums for samples at the node
    parent : TreeNode or None, optional(default=None)
        The parent of the node. None for root.
    split_info : SplitInfo or None
        The result of the split evaluation
    left_child : TreeNode or None
        The left child of the node. None for leaves.
    right_child : TreeNode or None
        The right child of the node. None for leaves.
    value : float or None
        The value of the leaf, as computed in finalize_leaf(). None for
        non-leaf nodes
    find_split_time : float
        The total time spent computing the histogram and finding the best
        split at the node.
    construction_speed : float
        The Number of samples at the node divided find_split_time.
    apply_split_time : float
        The total time spent actually splitting the node, e.g. splitting
        samples_indices into left and right child.
    hist_subtraction : bool
        Wheter the subtraction method was used for computing the histograms.
    """

    split_info = None
    left_child = None
    right_child = None
    value = None
    left_coefficient=None
    right_coefficient=None
    histograms = None
    sibling = None
    parent = None
    find_split_time = 0.
    construction_speed = 0.
    apply_split_time = 0.
    hist_subtraction = False

    def __init__(self, depth, sample_indices, sum_g_hf, sum_gx_hfx, sum_h, sum_hx, sum_hx2, parent=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.n_samples = sample_indices.shape[0]
        self.sum_g_hf = sum_g_hf
        self.sum_gx_hfx = sum_gx_hfx
        self.sum_h = sum_h
        self.sum_hx = sum_hx
        self.sum_hx2 = sum_hx2
        self.parent = parent

    def __repr__(self):
        # To help with debugging
        out = f"TreeNode: depth={self.depth}, "
        out += f"samples={len(self.sample_indices)}"
        if self.split_info is not None:
            out += f", feature_idx={self.split_info.feature_idx}"
            out += f", bin_idx={self.split_info.bin_idx}"
        return out

    def __lt__(self, other_node):
        """Comparison for priority queue.

        Nodes with high gain are higher priority than nodes with low gain.

        heapq.heappush only need the '<' operator.
        heapq.heappop take the smallest item first (smaller is higher
        priority).

        Parameters
        -----------
        other_node : TreeNode
            The node to compare with.
        """
        if self.split_info is None or other_node.split_info is None:
            raise ValueError("Cannot compare nodes with split_info")
        return self.split_info.gain > other_node.split_info.gain


class TreeGrower:
    """Tree grower class used to build a tree.

    The tree is fitted to predict the values of a Newton-Raphson step. The
    splits are considered in a best-first fashion, and the quality of a
    split is defined in splitting._split_gain.

    Parameters
    ----------
    X_binned : array-like of int, shape=(n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    gradients : array-like, shape=(n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    hessians : array-like, shape=(n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    max_leaf_nodes : int or None, optional(default=None)
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
    max_depth : int or None, optional(default=None)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf.
    min_samples_leaf : int, optional(default=20)
        The minimum number of samples per leaf.
    min_gain_to_split : float, optional(default=0.)
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    max_bins : int, optional(default=256)
        The maximum number of bins. Used to define the shape of the
        histograms.
    n_bins_per_feature : array-like of int or int, optional(default=None)
        The actual number of bins needed for each feature, which is lower or
        equal to ``max_bins``. If it's an int, all features are considered to
        have the same number of bins. If None, all features are considered to
        have ``max_bins`` bins.
    l2_regularization : float, optional(default=0)
        The L2 regularization parameter.
    min_det_to_split : float, optional(default=1e-3)
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        min_determinant_to_split are discarded.
    shrinkage : float, optional(default=1)
        The shrinkage parameter to apply to the leaves values, also known as
        learning rate.
    """
    def __init__(self, dataset, gradients, hessians, options: OptionSet):

        self.max_leaf_nodes = options['max_leaf_nodes']
        self.max_depth = options['max_depth']
        self.min_samples_leaf = options['min_samples_leaf']
        self.min_gain_to_split = options['min_gain_to_split']
        self.max_bins = options['max_bins']
        self.n_bins_per_feature = dataset.n_bins_per_feature
        self.numerical_thresholds = dataset.numerical_thresholds
        self.w_l2_reg = options['w_l2_reg']
        self.b_l2_reg = options['b_l2_reg']
        self.min_hessian_to_split = options['min_hessian_to_split']
        self.shrinkage = options['learning_rate']

        self._validate_parameters(dataset.X_binned, dataset.X, self.max_leaf_nodes, self.max_depth,
                                  self.min_samples_leaf, self.min_gain_to_split,
                                  self.w_l2_reg, self.b_l2_reg, self.min_hessian_to_split)

        if self.n_bins_per_feature is None:
            self.n_bins_per_feature = self.max_bins

        if isinstance(self.n_bins_per_feature, int):
            self.n_bins_per_feature = np.array(
                [self.n_bins_per_feature] * dataset.X_binned.shape[1],
                dtype=np.uint32)

        self.splitting_context = SplittingContext(
            dataset.X_binned, dataset.X, self.max_bins, self.n_bins_per_feature, gradients,
            hessians, self.w_l2_reg, self.b_l2_reg, self.min_hessian_to_split,
            self.min_samples_leaf, self.min_gain_to_split)
        self.X_binned = dataset.X_binned
        self.splittable_nodes = []
        self.finalized_leaves = []
        self.total_find_split_time = 0.  # time spent finding the best splits
        self.total_apply_split_time = 0.  # time spent splitting nodes
        self._initialize_root()
        self.n_nodes = 1

    def _validate_parameters(self, X_binned, X, max_leaf_nodes, max_depth,
                             min_samples_leaf, min_gain_to_split,
                             w_l2_reg, b_l2_reg, min_hessian_to_split):
        """Validate parameters passed to __init__.

        Also validate parameters passed to SplittingContext because we cannot
        raise exceptions in a jitclass.
        """
        if X_binned.dtype != np.uint8:
            raise NotImplementedError(
                "Explicit feature binning required for now")
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency.")
        if not X.flags.f_contiguous:
            raise ValueError(
                "X should be passed as Fortran contiguous "
                "array for maximum efficiency.")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError(f'max_leaf_nodes={max_leaf_nodes} should not be'
                             f' smaller than 1')
        if max_depth is not None and max_depth < 1:
            raise ValueError(f'max_depth={max_depth} should not be'
                             f' smaller than 1')
        if min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf={min_samples_leaf} should '
                             f'not be smaller than 1')
        if min_gain_to_split < 0:
            raise ValueError(f'min_gain_to_split={min_gain_to_split} '
                             f'must be positive.')
        if w_l2_reg < 0 or b_l2_reg < 0:
            raise ValueError(f'l2_regularization=w:{w_l2_reg},b:{b_l2_reg} must be '
                             f'positive.')
        if min_hessian_to_split < 0:
            raise ValueError(f'min_hessian_to_split={min_hessian_to_split} '
                             f'must be positive.')

    def grow(self):
        """Grow the tree, from root to leaves."""
        while self.can_split_further():
            self.split_next()

    def _initialize_root(self):
        """Initialize root node and finalize it if needed."""
        n_samples = self.X_binned.shape[0]
        depth = 0
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitting_context.partition.view(),
            sum_g_hf=self.splitting_context.gradients.sum(),
            sum_h=self.splitting_context.hessians.sum(),
            sum_gx_hfx=0,
            sum_hx=0,
            sum_hx2=0
        )
        if (self.max_leaf_nodes is not None and self.max_leaf_nodes == 1):
            self._finalize_leaf(self.root)
            return
        if self.root.n_samples < 2 * self.min_samples_leaf:
            # Do not even bother computing any splitting statistics.
            self._finalize_leaf(self.root)
            return

        self._compute_root_spittability()

    def _compute_root_spittability(self):
        """Compute histograms and best possible split of a root node
        """
        node = self.root
        tic = time()
        split_info, histograms = find_node_split(
            self.splitting_context, node.sample_indices)

        toc = time()
        node.find_split_time = toc - tic
        self.total_find_split_time += node.find_split_time
        node.construction_speed = node.n_samples / node.find_split_time
        node.split_info = split_info
        node.histograms = histograms

        if node.split_info.gain <= 0:  # no valid split
#            import pdb; pdb.set_trace()
            # Note: this condition is reached if either all the leaves are
            # pure (best gain = 0), or if no split would satisfy the
            # constraints, (min_hessians_to_split, min_gain_to_split,
            # min_samples_leaf)
            self._finalize_leaf(node)

        else:
            heappush(self.splittable_nodes, node)

    def _compute_sibling_splittability(self, left_child_node, right_child_node):
        if left_child_node.n_samples > right_child_node.n_samples:
            min_child_node, max_child_node = right_child_node, left_child_node
        else:
            min_child_node, max_child_node = left_child_node, right_child_node

        if max_child_node.n_samples < 2 * self.min_samples_leaf:
            self._finalize_leaf(max_child_node)
            self._finalize_leaf(min_child_node)
            return left_child_node, right_child_node

        tic = time()
        split_info, histograms = find_node_split(
            self.splitting_context, min_child_node.sample_indices)
        toc = time()
        min_child_node.find_split_time = toc - tic
        self.total_find_split_time += min_child_node.find_split_time
        min_child_node.construction_speed = min_child_node.n_samples / min_child_node.find_split_time
        min_child_node.split_info = split_info
        min_child_node.histograms = histograms

        tic = time()
        split_info, histograms = find_node_split(
            self.splitting_context, max_child_node.sample_indices)
        toc = time()
        max_child_node.find_split_time = toc - tic
        self.total_find_split_time += max_child_node.find_split_time
        max_child_node.construction_speed = max_child_node.n_samples / max_child_node.find_split_time
        max_child_node.split_info = split_info
        max_child_node.histograms = histograms

        if max_child_node.split_info.gain <= 0:  # no valid split
            # Note: this condition is reached if either all the leaves are
            # pure (best gain = 0), or if no split would satisfy the
            # constraints, (min_hessians_to_split, min_gain_to_split,
            # min_samples_leaf)
            self._finalize_leaf(max_child_node)

        else:
            heappush(self.splittable_nodes, max_child_node)

        if min_child_node.split_info.gain <= 0:  # no valid split
            # Note: this condition is reached if either all the leaves are
            # pure (best gain = 0), or if no split would satisfy the
            # constraints, (min_hessians_to_split, min_gain_to_split,
            # min_samples_leaf)
            self._finalize_leaf(min_child_node)

        else:
            heappush(self.splittable_nodes, min_child_node)

        return left_child_node, right_child_node

    def split_next(self):
        """Split the node with highest potential gain.

        Returns
        -------
        left : TreeNode
            The resulting left child.
        right : TreeNode
            The resulting right child.
        """
        if len(self.splittable_nodes) == 0:
            raise StopIteration("No more splittable nodes")

        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)

        tic = time()
        (sample_indices_left, sample_indices_right) = split_indices(
            self.splitting_context, node.split_info, node.sample_indices)
        toc = time()
        node.apply_split_time = toc - tic
        self.total_apply_split_time += node.apply_split_time

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(depth,
                                   sample_indices_left,
                                   node.split_info.left_g_hf,
                                   node.split_info.left_gx_hfx,
                                   node.split_info.left_h,
                                   node.split_info.left_hx,
                                   node.split_info.left_hx2,
                                   parent=node)
        right_child_node = TreeNode(depth,
                                    sample_indices_right,
                                    node.split_info.right_g_hf,
                                    node.split_info.right_gx_hfx,
                                    node.split_info.right_h,
                                    node.split_info.right_hx,
                                    node.split_info.right_hx2,
                                    parent=node)
        left_child_node.sibling = right_child_node
        right_child_node.sibling = left_child_node
        node.right_child = right_child_node
        node.left_child = left_child_node
        self.n_nodes += 2

        node.left_coefficient = self._compute_linear_coefficient(left_child_node)
        node.right_coefficient = self._compute_linear_coefficient(right_child_node)

        update_prediction_values(
            self.splitting_context, sample_indices_left,
            node.left_coefficient, node.split_info.feature_idx
        )
        update_prediction_values(
            self.splitting_context, sample_indices_right,
            node.right_coefficient, node.split_info.feature_idx
        )

        if (self.max_leaf_nodes is not None
            and n_leaf_nodes == self.max_leaf_nodes):
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            return left_child_node, right_child_node

        self._compute_sibling_splittability(left_child_node, right_child_node)

        return left_child_node, right_child_node

    def can_split_further(self):
        """Return True if there are still nodes to split."""
        return len(self.splittable_nodes) >= 1

    def _compute_linear_coefficient(self, node):

        sum_hx2_reg = node.sum_hx2 + self.splitting_context.w_l2_reg
        sum_h_reg = node.sum_h + self.splitting_context.b_l2_reg
        coefficient = (node.sum_hx * node.sum_g_hf - sum_h_reg * node.sum_gx_hfx) / \
                      (sum_hx2_reg * sum_h_reg - node.sum_hx ** 2)

        return coefficient

    def _finalize_leaf(self, node):
        if node.parent is None:
            self._finalize_leaf_plain(node)
        else:
            self._finalize_leaf_linear(node)

    def _finalize_leaf_linear(self, node):
        sum_hx2_reg = node.sum_hx2 + self.splitting_context.w_l2_reg
        sum_h_reg = node.sum_h + self.splitting_context.b_l2_reg
        node.value = self.shrinkage * (node.sum_hx * node.sum_gx_hfx - sum_hx2_reg * node.sum_g_hf) / (
            sum_hx2_reg * sum_h_reg - node.sum_hx**2)
        self.finalized_leaves.append(node)

    def _finalize_leaf_plain(self, node):
        """Compute the prediction value that minimizes the objective function.

        This sets the node.value attribute (node is a leaf iff node.value is
        not None).

        See Equation 5 of:
        XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
        https://arxiv.org/abs/1603.02754
        """
        node.value = -self.shrinkage * node.sum_g_hf / (
            node.sum_h + self.splitting_context.b_l2_reg)
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self):
        """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        numerical_thresholds : array-like of floats, optional (default=None)
            The actual thresholds values of each bin, expected to be in sorted
            increasing order. None if the training data was pre-binned.

        Returns
        -------
        A TreePredictor object.
        """
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        self._fill_predictor_node_array(
            predictor_nodes, self.root,
            numerical_thresholds=self.numerical_thresholds
        )
        return TreePredictor(nodes=predictor_nodes)

    def _fill_predictor_node_array(self, predictor_nodes, grower_node,
                                   numerical_thresholds=None, next_free_idx=0):
        """Helper used in make_predictor to set the TreePredictor fields."""
        node = predictor_nodes[next_free_idx]
        node['count'] = grower_node.n_samples
        node['depth'] = grower_node.depth
        if grower_node.split_info is not None:
            node['gain'] = grower_node.split_info.gain
        else:
            node['gain'] = -1

        if grower_node.value is not None:
            # Leaf node
            node['is_leaf'] = True
            node['value'] = grower_node.value
            return next_free_idx + 1
        else:
            # Decision node
            split_info = grower_node.split_info
            feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
            node['feature_idx'] = feature_idx
            node['bin_threshold'] = bin_idx
            if numerical_thresholds is not None:
                node['threshold'] = numerical_thresholds[feature_idx][bin_idx]
            node['left_coefficient'] = grower_node.left_coefficient * self.shrinkage
            node['right_coefficient'] = grower_node.right_coefficient * self.shrinkage
            next_free_idx += 1

            node['left'] = next_free_idx
            next_free_idx = self._fill_predictor_node_array(
                predictor_nodes, grower_node.left_child,
                numerical_thresholds=numerical_thresholds,
                next_free_idx=next_free_idx)

            node['right'] = next_free_idx
            return self._fill_predictor_node_array(
                predictor_nodes, grower_node.right_child,
                numerical_thresholds=numerical_thresholds,
                next_free_idx=next_free_idx)

    def update_raw_predictions(self, raw_predictions):
        # prepare leaves_data so that _update_raw_predictions can be
        # @njitted
        leaves_data = [(l.value, l.sample_indices)
                        for l in self.finalized_leaves]
        prediction_value = self.splitting_context.prediction_value * self.shrinkage
        _update_raw_predictions(leaves_data, raw_predictions, prediction_value)


@njit(parallel=True)
def _update_raw_predictions(leaves_data, raw_predictions, prediction_value):
    """Update raw_predictions by reading the predictions of the ith tree
    directly form the leaves.

    Can only be used for predicting the training data. raw_predictions
    contains the sum of the tree values from iteration 0 to i - 1. This adds
    the predictions of the ith tree to raw_predictions.

    Parameters
    ----------
    leaves_data: list of tuples (leaf.value, leaf.sample_indices)
        The leaves data used to update raw_predictions.
    raw_predictions : array-like, shape=(n_samples,)
        The raw predictions for the training data.
    prediction_value:array-like, shape=(n_samples,)
        The accumulated linear values
    """
    for leaf_idx in prange(len(leaves_data)):
        leaf_value, sample_indices = leaves_data[leaf_idx]
        for sample_idx in sample_indices:
            raw_predictions[sample_idx] += leaf_value
    n_samples = raw_predictions.shape[0]
    for i in prange(n_samples):
        raw_predictions[i] += prediction_value[i]
