"""This module contains njitted routines and data structures to:

- Find the best possible split of a node. For a given node, a split is
  characterized by a feature and a bin.
- Apply a split to a node, i.e. split the indices of the samples at the node
  into the newly created left and right childs.
"""
import numpy as np
from numba import njit, jitclass, prange, float32, uint8, uint32, boolean
import numba

from .histogram import _build_histogram
from .histogram import _build_histogram_root
from .histogram import HISTOGRAM_DTYPE, sum_histogram

import pdb

@jitclass([
    ('gain', float32),
    ('feature_idx', uint32),
    ('bin_idx', uint8),
    ('left_g_hf', float32),
    ('left_gx_hfx', float32),
    ('left_h', float32),
    ('left_hx', float32),
    ('left_hx2', float32),
    ('if_left_linear', boolean),
    ('right_g_hf', float32),
    ('right_gx_hfx', float32),
    ('right_h', float32),
    ("right_hx", float32),
    ("right_hx2", float32),
    ('if_right_linear', boolean),
    ('n_samples_left', uint32),
    ('n_samples_right', uint32),
])
class SplitInfo:
    """Pure data class to store information about a potential split.

    Parameters
    ----------
    gain : float32
        The gain of the split
    feature_idx : int
        The index of the feature to be split
    bin_idx : int
        The index of the bin on which the split is made
    left_g_hf, left_gx_hfx, left_h, left_hx, left_hx2: float32
        Accumulants in the left child
    right_g_hf, right_gx_hfx, right_h, right_hx, right_hx2: float32
        Accumulants in the right child
    n_samples_left : int
        The number of samples in the left child
    n_samples_right : int
        The number of samples in the right child
    """

    def __init__(self, gain=-1., feature_idx=0, bin_idx=0,
                 left_g_hf=0., left_gx_hfx=0., left_h=0., left_hx=0., left_hx2=0.,
                 right_g_hf=0., right_gx_hfx=0., right_h=0, right_hx=0, right_hx2=0,
                 n_samples_left=0, n_samples_right=0):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx

        self.left_g_hf = left_g_hf
        self.left_gx_hfx = left_gx_hfx
        self.left_h = left_h
        self.left_hx = left_hx
        self.left_hx2 = left_hx2

        self.right_g_hf = right_g_hf
        self.right_gx_hfx = right_gx_hfx
        self.right_h = right_h
        self.right_hx = right_hx
        self.right_hx2 = right_hx2

        self.n_samples_left = n_samples_left
        self.n_samples_right = n_samples_right


@jitclass([
    ('n_features', uint32),
    ('X_binned', uint8[::1, :]),
    ('X', float32[::1, :]),
    ('max_bins', uint32),
    ('n_bins_per_feature', uint32[::1]),
    ('min_samples_leaf', uint32),
    ('min_gain_to_split', float32),
    ('gradients', float32[::1]),
    ('hessians', float32[::1]),
    ('prediction_value', float32[::1]),
    ('w_l2_reg', float32),
    ('b_l2_reg', float32),
    ('min_hessian_to_split', float32),
    ('partition', uint32[::1]),
    ('left_indices_buffer', uint32[::1]),
    ('right_indices_buffer', uint32[::1]),
])
class SplittingContext:
    """Pure data class defining a splitting context.

    Ideally it would also have methods but numba does not support annotating
    jitclasses (so we can't use parallel=True). This structure is
    instanciated in the grower and stores all the required information to
    compute the SplitInfo and histograms of each node.

    Parameters
    ----------
    X_binned : array of int
        The binned input samples. Must be Fortran-aligned.
    max_bins : int, optional(default=256)
        The maximum number of bins. Used to define the shape of the
        histograms.
    n_bins_per_feature : array-like of int
        The actual number of bins needed for each feature, which is lower or
        equal to max_bins.
    gradients : array-like, shape=(n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : array-like, shape=(n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    l2_regularization : float
        The L2 regularization parameter.
    min_hessian_to_split : float
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        min_hessian_to_split are discarded.
    min_samples_leaf : int
        The minimum number of samples per leaf.
    min_gain_to_split : float, optional(default=0.)
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    """
    def __init__(self, X_binned, X, max_bins, n_bins_per_feature,
                 gradients, hessians, w_l2_reg, b_l2_reg,
                 min_hessian_to_split=1e-3, min_samples_leaf=20,
                 min_gain_to_split=0.):

        self.X_binned = X_binned
        self.X = X
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <max_bins> bins, but some of the
        # last bins may be unused if n_bins_per_feature[f] < max_bins
        self.max_bins = max_bins
        self.n_bins_per_feature = n_bins_per_feature
        self.gradients = gradients
        self.hessians = hessians
        self.prediction_value = np.zeros_like(self.gradients)
        # for root node, gradients and hessians are already ordered
        self.w_l2_reg = w_l2_reg
        self.b_l2_reg = b_l2_reg
        self.min_hessian_to_split = min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split

        # The partition array maps each sample index into the leaves of the
        # tree (a leaf in this context is a node that isn't splitted yet, not
        # necessarily a 'finalized' leaf). Initially, the root contains all
        # the indices, e.g.:
        # partition = [abcdefghijkl]
        # After a call to split_indices, it may look e.g. like this:
        # partition = [cef|abdghijkl]
        # we have 2 leaves, the left one is at position 0 and the second one at
        # position 3. The order of the samples is irrelevant.
        self.partition = np.arange(0, X_binned.shape[0], 1, np.uint32)
        # buffers used in split_indices to support parallel splitting.
        self.left_indices_buffer = np.empty_like(self.partition)
        self.right_indices_buffer = np.empty_like(self.partition)


@njit(parallel=True)
def update_prediction_values(context, sample_indices, coefficient, feature_idx):
    n_samples = sample_indices.shape[0]
    prediction_value = context.prediction_value
    for i in prange(n_samples):
        sample_idx = sample_indices[i]
        prediction_value[sample_idx] += context.X[sample_idx][feature_idx] * coefficient


@njit(parallel=True,
      locals={'sample_idx': uint32,
              'left_count': uint32,
              'right_count': uint32})
def split_indices(context, split_info, sample_indices):
    """Split samples into left and right arrays.

    Parameters
    ----------
    context : SplittingContext
        The splitting context
    split_ingo : SplitInfo
        The SplitInfo of the node to split
    sample_indices : array of int
        The indices of the samples at the node to split. This is a view on
        context.partition, and it is modified inplace by placing the indices
        of the left child at the beginning, and the indices of the right child
        at the end.

    Returns
    -------
    left_indices : array of int
        The indices of the samples in the left child. This is a view on
        context.partition.
    right_indices : array of int
        The indices of the samples in the right child. This is a view on
        context.partition.
    """
    # This is a multi-threaded implementation inspired by lightgbm.
    # Here is a quick break down. Let's suppose we want to split a node with
    # 24 samples named from a to x. context.partition looks like this (the *
    # are indices in other leaves that we don't care about):
    # partition = [*************abcdefghijklmnopqrstuvwx****************]
    #                           ^                       ^
    #                     node_position     node_position + node.n_samples

    # Ultimately, we want to reorder the samples inside the boundaries of the
    # leaf (which becomes a node) to now represent the samples in its left and
    # right child. For example:
    # partition = [*************abefilmnopqrtuxcdghjksvw*****************]
    #                           ^              ^
    #                   left_child_pos     right_child_pos
    # Note that left_child_pos always takes the value of node_position, and
    # right_child_pos = left_child_pos + left_child.n_samples. The order of
    # the samples inside a leaf is irrelevant.

    # 1. samples_indices is a view on this region a..x. We conceptually
    #    divide it into n_threads regions. Each thread will be responsible for
    #    its own region. Here is an example with 4 threads:
    #    samples_indices = [abcdef|ghijkl|mnopqr|stuvwx]
    # 2. Each thread processes 6 = 24 // 4 entries and maps them into
    #    left_indices_buffer or right_indices_buffer. For example, we could
    #    have the following mapping ('.' denotes an undefined entry):
    #    - left_indices_buffer =  [abef..|il....|mnopqr|tux...]
    #    - right_indices_buffer = [cd....|ghjk..|......|svw...]
    # 3. We keep track of the start positions of the regions (the '|') in
    #    ``offset_in_buffers`` as well as the size of each region. We also keep
    #    track of the number of samples put into the left/right child by each
    #    thread. Concretely:
    #    - left_counts =  [4, 2, 6, 3]
    #    - right_counts = [2, 4, 0, 3]
    # 4. Finally, we put left/right_indices_buffer back into the
    #    samples_indices, without any undefined entries and the partition looks
    #    as expected
    #    partition = [*************abefilmnopqrtuxcdghjksvw*****************]

    # Note: We here show left/right_indices_buffer as being the same size as
    # sample_indices for simplicity, but in reality they are of the same size
    # as partition.

    X_binned = context.X_binned.T[split_info.feature_idx]

    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    n_samples = sample_indices.shape[0]

    # Note: we could probably allocate all the arrays of size n_threads in the
    # splitting context as well, but gains are probably going to be minimal
    sizes = np.full(n_threads, n_samples // n_threads, dtype=np.int32)
    if n_samples % n_threads > 0:
        # array[:0] will cause a bug in numba 0.41 so we need the if. Remove
        # once issue numba 3554 is fixed.
        sizes[:n_samples % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])

    left_counts = np.empty(n_threads, dtype=np.int32)
    right_counts = np.empty(n_threads, dtype=np.int32)

    # Need to declare local variables, else they're not updated :/
    # (see numba issue 3459)
    left_indices_buffer = context.left_indices_buffer
    right_indices_buffer = context.right_indices_buffer

    # map indices from samples_indices to left/right_indices_buffer
    for thread_idx in prange(n_threads):
        left_count = 0
        right_count = 0

        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            sample_idx = sample_indices[i]
            if X_binned[sample_idx] <= split_info.bin_idx:
                left_indices_buffer[start + left_count] = sample_idx
                left_count += 1
            else:
                right_indices_buffer[start + right_count] = sample_idx
                right_count += 1

        left_counts[thread_idx] = left_count
        right_counts[thread_idx] = right_count

    # position of right child = just after the left child
    right_child_position = left_counts.sum()

    # offset of each thread in samples_indices for left and right child, i.e.
    # where each thread will start to write.
    left_offset = np.zeros(n_threads, dtype=np.int32)
    left_offset[1:] = np.cumsum(left_counts[:-1])
    right_offset = np.full(n_threads, right_child_position, dtype=np.int32)
    right_offset[1:] += np.cumsum(right_counts[:-1])

    # map indices in left/right_indices_buffer back into samples_indices. This
    # also updates context.partition since samples_indice is a view.
    for thread_idx in prange(n_threads):

        for i in range(left_counts[thread_idx]):
            sample_indices[left_offset[thread_idx] + i] = \
                left_indices_buffer[offset_in_buffers[thread_idx] + i]
        for i in range(right_counts[thread_idx]):
            sample_indices[right_offset[thread_idx] + i] = \
                right_indices_buffer[offset_in_buffers[thread_idx] + i]

    return (sample_indices[:right_child_position],
            sample_indices[right_child_position:])


@njit(parallel=True)
def find_node_split(context, sample_indices):
    """For each feature, find the best bin to split on at a given node.

    Returns the best split info among all features, and the histograms of
    all the features. The histograms are computed by scanning the whole
    data.

    Parameters
    ----------
    context : SplittingContext
        The splitting context
    sample_indices : array of int
        The indices of the samples at the node to split.

    Returns
    -------
    best_split_info : SplitInfo
        The info about the best possible split among all features.
    histograms : array of HISTOGRAM_DTYPE, shape=(n_features, max_bins)
        The histograms of each feature. A histogram is an array of
        HISTOGRAM_DTYPE of size ``max_bins`` (only
        ``n_bins_per_features[feature]`` entries are relevant).
    """

    n_samples = sample_indices.shape[0]

    # Pre-allocate the results datastructure to be able to use prange:
    # numba jitclass do not seem to properly support default values for kwargs.
    split_infos = [SplitInfo(-1., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0)
                   for i in range(context.n_features)]

    histograms = np.empty(
        shape=(np.int64(context.n_features), np.int64(context.max_bins)),
        dtype=HISTOGRAM_DTYPE
    )

    for feature_idx in prange(context.n_features):
        X_binned = context.X_binned.T[feature_idx]
        X = context.X.T[feature_idx]

        root_node = X_binned.shape[0] == n_samples

        if root_node:
            histogram = _build_histogram_root(
                context.max_bins, X_binned, X, context.prediction_value,
                context.gradients, context.hessians)
        else:
            histogram = _build_histogram(
                context.max_bins, sample_indices, X_binned, X, context.prediction_value,
                context.gradients, context.hessians)

        histogram_summ = sum_histogram(histogram)

        split_info = _find_best_bin_to_split_helper(
            context, feature_idx, histogram, n_samples, histogram_summ)

        split_infos[feature_idx] = split_info
        histograms[feature_idx, :] = histogram

    split_info = _find_best_feature_to_split_helper(split_infos)
    return split_info, histograms


@njit
def _find_best_feature_to_split_helper(split_infos):
    best_gain = None
    for i, split_info in enumerate(split_infos):
        gain = split_info.gain
        if best_gain is None or gain > best_gain:
            best_gain = gain
            best_split_info = split_info
    return best_split_info


@njit(locals={'left_g_hf': float32, 'left_hx_hfx': float32,
              'left_h': float32, 'left_hx': float32, 'left_hx2': float32,
              'n_samples_left': uint32},
      fastmath=True)
def _find_best_bin_to_split_helper(context, feature_idx, histogram, n_samples, h_summ):
    """Find best bin to split on, and return the corresponding SplitInfo.

    Splits that do not satisfy the splitting constraints (min_gain_to_split,
    etc.) are discarded here. If no split can satisfy the constraints, a
    SplitInfo with a gain of -1 is returned. If for a given node the best
    SplitInfo has a gain of -1, it is finalized into a leaf.
    """
    # Allocate the structure for the best split information. It can be
    # returned as such (with a negative gain) if the min_hessian_to_split
    # condition is not satisfied. Such invalid splits are later discarded by
    # the TreeGrower.
    best_split = SplitInfo(-1., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0)
    left_g_hf, left_gx_hfx, left_h, left_hx, left_hx2 = 0., 0., 0., 0., 0.
    n_samples_left = 0

    for bin_idx in range(context.n_bins_per_feature[feature_idx]):
        n_samples_left += histogram[bin_idx]['count']
        n_samples_right = n_samples - n_samples_left

        left_g_hf += histogram[bin_idx]['sum_g_hf']
        right_g_hf = h_summ[0]['sum_g_hf'] - left_g_hf

        left_gx_hfx += histogram[bin_idx]['sum_gx_hfx']
        right_gx_hfx = h_summ[0]['sum_gx_hfx'] - left_gx_hfx

        left_h += histogram[bin_idx]['sum_h']
        right_h = h_summ[0]['sum_h'] - left_h

        left_hx += histogram[bin_idx]['sum_hx']
        right_hx = h_summ[0]['sum_hx'] - left_hx

        left_hx2 += histogram[bin_idx]['sum_hx2']
        right_hx2 = h_summ[0]['sum_hx2'] - left_hx2

        if n_samples_left < context.min_samples_leaf:
            continue
        if n_samples_right < context.min_samples_leaf:
            # won't get any better
            break

        left_denominator = (left_hx2 * left_h - left_hx**2)
        right_denominator = (right_hx2 * right_h - right_hx**2)
        if min(left_denominator, right_denominator) < context.min_hessian_to_split:
            continue

        gain = _split_gain(
            left_g_hf, left_gx_hfx, left_h, left_hx, left_hx2,
            right_g_hf, right_gx_hfx, right_h, right_hx, right_hx2,
            h_summ[0]['sum_g_hf'], h_summ[0]['sum_gx_hfx'], h_summ[0]['sum_h'], h_summ[0]['sum_hx'], h_summ[0]['sum_hx2'],
            context.w_l2_reg, context.b_l2_reg)

        if gain > best_split.gain and gain > context.min_gain_to_split:
            best_split.gain = gain
            best_split.feature_idx = feature_idx
            best_split.bin_idx = bin_idx
            best_split.n_samples_left = n_samples_left
            best_split.n_samples_right = n_samples_right

            best_split.left_g_hf = left_g_hf
            best_split.left_gx_hfx = left_gx_hfx
            best_split.left_h = left_h
            best_split.left_hx = left_hx
            best_split.left_hx2 = left_hx2

            best_split.right_g_hf = right_g_hf
            best_split.right_gx_hfx = right_gx_hfx
            best_split.right_h = right_h
            best_split.right_hx = right_hx
            best_split.right_hx2 = right_hx2

    return best_split


@njit(fastmath=False)
def _split_gain(left_g_hf, left_gx_hfx, left_h, left_hx, left_hx2,
                   right_g_hf, right_gx_hfx, right_h, right_hx, right_hx2,
                   sum_g_hf, sum_gx_hfx, sum_h, sum_hx, sum_hx2,
                   w_l2_reg, b_l2_reg):

    def negative_loss_constant(g_hf, h):
        return g_hf ** 2 / (h + b_l2_reg)

    def negative_loss_linear(g_hf, gx_hfx, h, hx, hx2):
        hx2_reg = hx2 + w_l2_reg
        h_reg = h + b_l2_reg
        return (hx2_reg * g_hf**2 + h_reg * gx_hfx**2 - 2 * g_hf * gx_hfx * hx) / (hx2_reg * h_reg - hx**2)

    gain = negative_loss_linear(left_g_hf, left_gx_hfx, left_h, left_hx, left_hx2)
    gain += negative_loss_linear(right_g_hf, right_gx_hfx, right_h, right_hx, right_hx2)
    gain -= negative_loss_constant(sum_g_hf, sum_h)

    return gain
