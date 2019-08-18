"""This module contains njitted routines for building histograms.

A histogram is an array with n_bins entry of type HISTOGRAM_DTYPE. Each
feature has its own histogram. A histogram contains the sum of gradients and
hessians of all the samples belonging to each bin.
"""
import numpy as np
from numba import njit


HISTOGRAM_DTYPE = np.dtype([
    ('sum_g_hf', np.float32),
    ('sum_gx_hfx', np.float32),
    ('sum_h', np.float32),
    ('sum_hx', np.float32),
    ('sum_hx2', np.float32),
    ('count', np.uint32),
])


@njit
def sum_histogram(histogram):
    histogram_summ = np.zeros(1, dtype=HISTOGRAM_DTYPE)
    for bin_idx in range(histogram.shape[0]):
        histogram_summ[0]['sum_g_hf'] += histogram[bin_idx]['sum_g_hf']
        histogram_summ[0]['sum_gx_hfx'] += histogram[bin_idx]['sum_gx_hfx']
        histogram_summ[0]['sum_h'] += histogram[bin_idx]['sum_h']
        histogram_summ[0]['sum_hx'] += histogram[bin_idx]['sum_hx']
        histogram_summ[0]['sum_hx2'] += histogram[bin_idx]['sum_hx2']
        histogram_summ[0]['count'] += histogram[bin_idx]['count']
    return histogram_summ


@njit
def _build_histogram(n_bins, sample_indices, binned_feature,
                     feature_value, prediction_value,
                     gradients, hessians):
    """Build histogram in a naive way, without optimizing for cache hit."""
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    for i, sample_idx in enumerate(sample_indices):
        bin_idx = binned_feature[sample_idx]
        histogram[bin_idx]['sum_g_hf'] += gradients[sample_idx] + hessians[sample_idx] * prediction_value[sample_idx]
        histogram[bin_idx]['sum_gx_hfx'] += (
            gradients[sample_idx] * feature_value[sample_idx] +
            hessians[sample_idx] * prediction_value[sample_idx] * feature_value[sample_idx]
        )
        histogram[bin_idx]['sum_h'] += hessians[sample_idx]
        histogram[bin_idx]['sum_hx'] += hessians[sample_idx] * feature_value[sample_idx]
        histogram[bin_idx]['sum_hx2'] += hessians[sample_idx] * (feature_value[sample_idx])**2
        histogram[bin_idx]['count'] += 1
    return histogram


@njit
def _build_histogram_root(n_bins, binned_feature, feature_value, prediction_value, all_gradients, all_hessians):
    """Special case for the root node

    The root node has to find the split among all the samples from the
    training set. binned_feature and all_gradients and all_hessians already
    have a consistent ordering.
    """
    """Build histogram in a naive way, without optimizing for cache hit."""
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = binned_feature.shape[0]
    for i in range(n_node_samples):
        bin_idx = binned_feature[i]
        histogram[bin_idx]['sum_g_hf'] += all_gradients[i] + all_hessians[i] * prediction_value[i]
        histogram[bin_idx]['sum_gx_hfx'] += (
            all_gradients[i] * feature_value[i] +
            all_hessians[i] * prediction_value[i] * feature_value[i]
        )
        histogram[bin_idx]['sum_h'] += all_hessians[i]
        histogram[bin_idx]['sum_hx'] += all_hessians[i] * feature_value[i]
        histogram[bin_idx]['sum_hx2'] += all_hessians[i] * (feature_value[i]) ** 2
        histogram[bin_idx]['count'] += 1
    return histogram


@njit
def _subtract_histograms(n_bins, hist_a, hist_b):
    """Return hist_a - hist_b"""

    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)

    s_g_hf = 'sum_g_hf'
    s_gx_hfx = 'sum_gx_hfx'
    s_h = 'sum_h'
    s_hx = 'sum_hx'
    s_hx2 = 'sum_hx2'
    c = 'count'

    for i in range(n_bins):
        histogram[i][s_g_hf] = hist_a[i][s_g_hf] - hist_b[i][s_g_hf]
        histogram[i][s_gx_hfx] = hist_a[i][s_gx_hfx] - hist_b[i][s_gx_hfx]
        histogram[i][s_h] = hist_a[i][s_h] - hist_b[i][s_h]
        histogram[i][s_hx] = hist_a[i][s_hx] - hist_b[i][s_hx]
        histogram[i][s_hx2] = hist_a[i][s_hx2] - hist_b[i][s_hx2]
        histogram[i][c] = hist_a[i][c] - hist_b[i][c]

    return histogram
