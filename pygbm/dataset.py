import pandas as pd
import numpy as np
from sklearn.utils import check_X_y, check_random_state
from .utils import Timer
from .binning import BinMapper

from typing import Tuple


class Dataset():
    def __init__(self, X: np.array, y: np.array, max_bins: int=255, verbose: bool = False, random_state = None):
        X, y = check_X_y(X, y, dtype=[np.float32], order='F')
        self._X = X
        self._y = y
        self.max_bins = max_bins
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        if X.shape[0] == 1 or X.shape[1] == 1:
            raise ValueError(
                'Passing only one sample or one feature is not supported yet. '
                'See numba issue #3569.'
            )

        if verbose:
            print(f"Binning {X.nbytes / 1e9:.3f} GB of data: ", end="",
                  flush=True)

        with Timer() as binning_timer:
            self.bin_mapper_ = BinMapper(max_bins=self.max_bins,
                                        random_state=random_state)
            self._X_binned = self.bin_mapper_.fit_transform(self._X)

        if self.verbose:
            duration = binning_timer.interval
            throughput = X.nbytes / duration
            print(f"{duration:.3f} s ({throughput / 1e6:.3f} MB/s)")

    @property
    def shape(self) -> Tuple[int]:
        return self._X.shape

    @property
    def y(self):
        return self._y

    @property
    def X(self):
        return self._X

    @property
    def X_binned(self):
        return self._X_binned

    @property
    def numerical_thresholds(self):
        return self.bin_mapper_.numerical_thresholds_

    @property
    def n_bins_per_feature(self):
        return self.bin_mapper_.n_bins_per_feature_
