from typing import Optional


class BaseTreePredictor():
    def __init__(self):
        pass

    def get_n_leaf_nodes(self):
        pass

    def get_max_depth(self):
        pass

    def predict(self, X):
        pass


class BaseTreeGrower():
    def __init__(self):
        pass

    def grow(self) -> None:
        pass

    def make_predictor(self, numerical_thresholds) -> BaseTreePredictor:
        pass

    def update_raw_predictions(self, raw_predictions) -> None:
        pass

    @property
    def total_apply_split_time(self) -> Optional[float]:
        return None

    @property
    def total_find_split_time(self) -> Optional[float]:
        return None
