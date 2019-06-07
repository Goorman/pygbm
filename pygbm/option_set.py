import abc
from typing import Dict
from pygbm import options as O


class OptionSet():
    def __init__(self, **kwargs):
        self.parameter_dict = self.get_parameter_dict()

        for key, value in kwargs:
            if key not in self.parameter_dict:
                continue
            is_correct, msg = self.parameter_dict[key].set_value(value)

            if not is_correct:
                raise ValueError(f"Incorrect value {value} for parameter {key}: {msg}")

    @abc.abstractmethod
    def get_default_parameter_dict(self) -> Dict[str, O.Option]:
        pass

    def __getitem__(self, item):
        return self.parameter_dict[item].value


class GradientBoostingClassifierOptionSet(OptionSet):
    def get_default_parameter_dict(self):
        return {
            'learning_rate': O.PositiveFloatOption(default_value=1.0),
            'max_iter': O.PositiveIntegerOption(default_value=100),
            'max_leaf_nodes': O.PositiveIntegerOption(default_value=31),
            'max_depth': O.PositiveIntegerOption(default_value=10),
            'min_samples_leaf': O.PositiveIntegerOption(default_value=20),
            'w_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'b_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'max_bins': O.PositiveIntegerOption(default_value=255, max_value=255),
            'n_iter_no_change': O.PositiveIntegerOption(default_value=5),
            'tol': O.PositiveFloatOption(default_value=1e-7),
            'random_state': O.PositiveIntegerOption(default_value=None),
            'verbose': O.BooleanOption(default_value=False),
            'scoring': None, #TODO
            'loss': None, #TODO
        }


class GradientBoostingRegressorOptionSet(OptionSet):
    def get_default_parameter_dict(self):
        return None
        # TODO FILL WITH PARAMETERS

