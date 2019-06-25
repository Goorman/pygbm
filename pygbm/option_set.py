import abc
from typing import Dict
from pygbm import options as O


class OptionSet():
    def __init__(self, **kwargs):
        self.parameter_dict = self.get_parameter_dict()

        for key, value in kwargs:
            if key not in self.parameter_dict:
                continue
            try:
                self.parameter_dict[key].set_value(value)
            except ValueError as e:
                raise ValueError(f"Incorrect value {value} for parameter {key}: {str(e)}")

    @abc.abstractmethod
    def get_parameter_dict(self) -> Dict[str, O.Option]:
        pass

    def __getitem__(self, item):
        return self.parameter_dict[item].value


class GradientBoostingClassifierOptionSet(OptionSet):
    def get_parameter_dict(self):
        return {
            'learning_rate': O.PositiveFloatOption(default_value=1.0),
            'max_iter': O.PositiveIntegerOption(default_value=100),
            'max_leaf_nodes': O.PositiveIntegerOption(default_value=31, nullable=True),
            'max_depth': O.PositiveIntegerOption(default_value=10, nullable=True),
            'min_samples_leaf': O.PositiveIntegerOption(default_value=20),
            'w_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'b_l2_reg': O.PositiveFloatOption(default_value=1.0),
            'max_bins': O.PositiveIntegerOption(default_value=255, max_value=255),
            'n_iter_no_change': O.PositiveIntegerOption(default_value=5, nullable=True),
            'tol': O.PositiveFloatOption(default_value=1e-7),
            'random_state': O.PositiveIntegerOption(default_value=None, nullable=True),
            'verbose': O.BooleanOption(default_value=False),
            'scoring': O.StringOption(default_value=None, nullable=True),
            'loss': O.StringOption(
                default_value="auto",
                available_options=[
                    'binary_crossentropy', 'categorical_crossentropy', 'auto'],
                nullable=False
            )
        }


class GradientBoostingRegressorOptionSet(OptionSet):
    def get_parameter_dict(self):
        return None
        # TODO FILL WITH PARAMETERS
