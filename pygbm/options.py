import abc
from typing import Any, Tuple, Optional, List, Dict


class Option():
    def __init__(self):
        self.option_value = None
        try:
            self.set_value(self.default_value)
        except ValueError as e:
            raise ValueError(f'Incorrect default value: {str(e)}')

    def set_value(self, value):
        self.option_value = self._process_value(value)

    @abc.abstractmethod
    def _process_value(self, value: Any) -> Tuple[Any, str]:
        pass

    @property
    def value(self):
        return self.option_value

    @property
    @abc.abstractmethod
    def default_value(self):
        pass


class PositiveFloatOption(Option):
    def __init__(self, default_value, max_value=None, none_value=None):
        self._default_value = default_value
        self.max_value = max_value
        self.none_value = none_value
        super().__init__()

    def _process_value(self, value):
        if value is None:
            return self.default_value
        if self.none_value is not None and self.none_value == value:
            return None
        if not isinstance(value, (int, float)):
            raise ValueError("value must be numerical")
        if value <= 0:
            raise ValueError("value must be positive")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"value must not be larger than {self.max_value}")
        return value

    @property
    def default_value(self):
        return self._default_value


class IntegerOption(Option):
    def __init__(self, default_value, min_value=None, max_value=None, none_value=None):
        self._default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.none_value = none_value
        super().__init__()

    def _process_value(self, value):
        if value is None:
            return self.default_value
        if self.none_value is not None and self.none_value == value:
            return None
        if not isinstance(value, int):
            raise ValueError("value must be integer")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"value must not be larger than {self.max_value}")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"value must not be lesser than {self.min_value}")
        return value

    @property
    def default_value(self):
        return self._default_value


class PositiveIntegerOption(IntegerOption):
    def __init__(self, default_value, max_value=None, none_value=None):
        super().__init__(default_value, min_value=1, max_value=max_value, none_value=none_value)


class BooleanOption(Option):
    def __init__(self, default_value, none_value=None):
        assert isinstance(default_value, bool)
        self._default_value = default_value
        self.none_value = none_value
        super().__init__()

    def _process_value(self, value):
        if value is None:
            return self.default_value
        if self.none_value is not None and self.none_value == value:
            return None
        if not isinstance(value, bool):
            raise ValueError("value must be boolean")
        return value

    @property
    def default_value(self):
        return self._default_value


class StringOption(Option):
    def __init__(self, default_value, available_options: Optional[List[str]]=None, none_value=None):
        self._default_value = default_value
        self.available_options = available_options
        self.none_value=none_value
        super().__init__()

    def _process_value(self, value: Any):
        if value is None:
            return self.default_value
        if self.none_value is not None and self.none_value == value:
            return None
        if not isinstance(value, str):
            raise ValueError("value must be string")
        if self.available_options is not None and value not in self.available_options:
            raise ValueError(f"value must be one of [{', '.join(self.available_options)}]")
        return value

    @property
    def default_value(self):
        return self._default_value


class OptionSet():
    def __init__(self, options: Dict[str, Option]):
        self.options = options

    def update_from_estimator(self, estimator):
        estimator_params = estimator.get_params()
        for key in self.options:
            param_value = estimator_params.get(key, None)
            if param_value is not None:
                self[key] = param_value

    def __getitem__(self, item):
        return self.options[item].value

    def __setitem__(self, item, value):
        try:
            self.options[item].set_value(value)
        except ValueError as e:
            raise ValueError(f'Incorrect value "{value}" for parameter "{item}": {str(e)}')
