import abc
from typing import Any, Tuple, Optional, List


class Option():
    def __init__(self):
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
    def __init__(self, default_value, max_value=None, nullable=False):
        self._default_value = default_value
        self.max_value = max_value
        self.nullable = nullable
        super().__init__()

    def _process_value(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, float):
            raise ValueError("value must be float")
        if value <= 0:
            raise ValueError("value must be positive")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"value must be no more than {self.max_value}")
        return value

    @property
    def default_value(self):
        return self._default_value


class PositiveIntegerOption(Option):
    def __init__(self, default_value, max_value=None, nullable=False):
        self._default_value = default_value
        self.max_value = max_value
        self.nullable = nullable
        super().__init__()

    def _process_value(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, int):
            raise ValueError("value must be integer")
        if value <= 0:
            raise ValueError("value must be positive")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"value must be no more than {self.max_value}")
        return value

    @property
    def default_value(self):
        return self._default_value


class BooleanOption(Option):
    def __init__(self, default_value, nullable=False):
        assert isinstance(default_value, bool)
        self._default_value = default_value
        self.nullable = nullable
        super().__init__()

    def _process_value(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, bool):
            raise ValueError("value must be boolean")
        return value

    @property
    def default_value(self):
        return self._default_value


class StringOption(Option):
    def __init__(self, default_value, available_options: Optional[List[str]]=None, nullable=False):
        self._default_value = default_value
        self.available_options = available_options
        self.nullable = nullable
        super().__init__()

    def _process_value(self, value: Any):
        if value is None and self.nullable:
            return value
        if not isinstance(value, str):
            raise ValueError("value must be string")
        if self.available_options is not None and value not in self.available_options:
            raise ValueError(f"value must be one of [{','.join(self.available_options)}]")
        return value

    @property
    def default_value(self):
        return self._default_value
