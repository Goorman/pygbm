import abc
from typing import Any, Tuple


class Option():
    def __init__(self):
        self.option_value = self.default_value

    def set_value(self, value) -> Tuple[bool, str]:
        self.option_value, msg = self._process_value(value)

        return msg == '', msg

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
    def __init__(self, default_value, max_value=None):
        self._default_value = default_value
        self.max_value = max_value
        super().__init__()

    def _process_value(self, value):
        if not isinstance(value, float):
            return None, "value must be float"
        if value <= 0:
            return None, "value must be positive"
        if self.max_value is not None and value > self.max_value:
            return None, f"value must be no more than {self.max_value}"
        return value, ""

    @property
    def default_value(self):
        return self._default_value


class PositiveIntegerOption(Option):
    def __init__(self, default_value, max_value=None):
        self._default_value = default_value
        self.max_value = max_value
        super().__init__()

    def _process_value(self, value):
        if not isinstance(value, int):
            return None, "value must be integer"
        if value <= 0:
            return None, "value must be positive"
        if self.max_value is not None and value > self.max_value:
            return None, f"value must be no more than {self.max_value}"
        return value, ""

    @property
    def default_value(self):
        return self._default_value


class BooleanOption(Option):
    def __init__(self, default_value):
        assert isinstance(default_value, bool)
        self._default_value = default_value
        super().__init__()

    def _process_value(self, value):
        if not isinstance(value, bool):
            return None, "value must be boolean"
        return value, ""

    @property
    def default_value(self):
        return self._default_value