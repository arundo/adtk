from typing import Any


class MyBase:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class MySubclass(MyBase):
    def __init__(self, thing: int = 1) -> None:
        super().__init__(thing=thing)
        self.thing = thing  # type: int

    def show_thing(self) -> int:
        return self.thing

