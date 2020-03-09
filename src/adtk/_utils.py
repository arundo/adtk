"""Module for all utility functions.

"""

from typing import Dict, Optional, Type


def _get_all_subclasses_from_superclass(
    superclass: Type
) -> Dict[str, Optional[str]]:
    result = dict()
    for sb in superclass.__subclasses__():
        if sb.__name__[0] != "_":
            result.update({sb.__name__: sb.__doc__})
        else:
            result.update(_get_all_subclasses_from_superclass(sb))
    return result


class PandasBugError(Exception):
    def __init__(self) -> None:
        msg = (
            """Pandas before v0.25 has a known bug in method `rolling` when """
            """parameter `window` is offset and `closed` is 'left'. Your """
            """current execution is impacted by this bug. If you are using """
            """Python 3.5.3 or later, please upgrade pandas to v0.25 or """
            """later. If you are using Python 3.5.2 or earlier, please """
            """consider using integer instead of offset to define the left """
            """rolling window."""
        )
        super().__init__(msg)
