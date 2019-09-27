"""Module for all utility functions.

"""


def _get_all_subclasses_from_superclass(superclass):
    result = dict()
    for sb in superclass.__subclasses__():
        if sb.__name__[0] != "_":
            result.update({sb.__name__: sb.__doc__})
        else:
            result.update(_get_all_subclasses_from_superclass(sb))
    return result


def _print_all_subclass_from_superclass(superclass):
    subclasses = _get_all_subclasses_from_superclass(superclass)
    for key, value in subclasses.items():
        print("-" * 80)
        print(key)
        print(value)
