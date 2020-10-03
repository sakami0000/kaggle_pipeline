import itertools
from typing import Dict, Iterator, List, Tuple, Union


def get_all_subclasses(obj: Union[type, List[type], Tuple[type]]) -> Iterator[type]:
    """Generate all subclasses recursively.

    Parameters
    ----------
    obj : Union[type, List[type], Tuple[type]]
        Super class or list of super classes.

    Yields
    -------
    Iterator[type]
        Subclasses of the given class.
    """
    if isinstance(obj, (list, tuple)):
        subclasses = itertools.chain.from_iterable([o.__subclasses__() for o in obj])
    else:
        subclasses = obj.__subclasses__()

    for subclass in subclasses:
        yield from get_all_subclasses(subclass)
        yield subclass


def get_subclass_map(obj: Union[type, List[type], Tuple[type]]) -> Dict[str, type]:
    """Returns a dictionary object which stores all subclasses of the given class.

    Parameters
    ----------
    obj : Union[type, List[type], Tuple[type]]
        Super class or list of super classes.

    Returns
    -------
    Dict[str, type]
        A dictionary which maps name string -> object of subclasses of the given class.
    """
    return {
        sub_obj.__name__: sub_obj
        for sub_obj in get_all_subclasses(obj)
    }
