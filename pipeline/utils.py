import itertools
from logging import getLogger
import time
from typing import Any, Dict, Iterator, List, Tuple, Union

from torch.autograd.grad_mode import _DecoratorContextManager

logger = getLogger('__main__')


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


class timer(_DecoratorContextManager):
    """Context-manager that logs elapsed time of a process.
    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    Paramters
    ---------
    message : str
        The displayed message.

    Examples
    --------
    - Usage as a context-manager

        >>> with timer('read csv'):
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        [read csv] start.
        [read csv] done in 0.1 min.

    - Usage as a decorator

        >>> @timer()
        >>> def read_csv():
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        >>>     return train_df
        >>>
        >>> train_df = read_csv()
        [read_csv] start.
        [read_csv] done in 0.1 min.
    """

    def __init__(self, message: str = None):
        self.message = message

    def __call__(self, function):
        if self.message is None:
            self.message = function.__name__
        super().__call__(function)

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f'[{self.message}] start.')

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        elapsed_time = time.time() - self.start_time
        logger.info(f'[{self.message}] done in {elapsed_time / 60:.1f} min.')
