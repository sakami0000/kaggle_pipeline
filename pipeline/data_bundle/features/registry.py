import functools
import gc
from logging import getLogger
import inspect
from pathlib import PosixPath, Path
from typing import Callable, List, Union

import pandas as pd

from ...utils import timer

logger = getLogger('__main__')


def load_pickle(
    path: Union[str, PosixPath, List[str], List[PosixPath]]
) -> Union[None, pd.Series, pd.DataFrame, List[Union[None, pd.Series, pd.DataFrame]]]:
    """Load pickle format file or files.
    Returns a list of pd.Series or pd.DataFrame which is the same length as input `path`.
    If the specified file doesn't exist, returns `None`.
    """
    if isinstance(path, (str, PosixPath)):
        if Path(path).exists():
            data = pd.read_pickle(path)
            gc.collect()
            return data
    else:
        data = [
            pd.read_pickle(p) if p.exists() else None
            for p in path
        ]
        gc.collect()
        return data


class _FeatureRegistry(object):
    """Stores feature generation functions.
    The function registered here can be used by specifying in the config file.
    This class is already instantiated as "feature_registry"
    and all functions are read from this object.
    Features registered here are saved/loaded in pickle format.
    The directory where features are saved can be managed by setting attributes.

    See Also
    --------
    pipeline.runner.hooks.registry._RunnerRegistry

    Notes
    -----
    The name of a function should be same as its feature name.

    Attributes
    ----------
    check_function_definition : bool, default = False
        If true, check if function definition has not changed from
        previous cache in each load.
    cache_dir : pathlib.PosixPath
        Parent directory to save features.
    train_feature_dir : pathlib.PosixPath
        Directory to save train features.
    test_feature_dir : pathlib.PosixPath
        Directory to save test features.
    function_dir : pathlib.PosixPath
        Direcotry to save funciont definition.

    `cache_dir` is the parent directory of `train_feature_dir`, `test_feature_dir` and `function_dir`.

        cache_dir/
        ├── train_feature_dir/
        ├── test_feature_dir/
        └── function_dir/

    The default directory tree is following:

        ├── main.py
        ├── pipeline/
        └── input/
            └── feature/                <-- `cache_dir`
                ├── train_feature/      <-- `train_feature_dir`
                ├── test_feature/       <-- `test_feature_dir`
                └── function/           <-- `function_dir`

    Examples
    --------
    1. Define function and register it in "feature_registry".

        >>> from pipeline import feature_registry
        >>>
        >>> @feature_registry.register
        ... def timestamp():
        ...     train_time = pd.read_csv(TRAIN_PATH, usecols=['timestamp'])['timestamp']
        ...     test_time = pd.read_csv(TEST_PATH, usecols=['timestamp'])['timestamp']
        ...         return train_time, test_time

    2. Specify the defined function in cofiguration file.

        >>> features:
        >>>   - timestamp
    """

    def __init__(self):
        self.feature_functions = {}
        self.check_function_definition = False
        self._cache_dir = Path(__file__).parent / 'input/feature/'
        self._set_cache_directories()

    def _set_cache_directories(self):
        self._train_feature_dir = self._cache_dir / 'train_feature/'
        self._test_feature_dir = self._cache_dir / 'test_feature/'
        self._function_dir = self._cache_dir / 'function/'

    def _make_cache_directories(self):
        self._train_feature_dir.mkdir(parents=True, exist_ok=True)
        self._test_feature_dir.mkdir(parents=True, exist_ok=True)
        self._function_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: Union[str, PosixPath]):
        self._cache_dir = Path(cache_dir)
        self._set_cache_directories()

    @property
    def train_feature_dir(self):
        return self._train_feature_dir

    @train_feature_dir.setter
    def train_feature_dir(self, train_feature_dir: Union[str, PosixPath]):
        self._train_feature_dir = Path(train_feature_dir)

    @property
    def test_feature_dir(self):
        return self._test_feature_dir

    @test_feature_dir.setter
    def test_feature_dir(self, test_feature_dir: Union[str, PosixPath]):
        self._test_feature_dir = Path(test_feature_dir)

    @property
    def function_dir(self):
        return self._function_dir

    @function_dir.setter
    def function_dir(self, function_dir: Union[str, PosixPath]):
        self._function_dir = Path(function_dir)

    def register(self, function: Callable):
        """Register a function.
        This is also used for saving and loading features in pickle format.
        The name of decorated function will be same as file name.
        The decorated function must return two (train/test) pd.Series or pd.DataFrame objects.
        If the definition of the function has changed, rerun it and cache its results.

        Parameters
        ----------
        function : Callable
            The function to be registered.
        """
        self.feature_functions[function.__name__] = function

        @functools.wraps(function)
        def _pickle_cache(*args, **kwargs):
            file_name = function.__name__
            self._make_cache_directories()
            train_file = self._train_feature_dir / f'{file_name}.pkl'
            test_file = self._test_feature_dir / f'{file_name}.pkl'
            function_file = self._function_dir / f'{file_name}.txt'
            train_feature, test_feature = load_pickle([train_file, test_file])

            if train_feature is not None:
                if self.check_function_definition:
                    # check if function definition hasn't changed
                    with open(function_file, 'r') as f:
                        function_str = f.read()
                    if function_str == inspect.getsource(function):
                        logger.debug(f'{file_name} loaded.')
                        return train_feature, test_feature
                    
                else:
                    logger.debug(f'{file_name} loaded.')
                    return train_feature, test_feature

                logger.info(f'{file_name} definition has been changed. recreating.')

            with timer(f'creating {file_name}'):
                train_feature, test_feature = function(*args, **kwargs)
                train_feature.to_pickle(train_file)
                if test_feature is not None:
                    test_feature.to_pickle(test_file)

                if self.check_function_definition:
                    # cache function definition
                    with open(function_file, 'w') as f:
                        f.write(inspect.getsource(function))

            gc.collect()
            return train_feature, test_feature

        return _pickle_cache
