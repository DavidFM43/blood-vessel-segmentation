
from typing import Any, List, Union

from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS

# Annoyingly, RandomState(seed) requires seed to be in [0, 2 ** 32 - 1] (an
# unsigned int), while RandomState.randint only accepts and returns signed ints.
MAX_INT32 = 2**31
MIN_INT32 = -MAX_INT32

SeedType = Union[int, list, np.ndarray]


def _signed_to_unsigned(seed: SeedType) -> SeedType:
  if isinstance(seed, int):
    return seed + 2**32 if seed < 0 else seed
  if isinstance(seed, list):
    return [s + 2**32 if s < 0 else s for s in seed]
  if isinstance(seed, np.ndarray):
    return np.array([s + 2**32 if s < 0 else s for s in seed.tolist()])


def split(seed: SeedType, num: int = 2) -> SeedType:
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  return rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=[num, 2])


def fold_in(seed: SeedType, data: Any) -> List[Union[SeedType, Any]]:
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  new_seed = rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32)
  return [new_seed, data]


def PRNGKey(seed: SeedType) -> SeedType:  # pylint: disable=invalid-name
  return split(seed, num=2)[0]
