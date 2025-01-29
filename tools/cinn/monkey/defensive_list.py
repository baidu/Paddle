from typing import TypeVar, Generic, List, Callable, Iterator, Tuple
from guarded_box import GuardedBox
from hash_combine import HashCombine
from dataclasses import dataclass

K = TypeVar('K')
V = TypeVar('V')

# Defensive list
@dataclass
class DList(Generic[K, V]):
    defensive_list: List[Tuple[K, GuardedBox[V]]]

    def __init__(self, keys: List[K], values: List[V]):
        assert len(keys) == len(values)
        self.defensive_list = []
        hash_value = 0
        for i in range(len(keys)):
            hash_value = HashCombine(hash_value, hash(keys[i]))
            self.defensive_list.append((keys[i], GuardedBox(hash_value, values[i])))

    def Unguard(self) -> Iterator[Tuple[K, V]]:
        hash_value = 0
        for key, guarded_value in self.defensive_list:
            hash_value = HashCombine(hash_value, hash(key))
            yield key, guarded_value.Get(hash_value)
