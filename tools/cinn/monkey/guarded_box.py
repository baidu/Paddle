from typing import TypeVar, Generic
from dataclasses import dataclass
T = TypeVar('T')

@dataclass
class GuardedBox(Generic[T]):
    hash_key: int
    value: T

    def Get(self, hash_key: int):
        assert self.hash_key == hash_key
        return self.value
