from collections import OrderedDict
from typing import Dict
 
class LRUCache:
 
    # initialising capacity
    def __init__(self, initialDict: Dict):
            self.cache: OrderedDict[int, int] = OrderedDict()
            for value, key in enumerate(initialDict):
                self.cache[key] = value

    def get(self, key: int):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

    def replace(self, key: int):
        poppedValue = self.cache.popitem(last = False)[1]
        self.cache[key] = poppedValue
        self.cache.move_to_end(key)
        return poppedValue
