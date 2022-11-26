from ast import Tuple
import collections
from typing import Dict, OrderedDict


class LFUCache:
    def __init__(self, initialDict: Dict):
        self.least_freq = 1
        self.node_for_freq: Dict[int, OrderedDict[int, Tuple[int, int]]] = collections.defaultdict(collections.OrderedDict)
        self.node_for_key: Dict[int, Tuple[int, int]] = {}
        for value, key in enumerate(initialDict):
            self.node_for_key[key] = (value,1)
            self.node_for_freq[1][key] = (value,1)

    def get(self, key: int):
        if key in self.node_for_key:
            value, freq = self.node_for_key[key]
            self.node_for_freq[freq].pop(key)
            if len(self.node_for_freq[self.least_freq]) == 0:
                self.least_freq += 1
            self.node_for_freq[freq+1][key] = (value, freq+1)
            self.node_for_key[key] = (value, freq+1)
            return value

    def replace(self, key: int):
        replacedKey, valueAndFreq = self.node_for_freq[self.least_freq].popitem(last=False)
        poppedValue = valueAndFreq[0]
        self.node_for_key.pop(replacedKey)
        self.node_for_key[key] = (poppedValue,1)
        self.node_for_freq[1][key] = (poppedValue,1)
        self.least_freq = 1
        return poppedValue

