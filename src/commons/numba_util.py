from typing import List
import numpy as np
from numba import jit

from src.commons import road_util


@jit
def comb(n: int, k: int) -> int:
    res = 1
    for i in range(n, n - k, -1):
        res = res * i
    for i in range(2, k + 1):
        res = res / i
    return int(res)


@jit
def convertAction(cacheSize: int, replaces: int, operationId: int) -> List[int]:
    operations = []
    now = 0
    left = replaces
    while left != 0:
        n = cacheSize - now - 1
        k = left - 1
        c = comb(n, k)
        while n != 0 and operationId >= c:
            operationId = operationId - c
            now = now + 1
            n = n - 1
            c = comb(n, k)
        operations.append(now)
        now = now + 1
        left = left - 1
    return operations


@jit
def refreshArray(arr: np.ndarray, data):
    arr = np.roll(arr, -1)
    arr[-1] = data
    return arr

@jit
def getReward(replaced: bool, currentHit: int, currentPosition: float, baseRange: float):
    # ideal reward = -(Eexchange * Rlocal + (Emissed * (1 - hit)) * (1 - isBroadCast))
    exchangePenalty = 0 #-0.0625
    directedHitPenalty = 0 #-0.0625
    indirectedHitPenalty = -16.0
    missedPenalty = -64.0
    reward = 0.0
    if replaced == True: # 如果进行了替换
        reward = reward + exchangePenalty
    if currentPosition >= -0.5 * baseRange and currentPosition < 0.5 * baseRange:
        if currentHit == road_util.MISSED:
            reward = reward + missedPenalty
        elif currentHit == road_util.INDIRECT_HIT:
            reward = reward + indirectedHitPenalty
        else:
            reward = reward + directedHitPenalty
    return reward
