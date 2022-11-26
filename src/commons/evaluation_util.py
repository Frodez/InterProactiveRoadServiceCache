from copy import deepcopy
from dataclasses import dataclass
import json
from typing import List, Set, Tuple
import numpy as np

from src.algorithm.custom_ppo import CustomPPO, CustomVecEnv
from src.commons import road_util
from src.environment import RoadEnv
from src.commons.lfu_cache import LFUCache
from src.commons.lru_cache import LRUCache


@dataclass
class EvaluationResult:
    total: int
    directHit: int
    indirectHit: int
    missed: int
    avgReward: float
    lifetimeHit: int
    lifetimeMissed: int
    clairvoyantHit: int
    clairvoyantMissed: int

    def __str__(self) -> str:
        return json.dumps(self.__dict__)


def modelEvaluate(model: CustomPPO, envWrapper: CustomVecEnv, times=1) -> EvaluationResult:
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    totalRewards = 0
    lifetimeHit = 0
    lifetimeMissed = 0
    clairvoyantHit = 0
    clairvoyantMissed = 0
    for i in range(times):
        obs = envWrapper.reset()
        cachedContents: List[Set[int]] = [
            set(getFromObs(obs, "cachedContents").flatten())]
        requestContents: List[int] = []
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = envWrapper.step(
                action, lambda obs, index: model.predict(obs, deterministic=True)[0])
            total = total + 1
            totalRewards = totalRewards + rewards[0]
            info = infos[0]
            if info["directly"] == True:
                cachedContents.append(
                    set(getFromObs(obs, "cachedContents").flatten()))
                requestContents.append(getFromObs(obs, "content")[-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if dones[0]:
                break

        res = lifetimeEvaluate(cachedContents, requestContents)
        lifetimeHit = lifetimeHit + res[0]
        lifetimeMissed = lifetimeMissed + res[1]
        res = clairvoyantEvaluate(cachedContents[0], requestContents)
        clairvoyantHit = clairvoyantHit + res[0]
        clairvoyantMissed = clairvoyantMissed + res[1]
    return EvaluationResult(total=total, directHit=directHit, indirectHit=indirectHit, missed=missed,
                            avgReward=totalRewards / total, lifetimeHit=lifetimeHit, lifetimeMissed=lifetimeMissed,
                            clairvoyantHit=clairvoyantHit, clairvoyantMissed=clairvoyantMissed)


def lruEvaluate(envWrapper: CustomVecEnv, times=1):
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    totalRewards = 0
    lifetimeHit = 0
    lifetimeMissed = 0
    clairvoyantHit = 0
    clairvoyantMissed = 0
    for i in range(times):
        obs = envWrapper.reset()
        env: RoadEnv = envWrapper.envs[0]
        allCaches: List[LRUCache] = []
        for obsForBase in env.getAllBaseObs():
            initialDict = {}
            for j, content in enumerate(obsForBase["cachedContents"]):
                initialDict[content] = j
            allCaches.append(LRUCache(initialDict))

        def lruPredict(obs, index=env.totalBases // 2):
            currentPosition = getFromObs(obs, "position")[-1]
            if currentPosition < -0.5 * env.baseRange or currentPosition >= 0.5 * env.baseRange:
                return np.array([[env.cacheSize, 0]]), None
            currentContent = getFromObs(obs, "content")[-1]
            cache = allCaches[index]
            index = cache.get(currentContent)
            if index == None:
                replacedIndex = cache.replace(currentContent)
                return np.array([[replacedIndex, currentContent]]), None
            else:
                return np.array([[env.cacheSize, 0]]), None

        cachedContents: List[Set[int]] = [
            set(getFromObs(obs, "cachedContents").flatten())]
        requestContents: List[int] = []
        while True:
            action, _states = lruPredict(obs)
            obs, rewards, dones, infos = envWrapper.step(
                action, lambda obs, index: lruPredict(obs)[0][0])
            total = total + 1
            totalRewards = totalRewards + rewards[0]
            info = infos[0]
            if info["directly"] == True:
                cachedContents.append(
                    set(getFromObs(obs, "cachedContents").flatten()))
                requestContents.append(getFromObs(obs, "content")[-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if dones[0]:
                break

        res = lifetimeEvaluate(cachedContents, requestContents)
        lifetimeHit = lifetimeHit + res[0]
        lifetimeMissed = lifetimeMissed + res[1]
        res = clairvoyantEvaluate(cachedContents[0], requestContents)
        clairvoyantHit = clairvoyantHit + res[0]
        clairvoyantMissed = clairvoyantMissed + res[1]
    return EvaluationResult(total=total, directHit=directHit, indirectHit=indirectHit, missed=missed,
                            avgReward=totalRewards / total, lifetimeHit=lifetimeHit, lifetimeMissed=lifetimeMissed,
                            clairvoyantHit=clairvoyantHit, clairvoyantMissed=clairvoyantMissed)


def lfuEvaluate(envWrapper: CustomVecEnv, times=1):
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    totalRewards = 0
    lifetimeHit = 0
    lifetimeMissed = 0
    clairvoyantHit = 0
    clairvoyantMissed = 0
    for i in range(times):
        obs = envWrapper.reset()
        env: RoadEnv = envWrapper.envs[0]
        allCaches: List[LFUCache] = []
        for obsForBase in env.getAllBaseObs():
            initialDict = {}
            for j, content in enumerate(obsForBase["cachedContents"]):
                initialDict[content] = j
            allCaches.append(LFUCache(initialDict))

        def lfuPredict(obs, index=env.totalBases // 2):
            currentPosition = getFromObs(obs, "position")[-1]
            if currentPosition < -0.5 * env.baseRange or currentPosition >= 0.5 * env.baseRange:
                return np.array([[env.cacheSize, 0]]), None
            currentContent = getFromObs(obs, "content")[-1]
            cache = allCaches[index]
            index = cache.get(currentContent)
            if index == None:
                replacedIndex = cache.replace(currentContent)
                return np.array([[replacedIndex, currentContent]]), None
            else:
                return np.array([[env.cacheSize, 0]]), None

        cachedContents: List[Set[int]] = [
            set(getFromObs(obs, "cachedContents").flatten())]
        requestContents: List[int] = []
        while True:
            action, _states = lfuPredict(obs)
            obs, rewards, dones, infos = envWrapper.step(
                action, lambda obs, index: lfuPredict(obs)[0][0])
            total = total + 1
            totalRewards = totalRewards + rewards[0]
            info = infos[0]
            if info["directly"] == True:
                cachedContents.append(
                    set(getFromObs(obs, "cachedContents").flatten()))
                requestContents.append(getFromObs(obs, "content")[-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if dones[0]:
                break

        res = lifetimeEvaluate(cachedContents, requestContents)
        lifetimeHit = lifetimeHit + res[0]
        lifetimeMissed = lifetimeMissed + res[1]
        res = clairvoyantEvaluate(cachedContents[0], requestContents)
        clairvoyantHit = clairvoyantHit + res[0]
        clairvoyantMissed = clairvoyantMissed + res[1]
    return EvaluationResult(total=total, directHit=directHit, indirectHit=indirectHit, missed=missed,
                            avgReward=totalRewards / total, lifetimeHit=lifetimeHit, lifetimeMissed=lifetimeMissed,
                            clairvoyantHit=clairvoyantHit, clairvoyantMissed=clairvoyantMissed)


def randomEvaluate(envWrapper: CustomVecEnv, times=1):
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    totalRewards = 0
    lifetimeHit = 0
    lifetimeMissed = 0
    clairvoyantHit = 0
    clairvoyantMissed = 0
    for i in range(times):
        obs = envWrapper.reset()
        env = envWrapper.envs[0]

        cachedContents: List[Set[int]] = [
            set(getFromObs(obs, "cachedContents").flatten())]
        requestContents: List[int] = []
        while True:
            action, _states = np.array([env.action_space.sample()]), None
            obs, rewards, dones, infos = envWrapper.step(
                action, lambda obs, index: env.action_space.sample())
            total = total + 1
            totalRewards = totalRewards + rewards[0]
            info = infos[0]
            if info["directly"] == True:
                cachedContents.append(
                    set(getFromObs(obs, "cachedContents").flatten()))
                requestContents.append(getFromObs(obs, "content")[-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if dones[0]:
                break

        res = lifetimeEvaluate(cachedContents, requestContents)
        lifetimeHit = lifetimeHit + res[0]
        lifetimeMissed = lifetimeMissed + res[1]
        res = clairvoyantEvaluate(cachedContents[0], requestContents)
        clairvoyantHit = clairvoyantHit + res[0]
        clairvoyantMissed = clairvoyantMissed + res[1]
    return EvaluationResult(total=total, directHit=directHit, indirectHit=indirectHit, missed=missed,
                            avgReward=totalRewards / total, lifetimeHit=lifetimeHit, lifetimeMissed=lifetimeMissed,
                            clairvoyantHit=clairvoyantHit, clairvoyantMissed=clairvoyantMissed)


def getFromObs(obs, key) -> np.ndarray:
    return obs[key][0] if len(obs[key].shape) == 2 else obs[key]


def clairvoyantEvaluate(cachedContents: Set[int], requestContents: List[int]) -> Tuple[int, int]:
    hit = 0
    missed = 0
    cachedContents = deepcopy(cachedContents)
    for index, request in enumerate(requestContents):
        if request in cachedContents:
            hit = hit + 1
        else:
            missed = missed + 1
            contentsNotRequested = cachedContents.copy()
            for futureRequest in requestContents[index + 1:]:
                if len(contentsNotRequested) != 1:
                    if futureRequest in contentsNotRequested:
                        contentsNotRequested.remove(futureRequest)
                else:
                    break
            replacedContent = next(iter(contentsNotRequested))
            cachedContents.remove(replacedContent)
            cachedContents.add(request)
    return hit, missed


def lifetimeEvaluate(cachedContents: List[Set[int]], requestContents: List[int]) -> Tuple[int, int]:
    lifetimeHit = 0
    lifetimeMissed = 0
    cachedContents = deepcopy(cachedContents)
    length = len(cachedContents)
    for index in range(0, length):
        for content in cachedContents[index]:
            hit = 0
            for epoch in range(index + 1, length):
                if requestContents[epoch - 1] == content:
                    hit = hit + 1
                if content not in cachedContents[epoch]:
                    break
                else:
                    cachedContents[epoch].remove(content)
            if hit != 0:
                lifetimeHit = lifetimeHit + 1
            else:
                lifetimeMissed = lifetimeMissed + 1
    return lifetimeHit, lifetimeMissed
