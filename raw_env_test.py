from typing import List, Set
import numpy as np
from time import time
from src.commons import road_util
from src.commons.evaluation_util import clairvoyantEvaluate
from src.commons.lfu_cache import LFUCache
from src.environment import RoadEnv
from src.commons.lru_cache import LRUCache
from src.commons.road_util import RoadEnvConfig, zipfContentProbabilities


if __name__ == '__main__':
    evaluateTimes = 10

    config = RoadEnvConfig.load("./models/config.json")
    #config.baseRange = 20
    config.seed = None
    #config.averageDistance = 180
    #config.contentProbabilities = zipfContentProbabilities(100, 1.2)
    env = RoadEnv(config)

    #'''
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    rewards = 0
    idealHit = 0
    idealMissed = 0
    start = time()
    for i in range(evaluateTimes):
        obs = env.reset()
        allCaches: List[LRUCache] = []
        allObs = env.getAllBaseObs()
        for obsForI in allObs:
            cachedContents: np.ndarray = obsForI["cachedContents"]
            initialDict = {}
            for j, content in enumerate(cachedContents):
                initialDict[content] = j
            allCaches.append(LRUCache(initialDict))
        def lruPredict(obs, index=config.totalBases // 2):
            currentPosition = obs["position"][-1]
            if currentPosition < -0.5 * env.baseRange or currentPosition >= 0.5 * env.baseRange:
                return np.array([env.cacheSize, 0])
            currentContent = obs["content"][-1]
            cache = allCaches[index]
            index = cache.get(currentContent)
            if index == None:
                replacedIndex = cache.replace(currentContent)
                return np.array([replacedIndex, currentContent])
            else:
                return np.array([config.cacheSize, 0])
        cachedContents: Set[int] = set(obs["cachedContents"].flatten())
        requestContents: List[int] = []
        while True:
            action = lruPredict(obs)
            obs, reward, done, info = env.step(action, lambda obs, index: lruPredict(obs))
            total = total + 1
            rewards = rewards + reward
            if info["directly"] == True:
                requestContents.append(obs["content"][-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if done:
                break
        ideal = clairvoyantEvaluate(cachedContents, requestContents)
        idealHit = idealHit + ideal[0]
        idealMissed = idealMissed + ideal[1]
    interval = time() - start
    print("time={}, total={}, speed={}".format(interval, total, total / interval))
    print("ideal result: hit={}, missed={}".format(idealHit, idealMissed))
    print("lru result: avg_reward={}, total={}, directHit={}, indirectHit={}, missed={}".format(rewards / total, total, directHit, indirectHit, missed))
    #'''

    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    rewards = 0
    idealHit = 0
    idealMissed = 0
    start = time()
    for i in range(evaluateTimes):
        obs = env.reset()
        allCaches: List[LFUCache] = []
        allObs = env.getAllBaseObs()
        for obsForI in allObs:
            cachedContents: np.ndarray = obsForI["cachedContents"]
            initialDict = {}
            for j, content in enumerate(cachedContents):
                initialDict[content] = j
            allCaches.append(LFUCache(initialDict))
        def lfuPredict(obs, index=config.totalBases // 2):
            currentPosition = obs["position"][-1]
            if currentPosition < -0.5 * env.baseRange or currentPosition >= 0.5 * env.baseRange:
                return np.array([env.cacheSize, 0])
            currentContent = obs["content"][-1]
            cache = allCaches[index]
            index = cache.get(currentContent)
            if index == None:
                replacedIndex = cache.replace(currentContent)
                return np.array([replacedIndex, currentContent])
            else:
                return np.array([config.cacheSize, 0])
        cachedContents: Set[int] = set(obs["cachedContents"].flatten())
        requestContents: List[int] = []
        while True:
            action = lfuPredict(obs)
            obs, reward, done, info = env.step(action, lambda obs, index: lfuPredict(obs))
            total = total + 1
            rewards = rewards + reward
            if info["directly"] == True:
                requestContents.append(obs["content"][-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if done:
                break
        ideal = clairvoyantEvaluate(cachedContents, requestContents)
        idealHit = idealHit + ideal[0]
        idealMissed = idealMissed + ideal[1]
    interval = time() - start
    print("time={}, total={}, speed={}".format(interval, total, total / interval))
    print("ideal result: hit={}, missed={}".format(idealHit, idealMissed))
    print("lfu result: avg_reward={}, total={}, directHit={}, indirectHit={}, missed={}".format(rewards / total, total, directHit, indirectHit, missed))

    #'''
    total = 0
    directHit = 0
    indirectHit = 0
    missed = 0
    rewards = 0
    idealHit = 0
    idealMissed = 0
    start = time()
    for i in range(evaluateTimes):
        obs = env.reset()
        cachedContents: Set[int] = set(obs["cachedContents"].flatten())
        requestContents: List[int] = []
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action, lambda obs, index: env.action_space.sample())
            total = total + 1
            rewards = rewards + reward
            if info["directly"] == True:
                requestContents.append(obs["content"][-1])
                if info["hit"] == road_util.DIRECT_HIT:
                    directHit = directHit + 1
                elif info["hit"] == road_util.INDIRECT_HIT:
                    indirectHit = indirectHit + 1
                else:
                    missed = missed + 1
            if done:
                break
        ideal = clairvoyantEvaluate(cachedContents, requestContents)
        idealHit = idealHit + ideal[0]
        idealMissed = idealMissed + ideal[1]
    interval = time() - start
    print("time={}, total={}, speed={}".format(interval, total, total / interval))
    print("ideal result: hit={}, missed={}".format(idealHit, idealMissed))
    print("random result: avg_reward={}, total={}, directHit={}, indirectHit={}, missed={}".format(rewards / total, total, directHit, indirectHit, missed))
    # '''
    
    env.close()