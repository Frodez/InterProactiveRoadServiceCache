from copy import deepcopy
from typing import Callable, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import sys

import gym
from gym import spaces
import numpy as np

from src.commons import road_util, numba_util


class RoadEnv(gym.Env):

    def __init__(self, config: road_util.RoadEnvConfig) -> None:
        config.validate()

        self._config = deepcopy(config)
        self.totalBases = config.totalBases
        self.broadBases = config.broadBases
        self.baseRange = config.baseRange
        self.cacheSize = config.cacheSize
        self.historyLength = config.historyLength
        self.episodeDuration = config.episodeDuration

        self.requestIntervals = np.array(config.requestIntervals)
        self.contentProbabilities = np.array(config.contentProbabilities)
        generator = np.random.default_rng(config.seed)
        self.rng = generator

        totalContents = config.requestIntervals.shape[0]
        self.totalContents = totalContents

        operationOutSize = config.cacheSize + 1 # [0, cacheSize-1]为进行缓存替换，cacheSize为不进行缓存替换
        operationInSize = totalContents

        self.observation_space = spaces.Dict({
            "content": spaces.MultiDiscrete(np.repeat(totalContents, config.historyLength), seed=config.seed),
            #"prevContent": spaces.MultiDiscrete(np.repeat(totalContents, config.historyLength), seed=config.seed),
            "position": spaces.Box(low=-(config.broadBases + 0.5) * config.baseRange, high=0.5 * config.baseRange, shape=(config.historyLength,), seed=config.seed),
            #"prevPosition": spaces.Box(low=-(config.broadBases + 0.5) * config.baseRange, high=0.5 * config.baseRange, shape=(config.historyLength,), seed=config.seed),
            # 缓存不应该存相同内容
            "cachedContents": spaces.MultiDiscrete(np.repeat(totalContents, config.cacheSize), seed=config.seed)
        })

        action_space = spaces.MultiDiscrete([operationOutSize, operationInSize], seed=config.seed)

        self.action_space = action_space

        self.requestGenerator = RequestGenerator(config.averageDistance, config.speed, config.totalBases * config.baseRange, config.timeStep, config.requestIntervals,
                                                 config.contentProbabilities, generator)

        self.bases: List[OrderedDict] = []
        self.baseHits: List[int] = [road_util.MISSED for _ in range(config.totalBases)]

        self.reset()

    def reset(self) -> OrderedDict:
        # 清空并重新生成所有基站的历史请求记录，随机设置缓存内容，重置请求生成器
        self.bases.clear()
        for i in range(self.totalBases):
            base = self.observation_space.sample()
            base['cachedContents'] = self.rng.choice(self.totalContents, self.cacheSize, replace=False)
            self.bases.append(base)
        self.requestGenerator.reset()
        trainBaseIndex = self.totalBases // 2
        return self.bases[trainBaseIndex]

    def step(self, action: np.ndarray, modelFn: Callable[[OrderedDict, int], np.ndarray]) -> Tuple[OrderedDict, float, bool, dict]:
        # action:该基站(最中间的基站)的缓存替换动作
        # modelFn:其他基站的决策模型
        # 环境中储存了其他基站的历史请求记录和当前缓存内容
        # 使用modelFn，根据其他基站的历史请求记录和缓存内容决定其他基站的缓存替换动作
        # 使用请求生成器生成下一次请求，并对下一次请求广播范围内所有基站（若涉及该基站则除外）使用modelFn进行决策
        # 若请求不涉及该基站，则继续生成下一次请求
        # 若请求涉及该基站，则返回该基站的历史请求记录（本次请求为记录中最新者）和缓存内容
        trainBaseIndex = self.totalBases // 2  # 训练模型对应的基站为最中间的基站
        reward = self._doAction(trainBaseIndex, action)  # 做动作并计算reward
        while True:
            # 一直生成模拟请求，并根据modelFn进行决策更新各基站状态，若请求涉及到训练模型对应的基站则终止并返回
            content, position, prevContent, prevPosition, hit, lowIndex, highIndex, directBaseIndex = self._nextRequest()
            end = False
            for index in range(lowIndex, highIndex):
                # 更新请求涉及到的所有基站的状态
                relativePosition = position - (index + 0.5) * self.baseRange  # 使用相对位置
                relativePrevPosition = prevPosition - (index + 0.5) * self.baseRange  # 使用相对位置
                self._putCurrentRequest(index, content, relativePosition, prevContent, relativePrevPosition, hit)
                if index == trainBaseIndex:
                    # 为训练对应的基站，则跳过决策过程，并在更新所有基站后返回
                    end = True
                    continue
                else:
                    actionForOthers = modelFn(self.bases[index], index)
                    self._doAction(index, actionForOthers)
            if end == True:
                done = self.episodeDuration <= self.requestGenerator.absoluteTime()
                return self.bases[trainBaseIndex], reward, done, {"hit": hit, "directly": trainBaseIndex == directBaseIndex}

    def render(self):
        pass

    def close(self):
        pass

    @staticmethod
    def load(path):
        return RoadEnv(road_util.RoadEnvConfig.load(path))

    def getConfig(self) -> road_util.RoadEnvConfig:
        return deepcopy(self._config)

    def getAllBaseObs(self):
        return self.bases

    def _nextRequest(self) -> Tuple[int, float, int, float, int, int, int, int]:
        content, position, prevContent, prevPosition = self.requestGenerator.nextRequest()
        directBaseIndex = int(position / self.baseRange)
        # 计算广播的基站范围，只向右广播
        lowIndex = directBaseIndex
        highIndex = min(self.totalBases, directBaseIndex + self.broadBases + 1)
        hit = road_util.MISSED
        if np.any(self.bases[directBaseIndex]["cachedContents"] == content):
            hit = road_util.DIRECT_HIT
        else:
            for index in range(lowIndex, highIndex):
                if index == directBaseIndex:
                    continue
                elif np.any(self.bases[index]["cachedContents"] == content):
                    hit = road_util.INDIRECT_HIT
        return content, position, prevContent, prevPosition, hit, lowIndex, highIndex, directBaseIndex

    def _putCurrentRequest(self, baseIndex: int, content: int, position: float, prevContent: int, prevPosition: float, hit: int) -> None:
        base = self.bases[baseIndex]
        base["content"] = numba_util.refreshArray(base["content"], content)
        base["position"] = numba_util.refreshArray(base["position"], position)
        #base["prevContent"] = numba_util.refreshArray(base["prevContent"], prevContent)
        #base["prevPosition"] = numba_util.refreshArray(base["prevPosition"], prevPosition)
        self.baseHits[baseIndex] = hit

    def _doAction(self, baseIndex: int, action: np.ndarray) -> float:
        outId: int = action[0]
        inId: int = action[1]
        base = self.bases[baseIndex]
        cachedContents: np.ndarray = base["cachedContents"]
        # outId为cacheSize时，或者如果换入内容在缓存中，不进行替换
        replaced = bool(outId != self.cacheSize and np.any(cachedContents == inId) == False)
        if replaced == True:
            cachedContents[outId] = inId
        return numba_util.getReward(replaced, self.baseHits[baseIndex], base["position"][-1], self.baseRange)


class RequestGenerator(object):

    def __init__(self, averageDistance: float, speed: float, roadLength: float, timeStep: float, requestIntervals: np.ndarray, contentProbabilities: np.ndarray, generator=np.random.default_rng()) -> None:

        self.averageDistance = averageDistance
        self.speed = speed
        self.roadLength = roadLength
        self.requestIntervals = requestIntervals
        self.contentProbabilities = contentProbabilities
        self.rng = generator

        # 避免刚生成的车辆直接进入车道
        self.startPosition = 0 - 1.1 * speed * np.amax(requestIntervals)
        self.totalContents = requestIntervals.shape[0]
        self.contents = np.arange(requestIntervals.shape[0])
        self.lengthStep = timeStep * speed
        self.produceProbability = timeStep * speed / averageDistance

        self.processLength = 0.0
        self.units: List[_Unit] = []

    def reset(self):
        self.processLength = 0.0
        self.units.clear()

    def absoluteTime(self) -> float:
        return self.processLength / self.speed

    def nextRequest(self) -> Tuple[int, float, int, float]:
        while True:
            selectedIndex = -1
            step = sys.float_info.max
            # 查找在场车中下次请求在车道内，且与当前位置距离最短者
            for index, unit in enumerate(self.units):
                diff = unit.positionOnNextRequest - unit.positionNow
                if diff < step:
                    step = diff
                    selectedIndex = index
            # 若找到符合条件的车辆，则该车辆为发出请求的车辆
            if selectedIndex != -1:
                selectedUnit = self.units[selectedIndex]
                # 快速推进rounds个时间段
                rounds = int(step / self.lengthStep)
                self.processLength = self.processLength + step
                for unit in self.units:
                    unit.positionNow = unit.positionNow + step
                # 删除车道右边的车辆，由于快速推进，所以在此删除效率更高
                self.units = list(filter(lambda unit: unit.positionNow < self.roadLength, self.units))

                # 将该时间段(round个timeStep)内新出现的车辆加入进来
                # 由于采用二项分布判断是否应加入新车辆，故在极限情况下车辆的出现过程属于泊松过程
                for i in np.where(self.rng.binomial(1, self.produceProbability, rounds) == 1)[0]:
                    initContent = self._randomContent()
                    # 由于车道前还有一部分道路，故车辆不会直接进入车道
                    initPosition = self.startPosition + self.lengthStep * i
                    unit = _Unit(initContent, initPosition, self._nextPosition(initPosition, initContent), initPosition)
                    self.units.append(unit)
                # 若发出请求位置在车道内，则返回
                content = self._nextContent(selectedUnit.contentOnRequest)
                position = selectedUnit.positionNow + step
                prevContent = selectedUnit.contentOnRequest
                prevPosition = selectedUnit.positionOnRequest
                selectedUnit.contentOnRequest = content
                selectedUnit.positionOnRequest = position
                selectedUnit.positionOnNextRequest = self._nextPosition(position, content)
                if position >= 0 and position < self.roadLength:
                    return content, position, prevContent, prevPosition
            else:
                # 向前推进一个timeStep(用lengthStep表示)
                self.processLength = self.processLength + self.lengthStep
                for unit in self.units:
                    unit.positionNow = unit.positionNow + self.lengthStep
                # 将该timeStep内新出现的车辆加入进来
                # 由于采用二项分布判断是否应加入新车辆，故在极限情况下车辆的出现过程属于泊松过程
                if self.rng.binomial(1, self.produceProbability) == 1:
                    initContent = self._randomContent()
                    # 由于车道前还有一部分道路，故车辆不会直接进入车道
                    initPosition = self.startPosition
                    unit = _Unit(initContent, initPosition, self._nextPosition(initPosition, initContent), initPosition)
                    self.units.append(unit)

    def _randomContent(self):
        # 初始化时随机选择请求内容
        return self.rng.integers(0, self.totalContents)

    def _nextContent(self, currentContent: int) -> int:
        # 根据转移概率选择下次请求的内容
        return self.rng.choice(self.contents, p=self.contentProbabilities[currentContent])

    def _nextPosition(self, currentPosition: float, currentContent: int) -> float:
        # 上次请求的内容会决定下次请求与上次请求的间隔，且是固定值
        return currentPosition + self.requestIntervals[currentContent] * self.speed

@dataclass
class _Unit(object):

    contentOnRequest: int
    positionOnRequest: float
    positionOnNextRequest: float
    positionNow: float
