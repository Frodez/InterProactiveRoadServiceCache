from dataclasses import dataclass
import json
from typing import List, Set, Tuple
import scipy.stats as st
import numpy as np


DIRECT_HIT=0#在服务基站直接命中缓存
INDIRECT_HIT=1#在广播基站间接命中缓存
MISSED=2#未命中缓存


@dataclass
class RoadEnvConfig:

    totalBases: int  # 基站总数
    broadBases: int  # 每次收到请求时广播的半径，以收到广播的基站个数计算
    baseRange: float  # 基站的服务半径
    cacheSize: int  # 基站的缓存大小
    historyLength: int  # 基站保存历史请求的条目数
    averageDistance: float  # 车辆之间的平均间距
    speed: float  # 车辆的行驶速度
    timeStep: float  # 模拟生成请求的最小时间间隔
    episodeDuration: float  # 一次episode经过的时间间隔
    requestIntervals: np.ndarray  # 与上次请求内容相关的请求时间间隔
    contentProbabilities: np.ndarray  # 与上次请求内容相关的请求内容出现概率
    seed: int  # 随机数种子，可选

    def __init__(self, jsonStr: str = None) -> None:
        if jsonStr != None:
            def numpyHook(raw: dict):
                dct = {}
                for k, v in raw.items():
                    if isinstance(v, list):
                        dct[k] = np.array(v)
                    else:
                        dct[k] = v
                return dct
            self.__dict__ = json.loads(jsonStr, object_hook=numpyHook)

    @staticmethod
    def load(path):
        with open(path) as f:
            jsonStr = f.read()
        return RoadEnvConfig(jsonStr).validate()

    def validate(self):
        assert self.totalBases > 0
        assert self.broadBases >= 0
        assert self.baseRange > 0
        assert self.cacheSize > 0
        assert self.historyLength > 0
        assert self.averageDistance > 0
        assert self.speed > 0
        assert self.timeStep > 0
        assert self.episodeDuration > 0
        assert self.requestIntervals is not None
        assert self.contentProbabilities is not None
        assert len(self.requestIntervals.shape) == 1 and len(self.contentProbabilities.shape) == 2
        assert self.contentProbabilities.shape[0] == self.contentProbabilities.shape[
            1] and self.contentProbabilities.shape[0] == self.requestIntervals.shape[0]
        for i in range(self.contentProbabilities.shape[0]):
            assert abs(self.contentProbabilities[i].sum() - 1) <= 1e-6
        assert np.amax(self.requestIntervals) * self.speed < self.totalBases * self.baseRange
        return self

    def __str__(self) -> str:
        def numpyHook(obj):
            return obj.tolist() if isinstance(obj, np.ndarray) else json.JSONEncoder.default(self, obj)
        return json.dumps(self.__dict__, default=numpyHook)


def randomRequestIntervals(size: int, averageInterval: float = 1, bias=0.05, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    requestIntervals = generator.random((size)) + bias * averageInterval
    requestIntervals = requestIntervals / (requestIntervals.sum() * averageInterval / size)
    return requestIntervals


def randomContentProbabilities(size: int, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    contentProbabilities = generator.random((size, size))
    for i in range(size):
        contentProbabilities[i] = contentProbabilities[i] / contentProbabilities[i].sum()
    return contentProbabilities


def normalContentProbabilities(size: int, expectedChoices: int, scale: float = 1.0, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    contentProbabilities = np.zeros(shape=(size, size))
    cdfs = np.zeros((size + 1))
    for i in range(size):
        cdfs.fill(0)
        choices = max(generator.poisson(expectedChoices), 1)  # avoid zero choices
        locations = generator.random(size=(choices,)) * size
        for j in range(choices):
            cdfs = cdfs + st.norm.cdf(np.arange(size + 1), loc=locations[j], scale=scale)
        contentProbabilities[i] = cdfs[1:] - cdfs[:-1] + (1e-6 / size)  # base probability
        contentProbabilities[i] = contentProbabilities[i] / contentProbabilities[i].sum()
    return contentProbabilities


def zipfContentProbabilities(size:int, a: float, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    assert a > 1 + 1e-6
    contents = np.arange(size)
    generator.shuffle(contents)
    # 离散分布，且值域正好为正整数(由于生成出的数组包括0，所以要去掉第一个)
    cdfs: np.ndarray = st.zipf.pmf(np.arange(size + 1), a=a)[1:]
    probabilities: np.ndarray = cdfs[contents]
    probabilities = probabilities / probabilities.sum()
    contentProbabilities = np.zeros(shape=(size, size))
    for i in range(size):
        contentProbabilities[i] = probabilities
    return contentProbabilities


def definiteContentProbabilities(size: int, expectedChoices: int, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    contentProbabilities = np.zeros((size, size))
    for i in range(size):
        choices = max(generator.poisson(expectedChoices), 1)
        choices = min(choices, size)
        locations = generator.choice(np.arange(size), choices, replace=False)
        contentProbabilities[i][locations] = 1 / choices
    return contentProbabilities
