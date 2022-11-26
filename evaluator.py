import os

import torch
import matplotlib.pyplot as plt

from src.algorithm.custom_ppo import CustomCombinedExtractor, CustomVecEnv, CustomPPO
from src.commons.evaluation_util import EvaluationResult, lfuEvaluate, lruEvaluate, modelEvaluate, randomEvaluate
from src.commons.road_util import RoadEnvConfig
from src.environment import RoadEnv

if __name__ == '__main__':
    #threadNum = 2
    threadNum = os.cpu_count()
    torch.set_num_threads(threadNum)
    config = RoadEnvConfig.load("./models/config.json")
    config.seed = None
    #config.broadBases = 0
    #config.episodeDuration = 600
    env = RoadEnv(config)
    envWrapper = CustomVecEnv(env)
    model = CustomPPO("MultiInputPolicy", envWrapper, policy_kwargs={
                      "features_extractor_class": CustomCombinedExtractor}, seed=env._config.seed, device="cpu")
    try:
        model = model.load(path="./models/model.zip", env=envWrapper)
        print("use the previous model")
    except:
        print("use the fresh model")

    evaluateTimes = 100

    results: list[EvaluationResult] = []
    results.append(modelEvaluate(model, envWrapper, evaluateTimes))
    results.append(lruEvaluate(envWrapper, evaluateTimes))
    results.append(lfuEvaluate(envWrapper, evaluateTimes))
    results.append(randomEvaluate(envWrapper, evaluateTimes))
    for result in results:
        print(result)

    envWrapper.close()

    plt.subplot(121)
    plt.scatter(["IPRSC", "LRU", "LFU", "Random"], [(result.directHit + result.indirectHit) / (result.directHit + result.indirectHit + result.missed) for result in results])
    plt.subplot(122)
    plt.scatter(["IPRSC", "LRU", "LFU", "Random"], [result.directHit / (result.directHit + result.indirectHit + result.missed) for result in results])
    #plt.subplot(123)
    #plt.scatter(["IPRSC", "LRU", "LFU", "Random"], [result.avgReward for result in results])
    plt.savefig("./models/cache_hit_ratios.png")
