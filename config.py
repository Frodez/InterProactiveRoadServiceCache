from src.commons.road_util import RoadEnvConfig, normalContentProbabilities, zipfContentProbabilities, randomRequestIntervals
from src.environment import *
from src.algorithm import *
from src.commons import *

if __name__ == '__main__':
    requestIntervals = randomRequestIntervals(160, averageInterval=1.0, bias=0.05)
    contentProbabilities = zipfContentProbabilities(160, a=1.3)
    #contentProbabilities = normalContentProbabilities(100, expectedChoices=2, scale=0.25)

    config = RoadEnvConfig()

    config.totalBases = 4
    config.broadBases = 1
    config.baseRange = 100.0
    config.cacheSize = 16
    config.historyLength = 16
    config.averageDistance = 90.0
    config.speed = 30.0
    config.timeStep = 0.01
    config.episodeDuration = 480
    config.seed = 0
    config.requestIntervals = requestIntervals
    config.contentProbabilities = contentProbabilities

    with open("./models/config.json", mode="w") as f:
        f.write(config.__str__())