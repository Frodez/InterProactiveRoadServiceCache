import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rolloutBufferLen = 2048*32
    saveFreq = rolloutBufferLen*10
    evaluationFreq = rolloutBufferLen*1
    with open("./models/evaluation_result") as f:
        lines = f.readlines()
        directHitRatios = []
        indirectHitRatios = []
        avgRewards = []
        indexs = list(range(evaluationFreq, (len(lines) + 1) * evaluationFreq, evaluationFreq))
        for index, line in enumerate(lines):
            evalution_result = json.loads(line)
            directHitRatios.append(evalution_result["directHit"] / evalution_result["total"])
            indirectHitRatios.append(evalution_result["indirectHit"] / evalution_result["total"])
            avgRewards.append(evalution_result["avgReward"])
        plt.subplot(121)
        plt.plot(indexs, directHitRatios, "r", indexs, indirectHitRatios, "y")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.subplot(122)
        plt.plot(indexs, avgRewards, "b")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig("./models/evalution_result_by_steps.png")