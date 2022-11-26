import os
import json

import torch
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

from src.algorithm.custom_ppo import CustomCombinedExtractor, CustomVecEnv, CustomPPO
from src.commons.evaluation_util import modelEvaluate
from src.environment import RoadEnv

class CustomModelCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, evaluation_freq: int = 0, verbose: int = 0):
        super(CustomModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.evaluation_freq = evaluation_freq
        self.evaluation_results = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.evaluation_freq > 0 and self.n_calls % self.evaluation_freq == 0:
            if isinstance(self.model, CustomPPO):
                print("start to evaluate")
                evaluation_result = modelEvaluate(self.model, self.model.get_env(), times=1)
                self.evaluation_results.append(evaluation_result)
                path = os.path.join(self.save_path, f"evaluation_result")
                with open(path, mode="a") as f:
                    f.writelines(evaluation_result.__str__() + '\n')
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


if __name__ == '__main__':
    threadNum = os.cpu_count()
    torch.set_num_threads(threadNum)
    env = RoadEnv.load("./models/config.json")
    envWrapper = CustomVecEnv(env)

    rolloutBufferLen = 2048*32
    saveFreq = rolloutBufferLen*10
    evaluationFreq = rolloutBufferLen*1
    totalTimesteps = rolloutBufferLen*150
    
    callback = CustomModelCallback(save_freq=saveFreq, save_path='./models/', evaluation_freq=evaluationFreq)
    
    model = CustomPPO("MultiInputPolicy", envWrapper, n_steps=rolloutBufferLen, verbose=1, policy_kwargs={"features_extractor_class": CustomCombinedExtractor}, seed=env._config.seed, device="cpu")
    try:
        model = model.load(path="./models/model.zip", env=envWrapper)
        print("use the prevent model")
    except:
        print("use the fresh model")
        pass
    model.learn(total_timesteps=totalTimesteps, callback=callback)

    try:
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
    except:
        pass