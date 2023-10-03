"""

Metrics on AttackSuccessRate
---------------------------------------------------------------------

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
import numpy as np

class AdvConfidence(Metric):
    def __init__(self):
        self.attack_conf = []
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to Adversarial Confidence.
        
        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """

        self.results = results

        for i, results in enumerate(self.results):
            if isinstance(results, FailedAttackResult):
                continue
            elif isinstance(results, SkippedAttackResult):
                continue
            else:
                self.attack_conf.append(results.perturbed_result.score)
        self.all_metrics["attack_conf"] = self.mean_attack_conf()
        return self.all_metrics
        
    def mean_attack_conf(self):
        attack_conf = np.mean(self.attack_conf)
        attack_conf = round(attack_conf, 2)
        return attack_conf