"""

Uniform Swapping Labels
--------------------------


"""


from textattack.constraints import Constraint
from collections import Counter, defaultdict

class UniformSwap(Constraint):
    """A constraint on BLEU score difference.

    Args:
        max_bleu_score (int): Maximum BLEU score allowed.
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, threshold=1, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(threshold, int):
            raise TypeError("max_bleu_score must be an int")
        self.threshold = threshold

    def _check_constraint(self, transformed_text, reference_text):
        hyp = transformed_text.text_dict
        label_count = defaultdict(int)
        i = 0
        while True:
            if f"Label_{i}" in hyp:
                label_count[hyp[f"Label_{i}"]] += 1
                i += 1
            else:
                break

        # the difference between each label count should be within the threshold
        return max(label_count.values()) - min(label_count.values()) <= self.threshold
    
    def extra_repr_keys(self):
        return ["threshold"] + super().extra_repr_keys()
