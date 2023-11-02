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
        hyp = transformed_text.label_dist
        ref = reference_text.label_dist

        for k in hyp:
            if abs(hyp[k] - ref[k]) > self.threshold:
                return False
        return True
    
    def extra_repr_keys(self):
        return ["threshold"] + super().extra_repr_keys()
