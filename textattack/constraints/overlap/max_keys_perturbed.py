"""

Max Perturb Words Constraints
-------------------------------


"""

import math

from textattack.constraints import Constraint


class MaxKeysPerturbed(Constraint):
    """A constraint representing a maximum allowed perturbed words.

    Args:
        max_num_words (:obj:`int`, optional): Maximum number of perturbed words allowed.
        max_percent (:obj: `float`, optional): Maximum percentage of words allowed to be perturbed.
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(
        self, max_keys=None, max_percent=None, compare_against_original=True
    ):
        super().__init__(compare_against_original)
        if not compare_against_original:
            raise ValueError(
                "Cannot apply constraint MaxWordsPerturbed with `compare_against_original=False`"
            )

        if (max_keys is None) and (max_percent is None):
            raise ValueError("must set either `max_percent` or `max_num_words`")
        if max_percent and not (0 <= max_percent <= 1):
            raise ValueError("max perc must be between 0 and 1")
        self.max_keys = max_keys
        self.max_percent = max_percent

    def _check_constraint(self, transformed_text, reference_text):
        num_keys_diff = len(transformed_text.attack_attrs['modified_keys'])
        # num_words_diff = len(transformed_text.all_words_diff(reference_text))
        # if self.max_percent:
        #     min_num_words = min(len(transformed_text.words), len(reference_text.words))
        #     max_words_perturbed = math.ceil(min_num_words * (self.max_percent))
        #     max_percent_met = num_words_diff <= max_words_perturbed
        # else:
        #     max_percent_met = True
        if self.max_keys:
            max_num_keys_met = num_keys_diff < self.max_keys
        else:
            max_num_keys_met = True

        return max_num_keys_met

    def extra_repr_keys(self):
        metric = []
        if self.max_percent is not None:
            metric.append("max_percent")
        if self.max_keys is not None:
            metric.append("max_keys")
        return metric + super().extra_repr_keys()
