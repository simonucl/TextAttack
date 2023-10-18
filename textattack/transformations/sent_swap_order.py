"""
Swap the order of sentence in the Instructions
==========================================================
"""

import random
from itertools import permutations

from textattack.transformations import Transformation

class InsturctionSwapOrder(Transformation):
    """Transformation that randomly swaps the order of words in a sequence."""

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        text = current_text.text
        inputs = text.split('\n')
        no_demonstration = (len(inputs) - 1) // 2
        # Generate the permutation of the instructions
        permutations = list(permutations(range(no_demonstration)))

        # Iterate over the permutation
        for order in permutations:
            # Generate the new text
            new_text = ''
            for i in range(no_demonstration):
                new_text += inputs[2 * order[i]] + '\n'
                new_text += inputs[2 * order[i] + 1] + '\n'
            # Add the last line
            new_text += inputs[-1]
            transformed_texts.append(new_text)

        return transformed_texts

    @property
    def deterministic(self):
        return False