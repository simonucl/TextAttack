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
        text_dict = current_text.text_dict
        modified_keys = current_text.attack_attrs['modified_keys']

        # print(text_dict)
        if 'Premise_0' in text_dict:
            keys = ['Premise', 'Hypothesis']
        elif 'Example_0' in text_dict:
            keys = ['Example']

        demon_len = (len(text_dict) - len(keys) ) // (len(keys) + 1)

        keys.append('Label')
        
        for i in range(demon_len):
            transformed_texts_idx = []
            for j in range(i+1, demon_len):
                if f"{keys[0]}_{i}" in modified_keys or f"{keys[0]}_{j}" in modified_keys:
                    continue
                else:
                    key_iter = []
                    text_iter = []
                    for k in keys:
                        key_iter.append(f"{k}_{i}")
                        text_iter.append(text_dict[f"{k}_{j}"])
                        key_iter.append(f"{k}_{j}")
                        text_iter.append(text_dict[f"{k}_{i}"])
                    # print(key_iter)
                    transformed_texts_idx.append(current_text.replace_text_at_key(key_iter, text_iter))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    @property
    def deterministic(self):
        return False