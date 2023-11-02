"""
Swap the order of sentence in the Instructions
==========================================================
"""

import random
from itertools import permutations

from textattack.transformations import Transformation
import numpy as np

class InsturctionIrrelevant(Transformation):
    """Transformation that randomly swaps the order of words in a sequence."""

    def __init__(self, ood_dataset, modifying_percent=None, modifying_sample=None):
        """
        Please note that you need to specify the path to the corpus from which the OOD inputs will be sampled.
        It should be a .txt file where each line contains a sentence (plain text).
        """
        self.ood_dataset = ood_dataset
        ood_dataset_len = [len(x.split()) for x in self.ood_dataset]
        self.ood_dataset_len = np.array(ood_dataset_len)
        self.modifying_percent = modifying_percent
        self.modifying_sample = modifying_sample

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        text_dict = current_text.text_dict
        modified_keys = current_text.attack_attrs['modified_keys']

        if 'Premise_0' in text_dict:
            keys = 'Premise'
            demon_len = (len(text_dict) - 2) // 3
        elif 'Example_0' in text_dict:
            keys = 'Example'
            demon_len = (len(text_dict) - 1) // 2

        # demon_len = (len(text_dict) - len(keys) ) // (len(keys) + 1)

        if self.modifying_sample is not None:
            modifying_key = self.modifying_sample
        elif self.modifying_percent is not None:
            modifying_key = self.modifying_percent * demon_len

        # new_train_data = []
        # for dp in test_data:
        #     l = len(dp["input"].split())
        #     prob = np.exp(-np.power(random_text_lens-l, 2)/50)
        #     prob /= np.sum(prob)
        #     samples = np.random.choice(random_texts, size=args.k, replace=False, p=prob)
        #     assert len(samples)==len(train_data)
        #     new_train_data.append([])
        #     for train_dp, sample in zip(train_data, samples):
        #         new_train_data[-1].append({"task": train_dp["task"],
        #                                     "input": sample,
        #                                     "output": train_dp["output"],
        #                                     "options": train_dp["options"]})
        
        key_collections = [f"{keys}_{i}" for i in range(demon_len)]
        key_collections = set(key_collections) - set(modified_keys)

        
        # combination of key_collections with modifying_key number of keys
        key_collections = list(permutations(key_collections, modifying_key))
        print(key_collections)

        for keys in key_collections:
            replacing_texts = []
            for key in keys:
                l = len(text_dict[f"{key}"].split())
                prob = np.exp(-np.power(self.ood_dataset_len-l, 2)/50)
                prob /= np.sum(prob)
                samples = np.random.choice(self.ood_dataset, size=1, replace=False, p=prob)
                replacing_texts.append(samples[0])
            transformed_texts.append(current_text.replace_text_at_key(keys, replacing_texts))

        return transformed_texts

    @property
    def deterministic(self):
        return False