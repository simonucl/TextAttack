"""
Word Swap
-------------------------------
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

"""
import random
import string

from textattack.transformations import Transformation
from collections import defaultdict
from tqdm import tqdm

class WordSwapDuo(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """

    def __init__(self, verbalizer, letters_to_insert=None, **kwargs):
        super().__init__(**kwargs)
        if type(verbalizer) is dict:
            label_map = []
            for k, v in verbalizer.items():
                label_map.append(v if type(v) is not list else v[0])
        else:
            label_map = verbalizer
        self.label_map = label_map

        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        if word in self.label_map:
            return list(set(self.label_map) - set([word]))
        return []

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        label2indices = defaultdict(list)
        for i in indices_to_modify:
            word_to_replace = words[i]
            if word_to_replace in self.label_map:
                label2indices[word_to_replace].append(i)

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words_i = self._get_replacement_words(word_to_replace)

            seen = set()
            transformed_texts_idx = []
            for r in replacement_words_i:
                if r == word_to_replace:
                    continue
                replacement_indices = label2indices[r]
                for j in replacement_indices:
                    if (j == i) or ((i, j) in seen) or ((j, i) in seen):
                        continue
                    transformed_texts_idx.append(current_text.replace_words_at_indices([i, j], [r, word_to_replace]))
                    seen.add((i, j))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts
