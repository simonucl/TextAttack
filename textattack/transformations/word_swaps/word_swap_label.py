"""
Word Swap by Embedding
-------------------------------

Based on paper: `<arxiv.org/abs/1603.00892>`_

Paper title: Counter-fitting Word Vectors to Linguistic Constraints

"""
from .word_swap import WordSwap


class WordSwapLabel(WordSwap):
    """Transforms an input by replacing its words with synonyms in the word
    embedding space.

    Args:
        max_candidates (int): maximum number of synonyms to pick
        embedding (textattack.shared.AbstractWordEmbedding): Wrapper for word embedding
    >>> from textattack.transformations import WordSwapEmbedding
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapEmbedding()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_map = {
            "sentiment": ["positive", "negative"],
            "nli": ["entailment", "neutral", "contradiction"],
            "topic": [
                "World",
                "Sports",
                "Business",
                "Sci/Tech",
            ]
        }

    def _get_replacement_words(self, word):
        """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on the label_map
        """
        for k, v in self.label_map.items():
            if word in v:
                return list(set(v) - set([word]))
        return []

    def extra_repr_keys(self):
        return ["label_map"]
