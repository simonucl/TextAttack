"""

TextBugger
===============

(TextBugger: Generating Adversarial Text Against Real-world Applications)

"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InstructionModification
)
from textattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, IclUniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_recipe import AttackRecipe


class ICLTextAttack(AttackRecipe):
    """Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).

    TextBugger: Generating Adversarial Text Against Real-world Applications.

    https://arxiv.org/abs/1812.05271
    """

    @staticmethod
    def build(model_wrapper):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification(), InstructionModification(columns_allowed=["Hypothesis_", "Example_"])]
        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
        constraints.append(IclUniversalSentenceEncoder(
            threshold=0.4,
            metric="angular",
            compare_against_original=True,
            skip_text_shorter_than_window=False,
            ))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
