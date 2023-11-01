"""

TextBugger
===============

(TextBugger: Generating Adversarial Text Against Real-world Applications)

"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed
from textattack.constraints.pre_transformation.instruction_attack import InstructionModification

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapLabel,
    WordSwapDuo
)
from textattack.constraints.overlap import UniformSwap

from .attack_recipe import AttackRecipe


class SwapLabel2023(AttackRecipe):
    """Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).

    TextBugger: Generating Adversarial Text Against Real-world Applications.

    https://arxiv.org/abs/1812.05271
    """

    @staticmethod
    def build(model_wrapper, verbalizer, fix_dist=False):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        WORDSWAP = WordSwapDuo if fix_dist else WordSwapLabel
        transformation = CompositeTransformation(
            [
                WORDSWAP(verbalizer=verbalizer)
            ]
        )

        constraints = [InstructionModification(['sentence', 'Example_', 'Premise', 'Hypothesis']), RepeatModification(), MaxWordsPerturbed(max_percent=1.0)]
        
        constraints.append(UniformSwap(threshold=1))

        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
