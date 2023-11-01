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
    InsturctionSwapOrder,
    InsturctionIrrelevant
)

from .attack_recipe import AttackRecipe


class IrrelevantSampleAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, ood_dataset):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        transformation = CompositeTransformation(
            [
                InsturctionIrrelevant(ood_dataset=ood_dataset, modifying_sample=1)
            ]
        )

        constraints = []
        # constraints = [InstructionModification(['inference', 'Example_'])]
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
