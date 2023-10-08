"""

Input Column Modification
--------------------------

"""

from textattack.constraints import PreTransformationConstraint


class InstructionModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words within a specific
    input column.

    For example, can prevent modification of 'premise' during
    entailment.
    """

    def __init__(self, columns_to_ignore=["inference", "Label_"]):
        self.columns_to_ignore = columns_to_ignore

    def is_columns_to_ignore(self, column_name):
        for column in self.columns_to_ignore:
            if column in column_name:
                return True
        return False
    
    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        deleted.

        If ``current_text.column_labels`` doesn't match
            ``self.matching_column_labels``, do nothing, and allow all words
            to be modified.

        If it does match, only allow words to be modified if they are not
            in columns from ``columns_to_ignore``.
        """
        idx = 0
        indices_to_modify = set()
        for column, words in zip(
            current_text.column_labels, current_text.words_per_input
        ):
            num_words = len(words)
            if not self.is_columns_to_ignore(column):
                indices_to_modify |= set(range(idx, idx + num_words))
            idx += num_words
        return indices_to_modify

    def extra_repr_keys(self):
        return ["columns_to_ignore"]
