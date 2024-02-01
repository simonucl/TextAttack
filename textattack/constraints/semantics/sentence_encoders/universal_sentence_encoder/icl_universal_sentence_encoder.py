import torch
"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import (
    UniversalSentenceEncoder)
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class IclUniversalSentenceEncoder(UniversalSentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, large=large, metric=metric, **kwargs)

    def _sim_score(self, starting_text, transformed_text):
        starting_text_instructions = starting_text.instructions
        transformed_text_instructions = transformed_text.instructions
        # filter those that are the same
        modified_instructions = [(i, j) for i, j in zip(starting_text_instructions, transformed_text_instructions) if i != j]
        # flatten the list
        modified_instructions = [item for sublist in modified_instructions for item in sublist]
        # if there are no modified instructions, then return 1
        if len(modified_instructions) == 0:
            return 1
        embeddings = self.encode(modified_instructions)
        min_score = 1
        for i in range(len(modified_instructions) // 2):
            starting_embedding = embeddings[i]
            transformed_embedding = embeddings[i + 1]
            if not isinstance(starting_embedding, torch.Tensor):
                starting_embedding = torch.tensor(starting_embedding)

            if not isinstance(transformed_embedding, torch.Tensor):
                transformed_embedding = torch.tensor(transformed_embedding)

            starting_embedding = torch.unsqueeze(starting_embedding, dim=0)
            transformed_embedding = torch.unsqueeze(transformed_embedding, dim=0)            
            # compare the embeddings of the instructions
            # if they are too similar, then return 0
            min_score = min(min_score, self.sim_metric(starting_embedding, transformed_embedding))
        return min_score
    
    def _score_list(self, starting_text, transformed_texts):
        # Return an empty tensor if transformed_texts is empty.
        # This prevents us from calling .repeat(x, 0), which throws an
        # error on machines with multiple GPUs (pytorch 1.2).
        if len(transformed_texts) == 0:
            return torch.tensor([])
        starting_text_instructions = starting_text.instructions
        transformed_texts_instructions = [t.instructions for t in transformed_texts] # list of lists

        instructions = starting_text_instructions + [item for sublist in transformed_texts_instructions for item in sublist]

        embeddings = self.encode(instructions)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        starting_embeddings = embeddings[: len(starting_text_instructions)]
        transformed_embeddings = embeddings[len(starting_text_instructions) :]
        min_scores = [1] * len(transformed_texts)
        for k in range(len(transformed_texts_instructions)):
            len_instructions = len(transformed_texts_instructions[k])
            transformed_k_instructions = transformed_embeddings[k * len_instructions: (k + 1) * len_instructions] # shape: (len_instructions, 512)
            sim_score = self.sim_metric(starting_embeddings, transformed_k_instructions) # shape: (len_instructions, len(transformed_texts))
            print("Sim score:", sim_score)
            min_scores[k] = torch.min(sim_score, dim=0).values
        print("Min scores:", min_scores)

        return torch.tensor(min_scores)
