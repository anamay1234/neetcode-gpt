import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        # Combine all sentences so we can build ONE vocabulary
        sentences = positive + negative

        # Collect every unique word
        words = set()
        for sentence in sentences:
            words.update(sentence.split())

        # Sort alphabetically and assign IDs starting at 1
        # enumerate(..., start=1) means the first word gets ID 1
        vocabulary = {
            word: idx
            for idx, word in enumerate(sorted(words), start=1)
        }

        # Convert each sentence from words -> integer IDs
        encoded = []

        for sentence in sentences:
            ids = [vocabulary[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float32))

        # Pad shorter sentences with 0
        # batch_first=True gives shape (number_of_sentences, max_length)
        result = nn.utils.rnn.pad_sequence(
            encoded,
            batch_first=True,
            padding_value=0
        )

        return result
