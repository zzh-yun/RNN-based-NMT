"""Embeddings for the Neural Machine Translation (NMT) model.
Consists of word embeddings for one language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEmbeddings(nn.Module):
    """Class that converts input words to their embeddings.
    """

    def __init__(self, word_embed_size, vocabentry):
        """Init the Embedding layer for one language.

        Args:
            word_embed_size (int): Embedding size for the output word.
            vocabentry: Vocabulary entry for the language.
        """
        super().__init__()
        self.word_embed_size = word_embed_size
        self.embed = nn.Embedding(len(vocabentry.word2id), word_embed_size)

    def forward(self, x_padded):
        """Look up embeddings for the words in a batch of sentences.

        Args:
            x_padded (Tensor): Tensor of shape (sent_length, batch_size)
                of integers where each integer is an index into the word vocabulary.

        Returns:
            x_wordEmb (Tensor): Tensor of shape (sent_length, batch_size, word_embed_size),
                containing the embeddings for each word of the sentences in the batch.
        """
        return self.embed(x_padded)

#