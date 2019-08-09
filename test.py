# -*- coding: utf-8 -*-
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn as nn
from torch.nn.functional import pad

# gensim_model = KeyedVectors.load_word2vec_format(
#     'data/GoogleNews-vectors-negative300.bin', binary=True, limit=30000)
import numpy as np

#
# music_df = pandas.read_json('data/music/Digital_Music_5.json', lines=True)
# kindle_df = pandas.read_json('data/kindle_store/Kindle_Store_5.json',
#                              lines=True)


# word_id = [gensim_model.vocab[x].index for x in words]
word_id = torch.LongTensor([30000])
# print(word_id)
embedding = torch.FloatTensor(gensim_model.vectors)
zero_tensor = torch.zeros(size=embedding[:1].size())
embedding = torch.cat((embedding, zero_tensor), dim=0)
print(embedding.shape)
embedding = nn.Embedding.from_pretrained(embedding)
vectors = embedding(word_id)
print(vectors.shape)
print(vectors)





