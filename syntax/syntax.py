# -*- coding: utf-8 -*-

import spacy
import pandas as pd
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

data = pd.read_json('data/music/music_data.json')

data_sample = data.sample(10)

review_list = data_sample['review_text'].to_list()

review1 = review_list[0]

review1 = 'hello word'
doc = nlp(review1)

displacy.serve(doc, style="dep")

# print([x.text for x in doc])
#
# print([x.text for x in doc.sents])

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
# html = displacy.render(doc)
