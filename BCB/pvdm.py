import glob
import os
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

functions = pd.read_csv('functions.csv', names=('id', 'document'))

texts = [[word for word in document.split() ] for document in functions['document']]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# pvdm
sentences = [TaggedDocument(text, [i]) for i, text in enumerate(texts)]
model = Doc2Vec(sentences, dm=1, vector_size=300, window=5, min_count=1, workers=4, epochs=20, sample = 1e-3, seed=11)
model.save('model/pvdm.model')
dv = model.docvecs.vectors_docs
np.save('model/pvdm.dv', dv)
