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

# WV-avg
num_features = 300
sentences = [doc.split() for doc in functions['document']]
model = Word2Vec(sentences, workers=4, hs = 0, sg = 1, negative = 10, iter = 25,size=num_features, min_count = 1, window = 10, sample = 1e-3, seed=1)
model.save("model/word2vec.model")
model = Word2Vec.load("model/word2vec.model")
wv = model.wv
size = model.vector_size
dv = []
for text in texts:
    vec = np.zeros( num_features, dtype="float32" )
    for word in text:
        vec += wv[word]
    norm = np.sqrt(np.einsum('...i,...i', vec, vec))
    if(norm!=0):
        vec /= norm
    dv.append(vec)
np.save('model/avgvec.dv', np.array(dv))
