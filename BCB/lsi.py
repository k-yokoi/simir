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

# lsi model
lsi_model = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=200)
lsi_model.save('model/lsi.model')
corpus_lsi = lsi_model[corpus]
dv = gensim.matutils.corpus2dense(corpus_lsi, num_terms=lsi_model.num_topics).T
np.save('model/lsi.dv', dv)
