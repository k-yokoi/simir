from __future__ import print_function

#################################
#       Database Setting        #
#################################
host = 'localhost'
port = '5432'
dbname = 'bigclonebench'
user = 'postgres'
password = '****'
#################################


## select model
import sys
import numpy as np

args = sys.argv
if args[1] == 'lsi':
        dv = np.load('model/lsi.dv.npy')
elif args[1] == 'lda':
        dv = np.load('model/lda.dv.npy')
elif args[1] == 'pvdbow':
        docmodel = Doc2Vec.load("model/pvdbow.model")
        dv = docmodel.docvecs.vectors_docs
elif args[1] == 'pvdm':
        docmodel = Doc2Vec.load("model/pvdm.model")
        dv = docmodel.docvecs.vectors_docs
elif args[1] == 'avgvec':
        dv = np.load('model/AvgVec.dv.npy')
else:
        print('set argumment [lsi, lda, pvdbow, pvdm, avgvec]')
        sys.exit(1)

bin_vec_dim = dv.shape[1]





import os.path
import psycopg2
from gensim.models import Doc2Vec
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import re
import linecache

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
import pandas as pd
import glob
import os
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time
from sklearn.model_selection import train_test_split
import random
from gensim.models.doc2vec import Doc2Vec

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils import np_utils, Sequence
from keras.utils.vis_utils import plot_model
import keras as K
import pandas as pd
import os
import time

import matplotlib
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight, shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.initializers import *
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

def kMLPC(input_dim, epochs=200, batch_size=200):
    model = Sequential()
    model.add(Dense(100, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer=glorot_uniform(), input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size)


def classification(x1, x2):
    input = Input(shape=(bin_vec_dim,))
    # share layers
    #feed_forward_model = Model(inputs=input, outputs=feed_forward(input))
    #x1 = feed_forward_model(x1)
    #x2 = feed_forward_model(x2)
    concat_input = Input(shape=(bin_vec_dim*2,))
    # share layers
    merge_model = Model(inputs=concat_input,
                        outputs=Activation(activation='relu')(
                            BatchNormalization()(
                                Dense(100, kernel_initializer=kernel_init,
                                      bias_initializer=bias_init,
                                      input_shape=(bin_vec_dim*2,))(
                                    concat_input))))
    
    xc1 = K.layers.concatenate([x1, x2])
    xc1 = merge_model(xc1)
    
    xc2 = K.layers.concatenate([x2, x1])
    xc2 = merge_model(xc2)
    
    xc = K.layers.average([xc1, xc2])
    
    x = Dense(1, use_bias=False, activation='sigmoid',
              kernel_initializer=kernel_init,
              batch_input_shape=K.backend.get_variable_shape(xc))(xc)
    
    return x




functions = pd.read_csv('functions.csv', names=('id', 'document'))

connection = psycopg2.connect("host=" + host + " port=" + port + " dbname=" + dbname + " user=" + user + " password=" + password)
connection.get_backend_pid()
cur=connection.cursor()
sql = 'select function_id_one, function_id_two from clones where min_size >= 5'
cur.execute(sql)

clones = [row for row in cur]

cur.close()
connection.close()



function_id_set = set(functions['id'])
clones = [row for row in clones if row[0] in function_id_set and row[1] in function_id_set]

connection = psycopg2.connect("host=" + host + " port=" + port + " dbname=" + dbname + " user=" + user + " password=" + password)
connection.get_backend_pid()
cur=connection.cursor()
sql = "SELECT FP.function_id_one, FP.function_id_two from false_positives as FP, functions as A, functions as B "
sql += "where FP.function_id_one=A.id and FP.function_id_two=B.id and "
sql += "A.normalized_size>=5 and B.normalized_size>=5"
cur.execute(sql)

FP = [row for row in cur]

cur.close()
connection.close()


function_id_set = set(functions['id'])
FP = [row for row in FP if row[0] in function_id_set and row[1] in function_id_set]


df_clones= pd.DataFrame(clones, columns=['id_one', 'id_two'])
df_clones['clone'] = 1
df_fp= pd.DataFrame(FP, columns=['id_one', 'id_two'])
df_fp['clone'] = 0
df_clones = df_clones.append(df_fp)


functions_id = functions['id']
dv_dict = {}
for i, id in enumerate(functions_id):
    dv_dict[id] = dv[i]

X_left = np.array([dv_dict[id] for id in df_clones['id_one']])
X_right = np.array([dv_dict[id] for id in df_clones['id_two']])
t = np.array(df_clones['clone'])

X = X_left - X_right
X = np.abs(X)

X, y = shuffle(X, t)

skf = StratifiedKFold(n_splits=10)
avg_accuracy = 0.
avg_recall = 0.
avg_precision = 0.
avg_f1_score = 0.
skf_split = skf.split(X, y)

for train_idx, test_idx in skf_split:
        train_X = X[train_idx]
        train_Y = y[train_idx]
        
        train_X, train_Y = shuffle(train_X, train_Y)
        
        test_X = X[test_idx]
        test_Y = y[test_idx]
        
        validate_X = test_X[:256]
        validate_Y = test_Y[:256]

        model = Sequential()
        model.add(Dense(100, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer=glorot_uniform(), input_dim=bin_vec_dim))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #samples_generator = SequenceSamples(train_X_left,train_X_right,
        #                                    train_Y, batch_size)
        epochs = 1
        batch_size=256
        model.fit(train_X, train_Y,
                  epochs=epochs,verbose=1, batch_size=batch_size)
        y_pred = model.predict(test_X, verbose=1, batch_size=batch_size)
        y_pred = np.round(y_pred)
        accuracy = accuracy_score(test_Y, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(test_Y,
                                                                 y_pred, average='binary')
        print("accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                   accuracy, recall, precision, fscore))
        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1_score += fscore
        
avg_accuracy /= 10.0
avg_precision /= 10.0
avg_recall /= 10.0
avg_f1_score /= 10.0

print('Avg accuracy: %.4f, avg recall: %.4f, avg precision: %.4f, avg f1 '
          'score: %.4f' % (
              avg_accuracy, avg_recall, avg_precision, avg_f1_score))

