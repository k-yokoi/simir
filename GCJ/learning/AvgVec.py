from __future__ import print_function

import numpy as np
#from matplotlib import pyplot as plt
#import seaborn as sns
#from sklearn import datasets
import pandas as pd
import glob
import os
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_moons, make_circles, make_classification
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time
#from sklearn.model_selection import train_test_split
import random
import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils import np_utils, Sequence
from keras.utils.vis_utils import plot_model
import keras as K
import numpy as np
import pandas as pd
#import os
import time

#import matplotlib
#from matplotlib.ticker import NullFormatter
#from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight, shuffle
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


bin_vec_dim = None
#embedding_dim = 6
#dim = 128
#keep_prob = 0.6

batch_size = 256
#test_size = 256


# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logdir = '/tmp/logs'

kernel_init = K.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                             distribution='uniform')
bias_init = K.initializers.Constant(value=0.01)

def listup_files(path):
    return [os.path.abspath(p) for p in glob.glob(path, recursive=True)]

def load_documents():
    files = listup_files('gcj/*/*.java')
    documents = []
    tags = []
    for file in files:
        with open(file) as f:
            d = f.read()
        if len(d.split()) > 0:
            documents.append(d)
            tags.append(os.path.basename(os.path.dirname(file)))

    return documents, tags 

def train_docvecs(documents):
    t_beg = time.clock()
    sentences = [doc.split() for doc in documents]
    num_features = 300
    model = Word2Vec(sentences, workers=4, hs = 0, sg = 1, negative = 5, iter = 20,size=num_features, min_count = 1, window = 5, sample = 1e-3, seed=1)
    model.save("model/word2vec.model")

    texts = [[word for word in document.split() ] for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # frequency変数で1より上の単語のみを配列に構築
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

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

    t_end = time.clock()
    
    print('*' * 80)
    print('Vectorization :')
    print('Time cost: %.2f' % (t_end - t_beg))
    
    
    fout.write('*' * 80 + '\n')
    fout.write('Vectorization :\n')
    fout.write('Time cost: %.2f\n' % (t_end - t_beg))
    fout.flush()
    dv = np.array(dv)
    return dv

def load_dataset(dv, tags):
    global bin_vec_dim
    bin_vec_dim = dv.shape[1]
    
    parameter = dv.shape[0]
    sample = parameter * (parameter-1) // 2 + parameter
    X_left = np.empty((sample, dv.shape[1]))
    X_right = np.empty((sample, dv.shape[1]))
    t = np.empty(sample)
    count = 0
    for i in range(parameter):
        for j in range(i, parameter):
            X_left[count] = dv[i]
            X_right[count] = dv[j]
            t[count] = (1 if tags[i] == tags[j] else 0)
            count += 1
    
    return X_left, X_right, t

def classification(x1, x2):
    input = Input(shape=(bin_vec_dim,))
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
    
def train_10_fold_balanced(dv, tags):
    Xl, Xr, y = load_dataset(dv, tags)
    Xl, Xr, y = shuffle(Xl, Xr, y, random_state=0)
    skf = StratifiedKFold(n_splits=10)
    avg_accuracy = 0.
    avg_recall = 0.
    avg_precision = 0.
    avg_f1_score = 0.
    fold_index = 0
    
    
    
    for train_idx, test_idx in skf.split(Xl, y):
        t_beg = time.clock()
        print ('*' * 40 + str(fold_index) + '*' * 40)
        fold_path = os.path.join(vector_name, str(fold_index))
        if os.path.exists(vector_name) is not True:
            os.mkdir(vector_name)
        if os.path.exists(fold_path) is not True:
            os.mkdir(fold_path)
        train_X_left = Xl[train_idx]
        train_X_right = Xr[train_idx]
        train_Y = y[train_idx]
        
        train_X_left, train_X_right, train_Y = shuffle(train_X_left,
                                                    train_X_right, train_Y, random_state=0)
        
        test_X_left = Xl[test_idx]
        test_X_right = Xr[test_idx]
        test_Y = y[test_idx]
        
        validate_X_left = test_X_left[:256]
        validate_X_right = test_X_right[:256]
        validate_Y = test_Y[:256]

        X_left = Input(shape=(bin_vec_dim, ))
        X_right = Input(shape=(bin_vec_dim, ))

        predictions = classification(X_left, X_right)

        model = Model(inputs=[X_left, X_right], outputs=predictions)

        model.compile(optimizer=K.optimizers.adam(lr=0.001),
                    loss=K.losses.binary_crossentropy,
                    metrics=['accuracy'])
        model.fit([train_X_left, train_X_right], train_Y,
                epochs=4,verbose=1, batch_size=batch_size)

        t_end = time.clock()
        print('Time cost: %.2f' % (t_end - t_beg))
        
        model.save(filepath=os.path.join(fold_path, 'model.ckpt'))
        
        y_pred = model.predict([test_X_left, test_X_right], verbose=1, batch_size=batch_size)
        y_pred = np.round(y_pred)
        accuracy = accuracy_score(test_Y, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(test_Y,
                                                                y_pred, average='binary')
        print("accuracy: %.4f, recall: %.4f, "
                "precision: %.4f, f1 score: %.4f\n" % (
                accuracy, recall, precision, fscore))
        fout.write('*' * 80 + '\n')
        fout.write('Fold %d:\n' % (fold_index))
        fout.write('Time cost: %.2f\n' % (t_end - t_beg))
        fout.write("Fold index: %d, accuracy: %.4f, recall: %.4f, "
                "precision: %.4f, f1 score: %.4f\n" % (
                fold_index, accuracy, recall, precision, fscore))
        fout.flush()

        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1_score += fscore

        print('*' * 80)
        fold_index += 1

    avg_accuracy /= 10.0
    avg_precision /= 10.0
    avg_recall /= 10.0
    avg_f1_score /= 10.0

    print('Avg accuracy: %.4f, avg recall: %.4f, avg precision: %.4f, avg f1 '
            'score: %.4f' % (
                avg_accuracy, avg_recall, avg_precision, avg_f1_score))
    fout.write('*' * 80 + '\n')
    fout.write(
        'Avg accuracy: %.4f, avg recall: %.4f, avg precision: %.4f, avg f1 '
        'score: %.4f\n' % (avg_accuracy, avg_recall, avg_precision, avg_f1_score))

def predict_on_full_dataset(documents, tags):
    texts = [[word for word in document.split() ] for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # frequency変数で1より上の単語のみを配列に構築
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    model = models.Word2Vec.load("model/word2vec.model")

    texts = [[word for word in document.split() ] for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # frequency変数で1より上の単語のみを配列に構築
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    wv = model.wv
    num_features = wv.vector_size
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
    
    dv = np.array(dv)
    
    Xl, Xr, y = load_dataset(dv, tags)
    Xl, Xr, y = shuffle(Xl, Xr, y, random_state=0)

    model = K.models.load_model(vector_name + '/4/model.ckpt')
    y_pred = model.predict([Xl, Xr], verbose=1, batch_size=batch_size)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y,
                                                                y_pred, average='binary')
    print("prediction, accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                   accuracy, recall, precision, fscore))

    return 0

    
if __name__ == '__main__':
    # model_summary()
    if os.path.exists('result') is not True:
        os.mkdir("result")
    vector_name = 'avgvec'    
    fout = open('result/' + vector_name + '.txt', 'w')
    documents, tags = load_documents()
    dv = train_docvecs(documents)
    
    beg = time.time()
    train_10_fold_balanced(dv, tags)
    st = time.time()
    print("Total time: ", st-beg)

    st = time.time()
    predict_on_full_dataset(documents, tags)
    print("Predict time on the full dataset: ", time.time() - st)
    fout.write("Predict time on the full dataset: " + str(time.time() - st) + '\n')
    fout.close()