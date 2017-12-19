import numpy as np

import pickle, sys
import pandas as pd

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding, Dropout, Activation, Flatten, Reshape, Bidirectional, LSTM, Merge, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import optimizers
from sklearn import preprocessing, model_selection, metrics, pipeline


from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

np.random.seed(7)

EMBEDDING_DIM = 32
MAX_NB_WORDS = 200000
word_index = 10000 
maxlen = 80
act = 'sigmoid'

df = pd.read_csv('train.csv')
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)

nb_words = word_index

def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
    def add_n_skip_gram(q, n_gram_max):
            ngrams = []
            for n in range(3, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    temp_list = [q[w_index],q[w_index+2]]
                    ngrams.append('--'.join(temp_list))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    return docs


min_count = 2
#### for original
docs = create_docs(df)

print (len(docs))
tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(docs)



num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(docs)

word_index = tokenizer.word_index
docs = tokenizer.texts_to_sequences(docs)
docs = pad_sequences(sequences=docs, maxlen=maxlen, truncating = 'post')
input_dim = np.max(docs) + 1
print (input_dim)

print (len(docs))


def create_model(embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = create_model()

STAMP1 = 'fast_text' 
bst_model_path1 = STAMP1 + '.h5'
model_checkpoint1 = ModelCheckpoint(bst_model_path1, save_best_only=True, save_weights_only=True)
epochs = 25
x_train, x_test, y_train, y_test = train_test_split(docs,y, test_size=0.1)
hist = model.fit(x_train, y_train,
                 batch_size=256,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=0, monitor='val_loss'), model_checkpoint1])


def get_layer_outputs():
    test_image = x_train
    outputs    = [layer.output for layer in model.layers]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions
    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs_list

#get_layer_outputs()[2][0].shape)
softmax_val = get_layer_outputs()[2][0].tolist()
softmax_label = pd.DataFrame(softmax_val, columns = ['EAP', 'HPL','MWS'])
a2c = {0: 'EAP', 1 : 'HPL', 2 : 'MWS'}
y_str = []
for value in y_train:
    if value[0] == 1:
        y_str.append('EAP')
    if value[1] == 1:
        y_str.append('HPL')
    if value[2] == 1:
        y_str.append('MWS')

author_label = pd.DataFrame(y_str, columns = ['author'])
result = pd.concat([softmax_label, author_label],axis=1)

processed_name = 'processed.csv'
result.to_csv(processed_name, index=False)

