import numpy as np
import re
import os
import itertools
import csv
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 1-(1)
def load_data_and_labels_another():
    # read csv file
    data = []
    f = open('data/newData.csv', 'r', newline='', encoding='utf-8')
    reader = csv.reader(f)
    for line in reader:
        data.append(line)
    data = data[1:]
    x_text = []
    y = []
    labels = {}
    authors = ['EAP', 'HPS', 'MWS']

    # made x
    for list in data:
        x_text.append(list[1])

        # made y
        if list[0] == 'EAP':
            y = y + [[1, 0, 0]]
        elif list[0] == 'HPL':
            y = y + [[0, 1, 0]]
        elif list[0] == 'MWS':
            y = y + [[0, 0, 1]]
    y = np.array(y)
    return [x_text, y]

# 1-(1)
def load_test_data():
    # read csv file
    data = []
    f = open('data/test.csv', 'r', newline='', encoding='utf-8')
    reader = csv.reader(f)
    for line in reader:
        line = line[1]
        data.append(line)
    return data

# 1-(4)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1     
    for epoch in range(num_epochs):

        if shuffle: # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            # np.arange(3) = array([0,1,2]) , np.random.permutation() : randomly shuffle

        else: 
            shuffled_data = data

        # make batches at each epoch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
