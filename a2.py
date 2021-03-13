import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
import nltk 
from nltk.stem import WordNetLemmatizer 
nltk.download('averaged_perceptron_tagger') 
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

lemmatizer = WordNetLemmatizer()

def pos_tagger(tag): 
    if tag.startswith('J'): 
        return wordnet.ADJ 
    elif tag.startswith('V'): 
        return wordnet.VERB 
    elif tag.startswith('N'): 
        return wordnet.NOUN 
    elif tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None

def make_tuples(list):
    tuples = []
    for row in list:
        tuple = (row[2],row[3])
        tuples.append(tuple)
    return tuples
    
def preprocess(inputfile):
    
    inputdata = inputfile.readlines()
    processed_data = []
    for row in inputdata[1:]:
        split_data = row.replace('\n', '')
        listed_data = split_data.split('\t')
        processed_data.append(listed_data)
        
    lowercase_lists = []
   
    for row in processed_data:
        list_item = []
        for idx, item in enumerate(row):
            if idx != 2:
                list_item.append(item)
            else:
                list_item.append(item.lower())
        lowercase_lists.append(list_item)
    
    tuples = make_tuples(lowercase_lists)

    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tuples)) 
    
    lemmatized_words = [] 
    for word, tag in wordnet_tagged: 
        if tag is None: 
            # if there is no available tag, append the token as is 
            lemmatized_words.append((word, '<NT>')) 
        else:         
            # else use the tag to lemmatize the token 
            lemmatized_words.append((lemmatizer.lemmatize(word, tag), tag))

    for idx, word_tag_tuple in enumerate(lemmatized_words):
        lowercase_lists[idx][2] = word_tag_tuple[0]
        lowercase_lists[idx][3] = word_tag_tuple[1]

    return lowercase_lists[:]

# Code for part 2
class Instance:
    def __init__(self, neclass, features, pos=None):
        self.neclass = neclass
        self.features = features
        self.pos = pos

    def __str__(self):
        return "Class: {} Features: {} POS: {}".format(self.neclass, self.features, self.pos)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []
    for idx, row in enumerate(data):
        if row[4].startswith('B'): 
            neclass = row[4][2:]
            features = []
            sentence_number = row[1]

            # Add preceeding features
            for i in range(5, 0, -1):
                if data[idx - i][1] == sentence_number:
                    features.append(data[idx - i][2])

            # Add start padding
            if len(features) < 5:
                start_pads = [f"<S{x}>" for x in range(5, len(features), -1)]
                for pad in start_pads:
                    features.insert(0, pad)

            # Find end of NE
            counter = 1
            while data[idx + counter][4].startswith('I'):
                 counter += 1
            end = idx + counter - 1

            # Add subsequent features
            for i in range(1, 6):
                try:
                    if data[end + i][1] == sentence_number:
                        features.append(data[end + i][2])
                except IndexError:
                    continue

            # Add end padding
            if len(features) < 10:
                end_pads = [f"</S{x - 5}>" for x in range(10, len(features), -1)]
                for pad in end_pads:
                    features.append(pad)

            instances.append(Instance(neclass, features, row[3]))
    return instances

# Code for part 3
def create_table(instances):
    feature_counts = []
    total_word_counts = {}

    for instance in instances:
        class_name = instance.neclass
        features = instance.features

        feature_word_count = {}

        for word in features:
            if word in feature_word_count:
                feature_word_count[word] += 1
            else:
                feature_word_count[word] = 1

        for k, v in feature_word_count.items():
            if k in total_word_counts:
                total_word_counts[k] += v
            else:
                total_word_counts[k] = v

        class_and_features = {'class_name': class_name}
        class_and_features.update(feature_word_count)

        feature_counts.append(class_and_features)

    top_words = sorted(total_word_counts, key=total_word_counts.get, reverse=True)[:3000]
    df = pd.DataFrame(feature_counts).fillna(0)[['class_name'] + top_words]
    
    return df


def ttsplit(bigdf):
    df_train = bigdf.sample(frac=0.8,random_state=200)
    df_test = bigdf.drop(df_train.index).sample(frac=1.0)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
        
    return (
        df_train.drop('class_name', axis=1).to_numpy(),
        df_train['class_name'],
        df_test.drop('class_name', axis=1).to_numpy(),
        df_test['class_name']
    )


# Code for part 5
def confusion_matrix(truth, predictions):
    fig, ax = plt.subplots()
    
    labels = truth.unique()
    labels.sort()
    
    matrix = conf_matrix(truth, predictions, labels=labels)
    
    cm_plot = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    cm_plot.plot(ax=ax)

    

# Code for bonus part B
def create_instances_bonus(data):
    pos_map = {
        'a': '_ADJ_',
        'n': '_NOUN_',
        'r': '_ADV_',
        'v': '_VERB_',
        '<NT>': '<NT>'
    }
    
    instances = []
    for idx, row in enumerate(data):
        if row[4].startswith('B'): 
            neclass = row[4][2:]
            features = []
            pos = []
            sentence_number = row[1]

            # Add preceeding features
            for i in range(5, 0, -1):
                if data[idx - i][1] == sentence_number:
                    features.append(data[idx - i][2])
                    pos.append(pos_map[data[idx - i][3]])

            # Add start padding
            if len(features) < 5:
                start_pads = [f"<S{x}>" for x in range(5, len(features), -1)]
                for pad in start_pads:
                    features.insert(0, pad)
                    pos.insert(0, '<NT>')

            # Find end of NE
            counter = 1
            while data[idx + counter][4].startswith('I'):
                 counter += 1
            end = idx + counter - 1

            # Add subsequent features
            for i in range(1, 6):
                try:
                    if data[end + i][1] == sentence_number:
                        features.append(data[end + i][2])
                        pos.append(pos_map[data[end + i][3]])
                except IndexError:
                    continue

            # Add end padding
            if len(features) < 10:
                end_pads = [f"</S{x - 5}>" for x in range(10, len(features), -1)]
                for pad in end_pads:
                    features.append(pad)
                    pos.append('<NT>')

            instances.append(Instance(neclass, features, pos))
    return instances


def create_table_bonus(instances):
    feature_counts = []
    part_of_speech_counts = []
    total_word_counts = {}

    for instance in instances:
        class_name = instance.neclass
        features = instance.features
        pos = instance.pos

        feature_word_count = {}
        pos_count = {}

        for word in features:
            if word in feature_word_count:
                feature_word_count[word] += 1
            else:
                feature_word_count[word] = 1

        for k, v in feature_word_count.items():
            if k in total_word_counts:
                total_word_counts[k] += v
            else:
                total_word_counts[k] = v

        for p in pos:
            if p != '<NT>':
                if p in pos_count:
                    pos_count[p] += 1
                else:
                    pos_count[p] = 1


        class_and_features = {'class_name': class_name}
        class_and_features.update(feature_word_count)

        feature_counts.append(class_and_features)
        part_of_speech_counts.append(pos_count)

    top_words = sorted(total_word_counts, key=total_word_counts.get, reverse=True)[:3000]
    
    df = pd.DataFrame(feature_counts).fillna(0)[['class_name'] + top_words]
    pos_df = pd.DataFrame(part_of_speech_counts).fillna(0)
    
    full_df = pd.concat([df, pos_df], axis=1)
    
    return full_df


def bonusb(file):
    input_file = open(file, 'r')
    input_data = preprocess(input_file)
    
    instances = create_instances_bonus(input_data)
    pos_df = create_table_bonus(instances)

    train_X, train_y, test_X, test_y = ttsplit(pos_df)
    
    model = LinearSVC()
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    confusion_matrix(test_y, test_predictions)
    confusion_matrix(train_y, train_predictions)
