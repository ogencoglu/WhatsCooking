'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@topdatascience.com, oguzhan.gencoglu@tut.fi
Description    : 17th Place out of 1388 teams (top 2%) Submission for Kaggle What's Cooking Competition
'''

from __future__ import absolute_import
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from nltk.stem.wordnet import WordNetLemmatizer
import re
import itertools
import os.path
import json
from datetime import datetime

def k_to_one_hot(k_hot_vector):
    # This function converts k-hot target vector to one-hot target matrix
    
    classes = np.unique(k_hot_vector)
    one_hot_matrix = []
    
    for i in np.arange(len(classes)):
        row = (k_hot_vector == classes[i]).astype(int, copy = False)
        if len(one_hot_matrix) == 0:
            one_hot_matrix = row
        else:
            one_hot_matrix = np.vstack((one_hot_matrix, row))
            
    return classes, one_hot_matrix.conj().transpose()
    
    
def read_data(filename):
    # read data into lists
    
    with open(filename) as data_file:    
        data = json.load(data_file)
        
    ids, cuisines, ingredients = [], [], []
    if 'cuisine' in data[0].keys():
        for i in range(len(data)):
            ids.append(data[i]['id'])
            cuisines.append(data[i]['cuisine'])
            ingredients.append(data[i]['ingredients'])
    else:
        for i in range(len(data)):
            ids.append(data[i]['id'])
            ingredients.append(data[i]['ingredients'])    
                
    return ids, cuisines, ingredients
    
    
def create_submission(test_ids, guess):
    # create submission in proper format
    
    sub = np.transpose(np.vstack((test_ids, guess)))
    sub = np.vstack((['id', 'cuisine'], sub))
    sub_file_name = 'submission_' + str(datetime.now())[0:16] +'.csv'
    sub_file_name = sub_file_name.replace(' ', '_')
    sub_file_name = sub_file_name.replace(':', '-')
    np.savetxt(sub_file_name, sub, delimiter=",", fmt="%s")
    
    return None  
    

def remove_numbers(ing):
    # remove numbers from ingredients
    
    return [[re.sub("\d+", "", x) for x in y] for y in ing]

    
def remove_special_chars(ing):
    # remove certain special characters from ingredients
   
    ing = [[x.replace("-", " ") for x in y] for y in ing] 
    ing = [[x.replace("&", " ") for x in y] for y in ing] 
    ing = [[x.replace("'", " ") for x in y] for y in ing] 
    ing = [[x.replace("''", " ") for x in y] for y in ing] 
    ing = [[x.replace("%", " ") for x in y] for y in ing] 
    ing = [[x.replace("!", " ") for x in y] for y in ing] 
    ing = [[x.replace("(", " ") for x in y] for y in ing] 
    ing = [[x.replace(")", " ") for x in y] for y in ing] 
    ing = [[x.replace("/", " ") for x in y] for y in ing] 
    ing = [[x.replace("/", " ") for x in y] for y in ing] 
    ing = [[x.replace(",", " ") for x in y] for y in ing] 
    ing = [[x.replace(".", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u2122", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u00AE", " ") for x in y] for y in ing] 
    ing = [[x.replace(u"\u2019", " ") for x in y] for y in ing] 

    return ing
    
    
def make_lowercase(ing):
    # make all letters lowercase for all ingredients
    
    return [[x.lower() for x in y] for y in ing]
    
    
def remove_extra_whitespace(ing):
    # removes extra whitespaces
    
    return [[re.sub( '\s+', ' ', x).strip() for x in y] for y in ing] 
    
    
def stem_words(ing):
    # word stemming for ingredients
    
    lmtzr = WordNetLemmatizer()
    
    def word_by_word(strng):
        
        return " ".join(["".join(lmtzr.lemmatize(w)) for w in strng.split()])
    
    return [[word_by_word(x) for x in y] for y in ing] 
    
    
def remove_units(ing):
    # remove certain words from ingredients
    
    remove_list = ['g', 'lb', 's', 'n']
        
    def check_word(strng):
        
        s = strng.split()
        resw  = [word for word in s if word.lower() not in remove_list]
        
        return ' '.join(resw)

    return [[check_word(x) for x in y] for y in ing] 
    

def extract_feats(ingredients, uniques):
    # each ingredient + each word as feature
    
    feats_whole = np.zeros((len(ingredients), len(uniques)))
    for i in range(len(ingredients)):
        for j in ingredients[i]:
            feats_whole[i, uniques.index(j)] = 1
            
    new_uniques = []
    for m in uniques:
        new_uniques.append(m.split())
    new_uniques = list(set(list(itertools.chain.from_iterable(new_uniques))))
    
    feats_each = np.zeros((len(ingredients), len(new_uniques))).astype(np.uint8)
    for i in range(len(ingredients)):
        for j in ingredients[i]:
            for k in j.split():
                feats_each[i, new_uniques.index(k)] = 1
            
    return np.hstack((feats_whole, feats_each)).astype(bool)
    
    
def load_model():
    # load neural net model architectiure
    
    mdl = Sequential()
    mdl.add(Dense(512, init='glorot_uniform', activation='relu', 
                                        input_shape=(train_feats.shape[1],)))
    mdl.add(Dropout(0.5))
    mdl.add(Dense(128, init='glorot_uniform', activation='relu'))
    mdl.add(Dropout(0.5))
    mdl.add(Dense(20, activation='softmax'))
    mdl.compile(loss='categorical_crossentropy', optimizer='adadelta')
    
    return mdl    

    
if __name__ == '__main__':
    
    # preprocess training set
    print("\nPreprocessing...\n")  
    train_ids, train_cuisines, train_ingredients = read_data('train.json')
    train_ingredients = make_lowercase(train_ingredients)
    train_ingredients = remove_numbers(train_ingredients)
    train_ingredients = remove_special_chars(train_ingredients)
    train_ingredients = remove_extra_whitespace(train_ingredients)
    train_ingredients = remove_units(train_ingredients)
    train_ingredients = stem_words(train_ingredients)
    
    # preprocess test set
    test_ids, test_cuisines, test_ingredients = read_data('test.json')
    test_ingredients = make_lowercase(test_ingredients)
    test_ingredients = remove_numbers(test_ingredients)
    test_ingredients = remove_special_chars(test_ingredients)
    test_ingredients = remove_extra_whitespace(test_ingredients)
    test_ingredients = remove_units(test_ingredients)
    test_ingredients = stem_words(test_ingredients)
    
    # encode   
    print("Encoding...\n")  
    le = LabelEncoder()
    targets = le.fit_transform(train_cuisines)
    classes, targets = k_to_one_hot(targets)
    
    # extract features
    print("Feature extraction...\n") 
    uniques = list(set([item for sublist in train_ingredients + test_ingredients for item in sublist]))
    train_feats = extract_feats(train_ingredients, uniques)
    test_feats = extract_feats(test_ingredients, uniques)
  
    # train
    n_ensemble = 10
    for ens in range(n_ensemble):
        print("\n\tTraining...", ens)
        model = load_model()
        
        # if model already exists, continue training
        model_name = 'model' + str(ens) + '.hdf5'
        if os.path.isfile(model_name):
            model.load_weights(model_name)
            
        model.fit(train_feats, targets, nb_epoch=2500, batch_size=4096, 
                                                        show_accuracy = True)
        model.save_weights(model_name, overwrite=True)

    # create submission out of the ensemble
    preds = []
    for ens in range(n_ensemble):
        print("\nSubmission", ens)
        model = load_model()

        model_name = 'model' + str(ens) + '.hdf5'
        model.load_weights(model_name)            
        preds.append(model.predict_proba(test_feats))

    # final cuisine decision: argmax of sum of log probabilities  
    print("\nPredicting...")      
    preds = sum(np.log(preds))
    guess = le.inverse_transform(np.argmax(preds, axis=1))
    create_submission(test_ids, guess) 