import json
import re
import string
import pickle
from model import get_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import random
import numpy as np
import tables
from tqdm import tqdm
import pickle
from model import get_model
from keras import losses
import pickle
import pickle

model_path = 'save/'

def clean(s):
    del_str = string.punctuation + string.digits
    replace = ' ' * len(del_str)
    tran_tab = str.maketrans(del_str, replace)
    s = s.translate(tran_tab)
    arr = s.split()
    s = ' '.join([word.lower() for word in arr if len(word) > 1])
    return s

def get_code_vec(code_clean, longer_code):
    f = open(model_path + 'model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_code = dic['tokenizer_code']
    code_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_code.texts_to_sequences([code_clean]),
                                                                  padding='post', maxlen=longer_code)
    return code_test_vec

def get_desc_vec(review_clean, longer_desc):
    f = open(model_path + 'model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_desc = dic['tokenizer_desc']
    desc_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences([review_clean]),
                                                                  padding='post', maxlen=longer_desc)
    return desc_test_vec

def get_score(review, code):
    review_clean = clean(review)
    code_clean = clean(code)


    code_test_vec = []
    desc_test_vec = []
    bad_test_vec = []
    dumb_test_label = []

    f = open(model_path + 'model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_code = dic['tokenizer_code']
    tokenizer_desc = dic['tokenizer_desc']
    longer_code = dic['longer_code']
    longer_desc = dic['longer_desc']
    number_desc_tokens = dic['number_desc_tokens']
    number_code_tokens = dic['number_code_tokens']

    training_model, cos_model, model_code, model_query = get_model(longer_code, longer_desc, number_code_tokens, number_desc_tokens)
    training_model.load_weights(model_path + 'train.ckpt')
    cos_model.load_weights(model_path + 'cos.ckpt')
    model_code.load_weights(model_path + 'model_code.ckpt')
    model_query.load_weights(model_path + 'model_query.ckpt')

    code_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_code.texts_to_sequences([code_clean] * 2),
                                                                  padding='post', maxlen=longer_code)
    desc_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences([review_clean] * 2),
                                                                  padding='post', maxlen=longer_desc)
    # code_test_vec = get_code_vec(code_clean, longer_code)
    # desc_test_vec = get_desc_vec(review_clean, longer_desc)

    score = cos_model.predict([code_test_vec, desc_test_vec])
    return score

# demo
with open('data/k9mail/k9mail_test.json', 'r', encoding='UTF-8') as f:
    arr = json.load(f)
    for dic in arr:
        review = dic['review_raw']
        code = dic['method_content']
        score = get_score(review, code)
        print()
