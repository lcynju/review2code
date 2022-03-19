import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import random
import numpy as np
import tables
from tqdm import tqdm
import pickle
from model import get_model
from tensorflow.keras import losses
import pickle
import pickle

app_name = 'anki'

def read_file(path):
    file = open(path, 'r', encoding='UTF-8')
    return file.read()


# Both load_hdf5(), pad(), load_pickle() functions obtained from Deep Code search repository
def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size==-1:#if chunk_size is set to -1, then, load all data
        chunk_size=data_len
    start_offset = start_offset%data_len
    sents = []
    for offset in tqdm(range(start_offset, start_offset+chunk_size)):
        offset = offset%data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents


def pad(data, len=None):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def init_tokenizer(tokens):
    top_k = 5000
    tokenizer_code = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer_code.word_index['<pad>'] = 0
    tokenizer_code.index_word[0] = '<pad>'
    tokenizer_code.fit_on_texts(tokens)
    return tokenizer_code

file_format = 'txt'
data_path = "./my_data/"

def train():
    train_tokens_file_name = "train.tokens."+file_format
    train_desc_file_name = "train.desc."+file_format

    valid_tokens_file_name = 'valid.tokens.' + file_format
    valid_desc_file_name = 'valid.desc.' + file_format

    train_tokens = []
    train_desc = []
    train_bad_desc = []
    vocabulary = []
    train_num = 0
    if file_format=="txt":
        train_tokens = read_file(data_path+train_tokens_file_name).splitlines()
        train_desc = read_file(data_path+train_desc_file_name).splitlines()
        train_num = len(train_tokens)
        valid_tokens = read_file(data_path + valid_tokens_file_name).splitlines()
        valid_desc = read_file(data_path + valid_desc_file_name).splitlines()
        train_tokens.extend(valid_tokens)
        train_desc.extend(valid_desc)

        # Negative sampling
        train_bad_desc = read_file(data_path+train_desc_file_name).splitlines()
        valid_bad_desc = read_file(data_path + valid_desc_file_name).splitlines()
        train_bad_desc.extend(valid_bad_desc)
        random.shuffle(train_bad_desc)
    else:
        train_tokens = load_hdf5( "./data/train.tokens."+file_format, 0, 100000)
        train_desc = load_hdf5( "./data/train.desc."+file_format, 0, 100000)
        # Negative sampling
        train_bad_desc = load_hdf5( "./data/train.desc."+file_format, 0, 100000)
        random.shuffle(train_bad_desc)
        vocabulary = load_pickle("./data/vocab.tokens.pkl")

    # Tokenize process for the txt files
    code_vector = []
    desc_vector = []
    bad_desc_vector = []
    number_desc_tokens = 0
    number_code_tokens = 0
    tokenizer_code = ""
    tokenizer_desc = ""
    if file_format == "txt":

        tokenizer_code = init_tokenizer(train_tokens)
        code_vector = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_code.texts_to_sequences(train_tokens),
                                                                    padding='post')
        number_code_tokens = len(tokenizer_code.word_index) + 1
        print(number_code_tokens)

        tokenizer_desc = init_tokenizer(train_desc)

        desc_vector = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences(train_desc),
                                                                    padding='post')
        bad_desc_vector = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences(train_bad_desc),
                                                                        padding='post')

        number_desc_tokens = len(tokenizer_desc.word_index) + 1
        print(number_desc_tokens)

        longer_code = max(len(t) for t in code_vector)
        longer_desc = max(len(t) for t in desc_vector)

        print(longer_code, longer_desc)
        print(code_vector.shape, desc_vector.shape, bad_desc_vector.shape)

    else:
        # h5 files are already tokenize
        code_vector = train_tokens
        desc_vector = train_desc
        bad_desc_vector = train_bad_desc
        longer_code = max(len(t) for t in code_vector)
        longer_desc = max(len(t) for t in desc_vector)

        code_vector = pad(code_vector, longer_code)
        desc_vector = pad(desc_vector, longer_desc)
        bad_desc_vector = pad(bad_desc_vector, longer_desc)

        number_desc_tokens = len(vocabulary)
        number_code_tokens = len(vocabulary)
        print(longer_code, longer_desc, number_code_tokens, number_desc_tokens)

    dumb_label = np.zeros((code_vector.shape[0], 1))

    # 10000
    training_code = code_vector[:train_num,:]
    valid_code = code_vector[train_num:]

    training_desc = desc_vector[:train_num]
    valid_desc = desc_vector[train_num:]

    training_bad_desc = bad_desc_vector[:train_num]
    valid_bad_desc = bad_desc_vector[train_num:]

    train_dumb_label = dumb_label[:train_num]
    valid_dumb_label = dumb_label[train_num:]

    print(training_code.shape, training_desc.shape, training_bad_desc.shape)
    print(train_dumb_label.shape)

    training_model, cos_model, model_code, model_query = get_model(longer_code, longer_desc, number_code_tokens, number_desc_tokens)

    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_model.fit(x=[training_code, training_desc, training_bad_desc], y=train_dumb_label, epochs=5, verbose=1,
                        validation_data=([valid_code, valid_desc, valid_bad_desc], valid_dumb_label), callbacks=[earlystop_callback])
    training_model.save_weights('my_save/train.ckpt')
    cos_model.save_weights('my_save/cos.ckpt')
    model_code.save_weights('my_save/model_code.ckpt')
    model_query.save_weights('my_save/model_query.ckpt')
    dic = {
        'tokenizer_code': tokenizer_code,
        'tokenizer_desc': tokenizer_desc,
        'longer_code': longer_code,
        'longer_desc': longer_desc,
        'number_desc_tokens': number_desc_tokens,
        'number_code_tokens': number_code_tokens
    }
    f = open('my_save/model.pickle', 'wb')
    pickle.dump(dic, f)
    f.close()

def test():
    # ---test---
    code_test_vec = []
    desc_test_vec = []
    bad_test_vec = []
    dumb_test_label = []

    f = open('my_save/model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_code = dic['tokenizer_code']
    tokenizer_desc = dic['tokenizer_desc']
    longer_code = dic['longer_code']
    longer_desc = dic['longer_desc']
    number_desc_tokens = dic['number_desc_tokens']
    number_code_tokens = dic['number_code_tokens']

    training_model, cos_model, model_code, model_query = get_model(longer_code, longer_desc, number_code_tokens, number_desc_tokens)
    training_model.load_weights('my_save/train.ckpt')
    cos_model.load_weights('my_save/cos.ckpt')
    model_code.load_weights('my_save/model_code.ckpt')
    model_query.load_weights('my_save/model_query.ckpt')

    if file_format == "txt":

        test_tokens_file_name = "test.tokens.txt"
        test_desc_file_name = "test.desc.txt"

        test_tokens = read_file(data_path + test_tokens_file_name).splitlines()
        test_desc = read_file(data_path + test_desc_file_name).splitlines()
        test_bad_desc = read_file(data_path + test_desc_file_name).splitlines()
        random.shuffle(test_bad_desc)

        code_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_code.texts_to_sequences(test_tokens),
                                                                      padding='post', maxlen=longer_code)
        desc_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences(test_desc),
                                                                      padding='post', maxlen=longer_desc)
        bad_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences(test_bad_desc),
                                                                     padding='post')

        dumb_test_label = np.zeros((code_test_vec.shape[0], 1))

        print(code_test_vec.shape)
        print(desc_test_vec.shape)
    else:
        test_tokens = load_hdf5("./data/test.tokens." + file_format, 0, 100000)
        test_desc = load_hdf5("./data/test.desc." + file_format, 0, 100000)
        test_bad_desc = load_hdf5("./data/test.desc." + file_format, 0, 100000)
        random.shuffle(test_bad_desc)

        code_test_vec = pad(test_tokens, longer_code)
        desc_test_vec = pad(test_desc, longer_desc)
        bad_test_vec = pad(test_bad_desc, longer_desc)

        dumb_test_label = np.zeros((code_test_vec.shape[0], 1))

    # print( training_model.metrics_names)
    # training_model.evaluate(x=[code_test_vec, desc_test_vec, bad_test_vec], y=dumb_test_label)
    #
    # print( cos_model.metrics_names)
    # cos_model.evaluate(x=[code_test_vec, desc_test_vec], y=dumb_test_label)
    print(cos_model.predict( [code_test_vec[7:9,:]  , desc_test_vec[7:9,:]  ]  ))
    # print(cos_model.predict( [code_test_vec[8:9,:]  , desc_test_vec[90:91,:]  ]  ))

    test_code = code_test_vec[0:1,:]
    test_desc = desc_test_vec[0:1,:]
    # print(model_code.predict(test_code))
    # print(model_query.predict(test_desc))

# train()
test()