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
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

app_name = 'all'

model_path = 'my_save/{}/epoch_20/'.format(app_name)
f = open(model_path + 'model.pickle', 'rb')
dic = pickle.load(f)
tokenizer_code = dic['tokenizer_code']
tokenizer_desc = dic['tokenizer_desc']
longer_code = dic['longer_code']
longer_desc = dic['longer_desc']
number_desc_tokens = dic['number_desc_tokens']
number_code_tokens = dic['number_code_tokens']
training_model, cos_model, model_code, model_query = get_model(longer_code, longer_desc, number_code_tokens,
                                                               number_desc_tokens)


def clean(s):
    del_str = string.punctuation + string.digits
    replace = ' ' * len(del_str)
    tran_tab = str.maketrans(del_str, replace)
    s = s.translate(tran_tab)
    arr = s.split()
    s = ' '.join([word.lower() for word in arr if len(word) > 1])
    return s


def get_score(desc_test_vec, code_test_vec):
    score = cos_model.predict([code_test_vec, desc_test_vec])
    score = score.reshape((len(score)))
    return score

def get_review_vec(review):
    review_clean_list = []
    for re in review:
        review_clean = clean(re)
        review_clean_list.append(review_clean)
    f = open(model_path + 'model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_desc = dic['tokenizer_desc']
    longer_desc = dic['longer_desc']
    desc_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_desc.texts_to_sequences(review_clean_list),
                                                                  padding='post', maxlen=longer_desc)
    return desc_test_vec

def get_code_vec(code):
    code_clean_list = []
    for co in code:
        code_clean = clean(co)
        code_clean_list.append(code_clean)
    f = open(model_path + 'model.pickle', 'rb')
    dic = pickle.load(f)
    tokenizer_code = dic['tokenizer_code']
    longer_code = dic['longer_code']

    code_test_vec = tf.keras.preprocessing.sequence.pad_sequences(tokenizer_code.texts_to_sequences(code_clean_list),
                                                                  padding='post', maxlen=longer_code)
    return code_test_vec


name = 'cgeo'
start_time = time.time()
code_file = open("my_data/{}/{}_code.json".format(name, name), "rb")
review_file = open("my_data/{}/{}_list_test.json".format(name, name), "rb")
codes = json.load(code_file)
reviews = json.load(review_file)



code_content_list =[]
code_information_list=[]
method2path = dict()
for method_dic in tqdm(codes):
    code_commit_id = method_dic['commit_id']
    code_path = method_dic['method_path']
    code_name = method_dic['method_name']
    code_content = method_dic['method_content']
    code_id = code_commit_id+'#'+code_path+'#'+code_name
    code_information_list.append((code_id, code_path, code_name, code_content))
    code_content_list.append(code_content)
    method2path[code_id] = code_path


review_content_list = []
review_method_list=[]
for review_dic in tqdm(reviews):
    review_id = review_dic['review_id']
    review_content = review_dic['review_raw']
    real_methods_list = review_dic['method_list']
    review_method_list.append(real_methods_list)
    review_content_list.append(review_content)


code_vec_list = get_code_vec(code_content_list)
review_vec_list = get_review_vec(review_content_list)

scores = []
for i in tqdm(range(len(reviews))):
    query_vec = list(review_vec_list[i])
    query_vec_list = [query_vec] * len(code_vec_list)
    cur_score = get_score(np.array(query_vec_list), code_vec_list)
    scores.append(cur_score)

sort_id_list = np.argsort(np.array(scores), axis=-1, kind='quicksort', order=None)[:, ::-1]
end_time = time.time()
print(end_time-start_time)

file_set=set()
for key in method2path.keys():
    file_set.add(method2path[key])
print('code count'+str(len(method2path.keys())))
print('file count'+str(len(file_set)))

top1_hitting = 0
top3_hitting = 0
top5_hitting = 0
top10_hitting = 0
top20_hitting = 0
review_num = len(reviews)
ranks = []
result_list = []
for review_dic, real_methods_list, sort_id in zip(reviews, review_method_list, sort_id_list):

    new_real_method_list=[]
    for method in real_methods_list:
        method_name = method.split('#')[1]+'#'+method.split('#')[2]
        new_real_method_list.append(method_name)

    rank = 0
    find = False
    for idx in sort_id[:1000]:
        method_path = code_information_list[idx][1]
        method_name = code_information_list[idx][2]
        code_content = code_information_list[idx][3]
        if find is False:
            rank +=1
        if method_path + '#' +method_name in new_real_method_list:
            find = True
            ranks.append(1/rank)
            break
    if not find:
        ranks.append(0)

    pre_methods_list = []
    code_content_list = []
    visited_method_set = set()
    for i in sort_id:
        method_path = code_information_list[i][1]
        method_name = code_information_list[i][2]
        code_content = code_information_list[i][3]
        if method_path + '#' +method_name in visited_method_set:
            continue
        pre_methods_list.append(method_path+'#'+method_name)
        code_content_list.append(code_content)
        visited_method_set.add(method_path+'#'+method_name)
        if len(pre_methods_list) == 20:
            break


    result_dict = review_dic
    result_dict['predict_code'] = []

    if len(set(new_real_method_list) & set(pre_methods_list[:1])) != 0:
        top1_hitting += 1
    if len(set(new_real_method_list) & set(pre_methods_list[:3])) != 0:
        top3_hitting += 1
    if len(set(new_real_method_list) & set(pre_methods_list[:5])) != 0:
        top5_hitting += 1
    if len(set(new_real_method_list) & set(pre_methods_list[:10])) != 0:
        top10_hitting += 1
    if len(set(new_real_method_list) & set(pre_methods_list[:20])) != 0:
        top20_hitting += 1

    for pre_method, code_content in zip(pre_methods_list[:5], code_content_list[:5]):
        cur_pre_result = dict()
        cur_pre_result['predict_method'] = pre_method
        cur_pre_result['predict_content'] = code_content
        if pre_method in new_real_method_list:
            predict=True
        else:
            predict=False
        cur_pre_result['predict'] = predict
        result_dict['predict_code'].append(cur_pre_result)
    result_list.append(result_dict)

with open('result.json','w',encoding='utf-8') as f:
    json.dump(result_list, f, indent=2)
print('MRR:', str(np.mean(ranks)))
print('top1_hitting:', str(top1_hitting/review_num))
print('top3_hitting:', str(top3_hitting/review_num))
print('top5_hitting:', str(top5_hitting/review_num))
print('top10_hitting:', str(top10_hitting/review_num))
print('top20_hitting:', str(top20_hitting/review_num))



top1_hitting = 0
top3_hitting = 0
top5_hitting = 0
top10_hitting = 0
top20_hitting = 0
ranks = []
for real_methods_list, sort_id in zip(review_method_list, sort_id_list):
    real_code_path_list = []
    for method_id in real_methods_list:
        real_code_path_list.append(method2path[method_id])

    rank = 0
    find = False
    for idx in sort_id[:1000]:
        method_path = code_information_list[idx][1]
        method_name = code_information_list[idx][2]
        code_content = code_information_list[idx][3]
        if find is False:
            rank +=1
        if method_path in real_code_path_list:
            find = True
            ranks.append(1/rank)
            break
    if not find:
        ranks.append(0)


    predict_code_path_list = []
    for id in sort_id:
        pre_method_id = code_information_list[id][0]
        if method2path[pre_method_id] not in predict_code_path_list:
            predict_code_path_list.append(method2path[pre_method_id])
        if len(predict_code_path_list) == 20:
            break

    if len(set(real_code_path_list) & set(predict_code_path_list[:1])) !=0:
        top1_hitting +=1
    if len(set(real_code_path_list) & set(predict_code_path_list[:3])) != 0:
        top3_hitting += 1
    if len(set(real_code_path_list) & set(predict_code_path_list[:5])) != 0:
        top5_hitting += 1
    if len(set(real_code_path_list) & set(predict_code_path_list[:10])) != 0:
        top10_hitting += 1
    if len(set(real_code_path_list) & set(predict_code_path_list[:20])) != 0:
        top20_hitting += 1

print('MRR:', str(np.mean(ranks)))
print('top1_hitting:', str(top1_hitting/review_num))
print('top3_hitting:', str(top3_hitting/review_num))
print('top5_hitting:', str(top5_hitting/review_num))
print('top10_hitting:', str(top10_hitting/review_num))
print('top20_hitting:', str(top20_hitting/review_num))
