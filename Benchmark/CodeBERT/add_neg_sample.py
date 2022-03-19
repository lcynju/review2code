
import json
from tqdm import tqdm
import random

def add_neg_sample(data_list):

    id2method = dict()
    review2idlist= dict()

    for dic in tqdm(data_list):
        review_id = dic['review_id']
        review_raw = dic['review_raw']
        method_path = dic['method_path']
        method_name = dic['method_name']
        method_content = dic['method_content']
        method_id = method_path + '#' + method_name

        if review_id not in review2idlist:
            review2idlist[review_id] = {
                'review_raw':review_raw,
                'method_list':[method_id]
            }
        review2idlist[review_id]['method_list'].append(method_id)

        if method_id not in id2method:
            id2method[method_id]={
                'method_path':method_path,
                'method_name':method_name,
                'method_content':method_content
            }


    # 接下来是构造样本
    all_id_list = list(id2method.keys())
    new_data_list = []
    for review_id in tqdm(review2idlist):
        review_raw = review2idlist[review_id]['review_raw']
        method_list = review2idlist[review_id]['method_list']

        # 确保每个user review对应的方法唯一
        id_set = set(method_list)
        for method_id in method_list:
            # 首先构造正样本
            pos_data = {
                'review_id':review_id,
                'review_raw':review_raw,
                'method_path':id2method[method_id]['method_path'],
                'method_name':id2method[method_id]['method_name'],
                'method_content':id2method[method_id]['method_content'],
                'label':1
            }
            new_data_list.append(pos_data)
            # 构造负样本
            # 随机选择一个负样本
            choose_id = random.choice(all_id_list)
            while choose_id in id_set:
                choose_id = random.choice(all_id_list)
            id_set.add(choose_id)
            neg_data = {
                'review_id':review_id,
                'review_raw':review_raw,
                'method_path':id2method[choose_id]['method_path'],
                'method_name':id2method[choose_id]['method_name'],
                'method_content':id2method[choose_id]['method_content'],
                'label':0
            }
            new_data_list.append(neg_data)
    return new_data_list

train_neg_path = 'data/no_termux_train_neg.json'
valid_neg_path = 'data/no_termux_valid_neg.json'
#test_neg_path = 'data/no_termux_test_neg.json'

train_path = '../../data/all_data/no_termux_train.json'
valid_path = '../../data/all_data/no_termux_valid.json'
#test_path = '../../data/all_data/no_termux_test.json'


# train_neg_path = 'data/no_all_data_train_neg.json'
# valid_neg_path = 'data/no_all_data_valid_neg.json'
# 
# 
# train_path = '../../data/all_data/no_all_data_train.json'
# valid_path = '../../data/all_data/no_all_data_valid.json'

with open(train_path, 'r', encoding='utf-8') as f:
    train_list = json.load(f)
with open(valid_path, 'r', encoding='utf-8') as f:
    valid_list = json.load(f)
# with open(test_path, 'r', encoding='utf-8') as f:
#     test_list = json.load(f)
train_neg_list = add_neg_sample(train_list)
valid_neg_list = add_neg_sample(valid_list)
# test_neg_list = add_neg_sample(test_list)
random.shuffle(train_neg_list)
with open(train_neg_path, 'w', encoding='utf-8') as f:
    json.dump(train_neg_list, f, indent=2)
with open(valid_neg_path, 'w', encoding='utf-8') as f:
    json.dump(valid_neg_list, f, indent=2)
# with open(test_neg_path, 'w', encoding='utf-8') as f:
#     json.dump(test_neg_list, f, indent=2)