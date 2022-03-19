import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from utils import *
import numpy as np
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

device = 'cuda'
device_id = 4
torch.cuda.set_device(device_id)
label_list=[0,1]
max_seq_length = 512
output_mode = 'classification'
model_type = 'roberta'
batch_size = 512


tokenizer = RobertaTokenizer.from_pretrained("/root/gy/pretrain_model/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("./models/k9mail/checkpoint-best/")
model = model.to(device)
model.eval()
# 为了测试时间
start_time = time.time()
code_file = open("data/k9mail_code.json", "rb")
review_file = open("data/k9mail_list_test.json", "rb")
codes = json.load(code_file)
reviews = json.load(review_file)


code_vec_list = []
code_information_list=[]
method2path = dict()
unique_path_set = set()
unique_method_set = set()
for method_dic in tqdm(codes):
    code_path = method_dic['method_path']
    code_name = method_dic['method_name']
    code_content = method_dic['method_content'].replace('###',' ')
    code_id = code_path+'#'+code_name
    code_information_list.append((code_id, code_path, code_name, code_content))
    method2path[code_id] = code_path
    unique_path_set.add(code_path)
    unique_method_set.add(code_path+'#'+code_name)

# 用来存储分数
scores = []
query_vec_list=[]
query_method_list=[]

for review_dic in tqdm(reviews):
    review_content = review_dic['review_raw']
    real_methods_list = review_dic['method_list']
    query_method_list.append(real_methods_list)
    cur_score_list = []
    examples = []
    for code_information in code_information_list:
        code_content = code_information[3]
        text_a = review_content
        text_b = code_content
        examples.append(InputExample(guid="text-1", text_a=text_a, text_b=text_b, label=0))
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 1,
                                            pad_on_left=bool(model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    eval_dataset = TensorDataset(all_input_ids, all_input_mask)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}

            ouputs = model(**inputs)
            logits = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # score = F.softmax(logits, dim=-1)[:,1].cpu().detach().numpy()
            score = logits[:,1].cpu().detach().numpy()
            cur_score_list.extend(score)
    scores.append(cur_score_list)

sort_id_list = np.argsort(np.array(scores), axis=-1, kind='quicksort', order=None)[:, ::-1]
end_time = time.time()
print(end_time-start_time)

file_set=set()
for key in method2path.keys():
    file_set.add(method2path[key])
print('代码的数量：'+str(len(method2path.keys())))
print('文件的个数：'+str(len(file_set)))
print('文件的个数：'+str(len(unique_path_set)))
print('方法的个数：'+str(len(unique_method_set)))


top1_hitting = 0
top3_hitting = 0
top5_hitting = 0
top10_hitting = 0
top20_hitting = 0
review_num = len(reviews)
ranks = []
result_list = []
for review_dic, real_methods_list, sort_id in zip(reviews, query_method_list, sort_id_list):
    # 处理真实的方法名称
    new_real_method_set = set()
    # 处理真实的文件名称
    new_real_file_set = set()
    for method in real_methods_list:
        method_name = method.split('#')[1]+'#'+method.split('#')[2]
        new_real_method_set.add(method_name)
        new_real_file_set.add(method.split('#')[1])
    new_real_method_list = list(new_real_method_set)
    new_real_file_list =list(new_real_file_set)


    # 计算MRR
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


    # 计算topk hitting
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
        if len(pre_methods_list) == 100:
            break


    result_dict = review_dic
    result_dict['method_list'] = new_real_method_list
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

    pre_method_count = 0
    for pre_method, code_content in zip(pre_methods_list, code_content_list):
        pre_method_count +=1
        cur_pre_result = dict()
        cur_pre_result['pre_method_count'] = pre_method_count
        cur_pre_result['predict_method'] = pre_method
        cur_pre_result['predict_content'] = code_content
        pre_file = pre_method.split('#')[0]
        if pre_file in new_real_file_list:
            isFilePredict = True
        else:
            isFilePredict = False

        if pre_method in new_real_method_list:
            isMethodPredict=True
        else:
            isMethodPredict=False
        cur_pre_result['isFilePredict'] = isFilePredict
        cur_pre_result['isMethodPredict'] = isMethodPredict
        result_dict['predict_code'].append(cur_pre_result)
        if isMethodPredict and isFilePredict:
            break
    result_list.append(result_dict)
# 保存预测的结果
with open('k9mail_result.json','w',encoding='utf-8') as f:
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
for real_methods_list, sort_id in zip(query_method_list, sort_id_list):
    real_code_path_list = []
    for method_id in real_methods_list:
        real_code_path_list.append(method2path[method_id])

    # 计算MRR
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



