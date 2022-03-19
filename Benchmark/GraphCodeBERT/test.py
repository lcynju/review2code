

import json
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from model import Model, ClassifierModel
from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

device='cuda:2'

tokenizer = RobertaTokenizer.from_pretrained("./pretrained_model/graphcodebert-base")
model = RobertaModel.from_pretrained("./pretrained_model/graphcodebert-base/")
model = Model(model)
model.load_state_dict(torch.load("./saved_models/checkpoint-best-mrr/anki_model.bin"), strict=False)
# tokenizer = RobertaTokenizer.from_pretrained("./pretrained_model/graphcodebert-base")
# model = RobertaModel.from_pretrained("./pretrained_model/python_model")
# model = Model(model)
model = model.to(device)
model.eval()
# 为了测试时间
start_time = time.time()
code_file = open("data/anki_code.json", "rb")
review_file = open("data/anki_list_test.json", "rb")
codes = json.load(code_file)
reviews = json.load(review_file)
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg



def get_nl(nl, nl_length):
    nl_tokens = tokenizer.tokenize(nl)[:nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    return torch.tensor([nl_ids])

def get_code(code, code_length, data_flow_length):
    # code
    parser = parsers['java']
    # extract data flow
    code_tokens, dfg = extract_dataflow(code, parser, 'java')

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    code_tokens = code_tokens[:code_length + data_flow_length - 2 - min(len(dfg), data_flow_length)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:code_length + data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = code_length + data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    attn_mask = np.zeros((code_length + data_flow_length,
                          code_length + data_flow_length), dtype=bool)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(code_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (torch.tensor([code_ids]),
            torch.tensor([attn_mask]),
            torch.tensor([position_idx]))


query_vec_list=[]
query_method_list=[]
for review_dic in tqdm(reviews):
    review_content = review_dic['review_raw']
    real_methods_list = review_dic['method_list']
    nl_inputs = get_nl(review_content, 128).to(device)
    query_vec = model(nl_inputs=nl_inputs)
    #query_vec = model(tokenizer(review_content, return_tensors='pt', max_length=512, truncation=True).to(device)['input_ids'])[1]
    query_vec_list.append(query_vec.cpu().detach().numpy())
    query_method_list.append(real_methods_list)


code_vec_list = []
code_information_list=[]
method2path = dict()

for method_dic in tqdm(codes):

    code_path = method_dic['method_path']
    code_name = method_dic['method_name']
    code_content = method_dic['method_content']
    code_id = code_path+'#'+code_name
    code_inputs, attn_mask, position_idx = get_code(code_content, 256,64)
    code_inputs, attn_mask, position_idx=code_inputs.to(device), attn_mask.to(device), position_idx.to(device)
    code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
    #code_vec = model(tokenizer(code_content, return_tensors='pt', max_length=512, truncation=True).to(device)['input_ids'])[1]
    code_vec_list.append(code_vec.cpu().detach().numpy())
    code_information_list.append((code_id, code_path, code_name, code_content))
    method2path[code_id] = code_path

query_vec_list = np.concatenate(query_vec_list, 0)
code_vec_list = np.concatenate(code_vec_list, 0)
scores = np.matmul(query_vec_list, code_vec_list.T)
sort_id_list = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
end_time = time.time()
print(end_time-start_time)

file_set=set()
for key in method2path.keys():
    file_set.add(method2path[key])

top1_hitting = 0
top3_hitting = 0
top5_hitting = 0
top10_hitting = 0
top20_hitting = 0
review_num = len(query_method_list)
ranks = []
result_list = []
for review_dic, real_methods_list, sort_id in zip(reviews, query_method_list, sort_id_list):
    # 处理真实的方法名称
    new_real_method_set=set()
    # 处理真实的文件名称
    new_real_file_set = set()
    for method in real_methods_list:
        method_name = method.split('#')[1]+'#'+method.split('#')[2]
        new_real_method_set.add(method_name)
        new_real_file_set.add(method.split('#')[1])
    new_real_method_list = list(new_real_method_set)
    new_real_file_list = list(new_real_file_set)


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
with open('anki_result.json', 'w', encoding='utf-8') as f:
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

