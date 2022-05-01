import torch
from model import BertClassification
from utils import load_config
import argparse
import pickle
from dataloader import TextDataset, BatchTextCall, choose_bert_type
from tqdm import tqdm
import requests
import random

def get_wiki_predictions(query, limit):
    predict_entity_id_list = [''] * limit
    predict_entity_label_list = [''] * limit
    predict_entity_description_list = [''] * limit
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php?action=wbsearchentities&search="
            + query + "&language=en&limit=" + str(limit) + "&format=json").json()
        predict_entity_id_list = [ent['id'] for ent in response['search']]
        predict_entity_label_list = [ent['label'] for ent in response['search']]
        predict_entity_description_list = [ent['description'] for ent in response['search']]
        return predict_entity_id_list, predict_entity_label_list, predict_entity_description_list
    except:
        return predict_entity_id_list, predict_entity_label_list, predict_entity_description_list

def single_test(sentence1, sentence2):
    source = tokenizer(sentence1, sentence2, max_length=config.sent_max_len, truncation='longest_first', padding='max_length', return_tensors='pt')

    token = source.get('input_ids').squeeze(1)
    # 注意力编码，全1则为自注意力
    mask = source.get('attention_mask').squeeze(1)
    # segment 是区分上下句的编码，上句全0，下句全1，用在Bert的句子预测任务上
    segment = source.get('token_type_ids').squeeze(1)

    model.eval()
    with torch.no_grad():
        outputs = model(token, segment, mask)
        return outputs


def evaluate(path):
    end_line = 10000  # 需要测试的行数
    pbar = tqdm(total=end_line)

    label_dic = {True: 1, False: 0}
    right_predict_count = 0  # 预测正确的条目数
    all_count = 0  # 总条目数
    with open(path, 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
                true_score = []  # 记录True得分
                true_position = d['label_list'].index(True)
                texts1 = (d['entity'] + '|' + d['sentence'])
                texts2 = (d['predict_entity_label_list'][0] + '|' + d['predict_entity_description_list'][0])
                true_score.append(single_test(texts1, texts2)[0][1].item())
                # 成功预测多个数据,需要处理多个候选
                if len(d['predict_entity_label_list']) > 1:
                    for i in range(1, len(d['predict_entity_label_list'])):
                        texts1 = (d['entity'] + '|' + d['sentence'])
                        texts2 = (d['predict_entity_label_list'][i] + '|' + d['predict_entity_description_list'][i])
                        true_score.append(single_test(texts1, texts2)[0][1].item())
                if true_score.index(max(true_score)) == true_position:
                    right_predict_count += 1
                all_count += 1
                pbar.update(1)
                print("Accuracy: " + str(right_predict_count / all_count))
                if all_count == end_line:
                    break
            except EOFError:
                break
        print("right predict: " + str(right_predict_count))
        print("all predictions: " + str(all_count))
        print("Accuracy: " + str(right_predict_count/all_count))


def evaluate_with_candidate(path, limits):
    jump_count = 0

    end_line = 10000  # 需要测试的行数

    jump_prop = 1  # 跳过90%

    pbar = tqdm(total=end_line)
    label_dic = {True: 1, False: 0}

    right_generate_count = 0  # 候选成功覆盖到的条目数
    right_predict_count = 0  # 预测正确的条目数
    all_count = 0  # 总条目数
    true_all_count = 0  # 参与运算的总条目

    with open(path, 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
                jump_count += 1
                if jump_count < 0: continue

                flag = random.random()

                # 随机跳过
                if flag > jump_prop:
                    all_count += 1
                    pbar.update(1)
                    if all_count >= end_line: break
                    continue

                target_entity_id = d['predict_entity_list'][d['label_list'].index(True)]  # target 预测ID
                predict_entity_id_list, predict_entity_label_list, predict_entity_description_list = get_wiki_predictions(d['entity'], limits)
                if predict_entity_id_list.count(target_entity_id) != 0:
                    right_generate_count += 1
                    # 对每一个候选实体进行打分
                    true_score = []  # 记录每一个candidate True得分
                    true_position = predict_entity_id_list.index(target_entity_id)  # 记录target下标位置

                    texts1 = (d['entity'] + '|' + d['sentence'])
                    for i in range(0, len(predict_entity_label_list)):
                        texts2 = (predict_entity_label_list[i] + '|' + predict_entity_description_list[i])
                        true_score.append(single_test(texts1, texts2)[0][1].item())
                    if true_score.index(max(true_score)) == true_position:
                        right_predict_count += 1
                all_count += 1
                true_all_count += 1
                print(str(limits) + "train-Generate Acc: " + str(right_generate_count / true_all_count), end='')
                if right_generate_count != 0: print("Predict Acc/Generate: " + str(right_predict_count / right_generate_count), end='')
                print("Predict Acc/All: " + str(right_predict_count / true_all_count))
                pbar.update(1)
                if all_count >= end_line:
                    print("Generate Accuracy: " + str(right_generate_count / true_all_count), end='')
                    if right_generate_count != 0: print("Predict Acc/Generate: " + str(right_predict_count / (right_generate_count)),
                          end='')
                    print("Predict Acc/All: " + str(right_predict_count / true_all_count))
                    break
            except EOFError:
                break


if __name__ == '__main__':
    sentence1 = 'HIT|I got my PhD in Physics in HIT, China.'
    sentence2 = 'Saverio Grandi|'

    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    model = BertClassification(bert_encode_model, hidden_size=config.hidden_size, num_classes=2, pooling_type=config.pooling_type)
    model.load_state_dict(torch.load(config.save_path + '/batch64lr5e-4hidden768bertbasecased/checkpoint_model_epoch_28.pt', map_location=torch.device('cpu')), False)

    # evaluate('../data/49000-125854/test.txt')
    # single_test(sentence1, sentence2)
    evaluate_with_candidate('../data/49000-125854/train.txt', 50)
