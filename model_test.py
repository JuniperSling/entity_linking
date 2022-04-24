import torch
from model import BertClassification
from utils import load_config
import argparse
import pickle
from dataloader import TextDataset, BatchTextCall, choose_bert_type

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
    label_dic = {True: 1, False: 0}
    right_predict_count = 0  # 预测正确的条目数
    all_count = 0  # 总条目数
    with open(path, 'rb') as f:
        while True:
            try:
                true_score = []  # 记录每个预测结果True标签的得分
                d = pickle.load(f)
                true_position = d['label_list'].index(True)
                texts1 = (d['entity'] + '|' + d['sentence'])
                texts2 = (d['predict_entity_label_list'][0] + '|' + d['predict_entity_description_list'][0])
                print("current line" + str(all_count))
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
                print("right predict: " + str(right_predict_count), end='')
                print("all predictions: " + str(all_count), end='')
                print("Accuracy: " + str(right_predict_count / all_count), end='')
            except EOFError:
                break
        print("right predict: " + str(right_predict_count))
        print("all predictions: " + str(all_count))
        print("Accuracy: " + str(right_predict_count/all_count))


if __name__ == '__main__':
    sentence1 = 'songwriter|Yalın (real name Hüseyin Yalın) is a Turkish pop singer and songwriter'
    sentence2 = 'Saverio Grandi|'

    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    model = BertClassification(bert_encode_model, hidden_size=config.hidden_size, num_classes=2, pooling_type=config.pooling_type)
    model.load_state_dict(torch.load(config.save_path + '/batch32lr5e-4hidden768bertbase/checkpoint_model_epoch_13.pt', map_location=torch.device('cpu')), False)

    evaluate('../data/49000-125854/test.txt')
