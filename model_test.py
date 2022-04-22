import torch
from model import BertClassification
from utils import load_config
import argparse
from dataloader import TextDataset, BatchTextCall, choose_bert_type
from transformers import BertConfig, BertModel
from torch.utils.data import Dataset, DataLoader
from dataloader import TextDataset, BatchTextCall, choose_bert_type
import os

parser = argparse.ArgumentParser(description='bert classification')
parser.add_argument("-c", "--config", type=str, default="./config.yaml")
args = parser.parse_args()
config = load_config(args.config)
tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)

model = BertClassification(bert_encode_model, hidden_size=config.hidden_size, num_classes=10, pooling_type=config.pooling_type)
model.load_state_dict(torch.load(config.save_path + '/checkpoint_model_epoch_7.pt', map_location=torch.device('cpu')), False)

label2ind_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
class_dic = ["股票", "房产", "财经", "教育", "科技", "一眼烂", "时政", "体育", "游戏", "时尚"]
sentence = '今天世界航天日，致敬中国航天女将'
source = tokenizer(sentence, max_length=config.sent_max_len, truncation=True, padding='max_length', return_tensors='pt')
token = source.get('input_ids').squeeze(1)
# 注意力编码，全1则为自注意力
mask = source.get('attention_mask').squeeze(1)
# segment 是区分上下句的编码，上句全0，下句全1，用在Bert的句子预测任务上
segment = source.get('token_type_ids').squeeze(1)

model.eval()
with torch.no_grad():
    outputs = model(token, segment, mask)
    print("一眼丁真，鉴定  \"" + sentence + "\"  为：" + class_dic[torch.max(outputs, 1)[1]])

