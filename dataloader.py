import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle

def choose_bert_type(path, bert_type="bert-base-uncased"):
    """
    choose bert type
    return: tokenizer, model
    """
    # load model from local
    if path != None:
        tokenizer = BertTokenizer.from_pretrained(path)
        model_config = BertConfig.from_pretrained(path)
        if bert_type == "tiny_bert":
            model = AlbertModel.from_pretrained(path, config=model_config)
        elif bert_type == "bert":
            model = BertModel.from_pretrained(path, config=model_config)
        else:
            model = None
            print("ERROR, not choose model!")

    # load model from huggingface
    else:
        try:
            print("loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(bert_type)
            print("loading model...")
            model = AutoModelForMaskedLM.from_pretrained(bert_type)
        except:
            tokenizer = None
            model = None
            print("ERROR, no model found on huggingface.co!")
    return tokenizer, model


def load_data(path, label_dic):
    texts1 = []
    texts2 = []
    labels = []
    with open(path, 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
                texts1.append(d['entity'] + '|' + d['sentence'])
                texts2.append(d['predict_entity_label_list'][0] + '|' + d['predict_entity_description_list'][0])
                labels.append(label_dic[d['label_list'][0]])
                # 成功预测两个数据,需要处理两个
                if len(d['predict_entity_label_list']) > 1:
                    texts1.append(d['entity'] + '|' + d['sentence'])
                    texts2.append(d['predict_entity_label_list'][1] + '|' + d['predict_entity_description_list'][1])
                    labels.append(label_dic[d['label_list'][1]])
            except EOFError:
                break
    print("data path: " + path + "   data count: " + str(len(texts1)) + "lines")
    return texts1, texts2, labels


class TextDataset(Dataset):
    def __init__(self, filepath, label_dict):
        super(TextDataset, self).__init__()
        # train 和 label 各自返回为list
        self.train1, self.train2, self.label = load_data(filepath, label_dict)

    def __len__(self):
        return len(self.train1)

    def __getitem__(self, item):
        text1 = self.train1[item]
        text2 = self.train2[item]
        label = self.label[item]
        return text1, text2, label


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, batch_text1, batch_text2):
        return self.tokenizer(batch_text1, batch_text2, max_length=self.max_len,
                              truncation='longest_first', padding='max_length', return_tensors='pt')

    def __call__(self, batch):
        batch_text1 = [item[0] for item in batch]
        batch_text2 = [item[1] for item in batch]
        batch_label = [item[2] for item in batch]

        source = self.text2id(batch_text1, batch_text2)
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)
        segment = source.get('token_type_ids').squeeze(1)
        label = torch.tensor(batch_label)
        return token, segment, mask, label
