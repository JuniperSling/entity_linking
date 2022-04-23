import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import TextDataset, BatchTextCall, choose_bert_type
from model import BertClassification
from utils import load_config


def evaluation(model, test_dataloader, loss_func, label2ind_dict, save_path, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (token, segment, mask, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        out = model(token, segment, mask)
        loss = loss_func(out, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        # dim = 1 表示输出所在行的最大值，max返回(值，下标)
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if valid_or_test == "test":
        report = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(test_dataloader), report, confusion
    return acc, total_loss / len(test_dataloader)


def train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True
    label_dict = {True: 1, False: 0}
    label2ind_dict = {'正样本': 1, '负样本': 0}
    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    # 其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    # 加载分词器和预训练模型
    tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)
    print('loading train data...')
    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"), label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  collate_fn=train_dataset_call)
    print('loading valid data...')
    valid_dataset = TextDataset(os.path.join(config.data_dir, "dev.txt"), label_dict)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  collate_fn=train_dataset_call)
    print('loading test data...')
    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"), label_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                 collate_fn=train_dataset_call)

    multi_classification_model = BertClassification(bert_encode_model, hidden_size=config.hidden_size,
                                                    num_classes=2, pooling_type=config.pooling_type)
    if torch.cuda.is_available():
        multi_classification_model.cuda()
        print("Running on CUDA")
    else:
        print("Running on CPU")
    # multi_classification_model.load_state_dict(torch.load(config.save_path))

    optimizer = torch.optim.AdamW(multi_classification_model.parameters(),
                                  lr=config.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01, amsgrad=False)
    # betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    # beta1：一阶矩估计的指数衰减率（如 0.9）
    # beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
    # eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
    loss_func = F.cross_entropy

    loss_total, top_acc = [], 0
    early_stop_count = 0  # 连续10轮没有上升就停止
    acc, loss, report, confusion = evaluation(multi_classification_model,
                                              test_dataloader, loss_func, label2ind_dict,
                                              config.save_path)
    print("at first:")
    print("Accuracy: %.4f   Loss in test %.4f" % (acc, loss))
    print(report, confusion)
    train_loss = []
    test_acc = []
    test_loss = []

    for epoch in range(config.epoch):
        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))
        # enumerate(iterable) 返回一个tuple(index, it)
        for i, (token, segment, mask, label) in enumerate(tqdm_bar):
            if torch.cuda.is_available():
                token = token.cuda()
                segment = segment.cuda()
                mask = mask.cuda()
                label = label.cuda()

            # 在每一个batch时并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
            multi_classification_model.zero_grad()

            # 自动调用forward()函数
            out = multi_classification_model(token, segment, mask)

            # 计算损失
            loss = loss_func(out, label)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        train_loss.append(np.mean(loss_total))
        acc, loss, report, confusion = evaluation(multi_classification_model,
                                                  test_dataloader, loss_func, label2ind_dict,
                                                  config.save_path)
        print("Accuracy: %.4f   Loss in test %.4f" % (acc, loss))
        test_acc.append(acc)
        test_loss.append(loss)
        print("train loss list", end='')
        print(train_loss)
        print("test acc list", end='')
        print(test_acc)
        print("test loss list", end='')
        print(test_loss)
        if top_acc < acc:
            early_stop_count = 0
            top_acc = acc
            torch.save(multi_classification_model.state_dict(), 'checkpoint_model_epoch_{}.pt'.format(epoch))
            print(report, confusion)
        else:
            print(report, confusion)
            early_stop_count += 1
            if early_stop_count > config.early_stop_count:
                print('early stop!')
                break
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    # print(type(config.lr), type(config.batch_size))
    train(config)
