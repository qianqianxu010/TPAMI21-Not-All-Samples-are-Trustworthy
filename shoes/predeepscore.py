import re
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import numpy as np
import pdb


def parse_text(line):
    patternLabel = 'Label:\s+([+,-]?\d+)'
    patternScore = 'Score:\s+([+,-]?\d+\.\d+)'
    label = int(re.findall(patternLabel, line)[0])
    score = float(re.findall(patternScore, line)[0])
    return label, score


def read_shoes_data_line(line):
    datasplit = line.split(',')
    label = int(datasplit[0])
    strength = int(datasplit[5])
    return label, strength


def cnt_pair_num(file):
    with open(file, 'r') as f:
        res = np.array([read_shoes_data_line(line) for line in f.readlines()])
        dis_pair_num = res.shape[0]
        all_pairs = np.sum(res[:, 1])
        num_attr = len(np.unique(res[:, 0]))
        pairs_per_attr = [np.sum(res[:, 0] == i) for i in range(num_attr)]
    return dis_pair_num, all_pairs, pairs_per_attr


def parse_text_LFW(line):
    patternLabel = 'Label:\s+([-,+]?\d+)'
    patternScore = 'Score:\s+([-,+]?\d+\.\d+)'
    patternAttr = 'Attribute:\s+(\d)'
    # pdb.set_trace()

    attr = int(re.findall(patternAttr, line)[0])
    label = int(re.findall(patternLabel, line)[0])
    score = float(re.findall(patternScore, line)[0])
    return attr, label, score


def read_deep_results_Age(file):
    with open(file, 'r') as f:
        res = np.array([parse_text(line) for line in f.readlines()])
    label, score = res[:, 0], res[:, 1]
    label = 2.0 * (label > 0) - 1
    label_pred = 2.0 * (score > 0) - 1
    metrics = [accuracy_score, f1_score, precision_score, recall_score]
    acc, f, pre, re = [func(label, label_pred) for func in metrics]
    auc = roc_auc_score(label, score)
    return np.array((acc, f, pre, re, auc))


def read_deep_results_LFW(file, attr_num):
    with open(file, 'r') as f:
        res = np.array([parse_text_LFW(line) for line in f.readlines()])
    attr, label, score = res[:, 0], res[:, 1], res[:, 2]
    label = 2.0 * (label > 0) - 1
    label_pred = 2.0 * (score > 0) - 1
    # pdb.set_trace()
    acc = [accuracy_score(label[attr == i], label_pred[attr == i])
           for i in range(attr_num)]
    return np.array(acc)


if __name__ == '__main__':
    dataset = 'age'  # 'age', 'lfw', 'shoes'
    if dataset == 'age':
        res_dir = '../age/age_logit_lambda_0.2'
        res = read_deep_results_Age(
            os.path.join(res_dir, 'with_gamma_logit'))
    elif dataset == 'lfw':
        res_dir = '../lfw/lfw_logit_lambda_0.2'
        res = read_deep_results_LFW(
            os.path.join(res_dir, 'with_gamma_logit'), 10)
    elif dataset == 'shoes':
        res_dir = './shoes_logit_lambda_0.2'
        res = read_deep_results_LFW(
            os.path.join(res_dir, 'with_gamma_logit'), 7)
    else:
        print('Dataset not exist!')
        exit()
    np.savetxt(os.path.join(res_dir, 'res_with_gamma_logit.txt'),
               res, fmt='%.4f', delimiter='&')
