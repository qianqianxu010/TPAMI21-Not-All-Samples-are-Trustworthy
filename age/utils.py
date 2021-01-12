import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
from abc import abstractmethod


'''
 data format:
 0001.jpg, 0002.jpg, 1,  0
 0034.jpg, 0023.jpg, -1, 1
 0022.jpg, 0002.jpg, 0,  1
 0001.jpg, 0045.jpg, 1,  2
 if binary == True, label1==-1 will be set 0.
'''


class PairwiseDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, binary=False,
                 load_img=True, load_rgb=False, attr_num=10, root_dir='./data'):
        self.transform = transform
        self.target_transform = target_transform
        self.binary = binary
        self.root_dir = root_dir
        self.load_img = load_img
        self.load_rgb = load_rgb
        self.attr_num = attr_num
        self.lens = 0
        self.lines = list()

    @abstractmethod
    def _parse_each_item(self, line):
        pass

    @abstractmethod
    def _get_img_name(self, img_id):
        pass

    def get_outlier_num(self):
        d_pairs = [dict()] * self.attr_num
        total_pairs = 0
        total_outliers = 0
        for i in range(self.lens):
            img_id0, img_id1, label, attr_id, pair_id, strength = self._parse_each_item(
                self.lines[i])
            if img_id0 < img_id1:
                id1, id2 = img_id0, img_id1
            else:
                id1, id2 = img_id1, img_id0
                label = -label
            if (id1, id2) not in d_pairs[int(attr_id)]:
                d_pairs[int(attr_id)][(id1, id2)] = (label, strength)
            else:
                total_pairs += 1
                other_label, other_strength = d_pairs[int(attr_id)][(id1, id2)]
                if strength > other_strength:
                    total_outliers += other_strength
                # elif strength < other_strength:
                else:
                    total_outliers += strength
        # print(total_pairs)
        return total_outliers

    def __getitem__(self, index):
        img_id0, img_id1, label, attr_id, pair_id, strength = self._parse_each_item(
            self.lines[index])
        label, attr_id, pair_id, strength = np.float32(label), np.float32(
            attr_id), np.float32(pair_id), np.float32(strength)
        if self.binary:  # default is False
            if int(label) == -1:
                label = np.float32(0.)

        if self.load_img:
            img0 = Image.open(self._get_img_name(img_id0))
            img1 = Image.open(self._get_img_name(img_id1))
            if self.load_rgb:
                img0 = img0.convert('RGB')
                img1 = img1.convert('RGB')

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            img_id0, img_id1 = np.float32(img_id0), np.float32(img_id1)
            return img_id0, img_id1, img0, img1, label, attr_id, pair_id, strength
        else:
            img_id0, img_id1 = np.float32(img_id0), np.float32(img_id1)
            return img_id0, img_id1, label, attr_id, pair_id, strength

    def __len__(self):
        return self.lens


class PairwiseImgDataSet(Dataset):

    def __init__(self, txt, load_rgb=False, transform=None, target_transform=None, root_dir='./data'):
        self.load_rgb = load_rgb
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir
        self.lines = list()
        self.lens = 0

    @abstractmethod
    def _parse_each_item(self, line):
        pass

    @abstractmethod
    def _get_img_name(self, img_id):
        pass

    def _get_all_img_id(self):
        img_ids = set()
        for i in range(len(self.lines)):
            img_id1, img_id2, attr_id = self._parse_each_item(self.lines[i])
            img_ids.add((img_id1, attr_id))
            img_ids.add((img_id2, attr_id))
        self.img_ids = list(img_ids)

    def __getitem__(self, index):
        img_id, attr_id = self.img_ids[index]
        img = Image.open(self._get_img_name(img_id))
        if self.load_rgb:
            img = img.convert('RGB')
        img_id, attr_id = np.float32(img_id), np.float32(attr_id)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_id, attr_id

    def __len__(self):
        return self.lens


class ShoesDataSet(PairwiseDataset):

    def __init__(self, txt, file_dict, transform=None, target_transform=None,
                 binary=False, load_img=True, load_rgb=False, attr_num=7, root_dir='./data'):
        super(ShoesDataSet, self).__init__(txt, transform,
                                           target_transform, binary, load_img, load_rgb, attr_num, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.file_dict = file_dict

    def _parse_each_item(self, line):
        split = line.split(',')
        img_id0, img_id1 = int(split[1]), int(split[2])
        label = int(split[4])  # comp_value
        pair_id = int(split[3])  # edge(x, y)_id
        attr_id = int(split[0])  # attr_id
        strength = int(split[5])  # strength
        return img_id0, img_id1, label, attr_id, pair_id, strength

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.file_dict[int(img_id)][0])


class ShoesImgDataSet(PairwiseImgDataSet):

    def __init__(self, txt, file_dict, load_rgb=False, transform=None,
                 target_transform=None, root_dir='./data'):
        super(ShoesImgDataSet, self).__init__(
            txt, load_rgb, transform, target_transform, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        # pdb.set_trace()
        self.file_dict = file_dict
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        attr_id, img_id1, img_id2 = int(split[0]), int(split[1]), int(split[2])
        return img_id1, img_id2, attr_id

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.file_dict[img_id][0])


class AgeDataSet(PairwiseDataset):
    def __init__(self, txt, file2id_dict, id2file_dict, transform=None,
                 target_transform=None, binary=False, load_img=True, load_rgb=True, attr_num=1, root_dir='./data/'):
        super(AgeDataSet, self).__init__(txt, transform,
                                         target_transform, binary, load_img, load_rgb, attr_num, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.file2id_dict = file2id_dict
        self.id2file_dict = id2file_dict

    def _parse_each_item(self, line):
        split = line.split(',')
        img_id0, img_id1 = self.file2id_dict[split[0]
                                             ], self.file2id_dict[split[1]]
        label = int(split[2])  # comp_value
        pair_id = int(split[3])  # edge(x, y)_id
        strength = int(split[4])  # strength
        attr_id = 0  # attr_id
        return img_id0, img_id1, label, attr_id, pair_id, strength

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.id2file_dict[img_id])


class AgeImgDataSet(PairwiseImgDataSet):

    def __init__(self, txt, file2id_dict, id2file_dict, load_rgb=True,
                 transform=None, target_transform=None, root_dir='./data'):
        super(AgeImgDataSet, self).__init__(
            txt, load_rgb, transform, target_transform, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.file2id_dict = file2id_dict
        self.id2file_dict = id2file_dict
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        attr_id = 0
        img_id1, img_id2 = self.file2id_dict[split[0]
                                             ], self.file2id_dict[split[1]]
        return img_id1, img_id2, attr_id

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.id2file_dict[img_id])


class LFWDataSet(PairwiseDataset):

    def __init__(self, txt, transform=None, target_transform=None, binary=False,
                 load_img=True, load_rgb=False, attr_num=10, root_dir='./data/images/'):
        super(LFWDataSet, self).__init__(txt, transform, target_transform,
                                         binary, load_img, load_rgb, attr_num, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)

    def _parse_each_item(self, line):
        split = line.split(',')
        img_id0, img_id1 = int(split[0].split(
            '.')[0]), int(split[1].split('.')[0])
        label = int(split[2])  # comp_value
        attr_id = int(split[3])  # attr_id
        pair_id = int(split[4])  # edge(x, y)_id
        strength = int(split[5])  # strength
        return img_id0, img_id1, label, attr_id, pair_id, strength

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, str(img_id) + '.jpg')


class LFWImgDataSet(PairwiseImgDataSet):

    def __init__(self, txt, load_rgb=False, transform=None, target_transform=None, root_dir='./data/images/'):
        super(LFWImgDataSet, self).__init__(
            txt, load_rgb, transform, target_transform, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        attr_id, img_id1, img_id2 = int(split[3]), int(
            split[0].split('.')[0]), int(split[1].split('.')[0])
        return img_id1, img_id2, attr_id

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, str(img_id) + '.jpg')


class singleDataSet(Dataset):

    def __init__(self, txt, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        split = line.split('/')  # split by comma
        name = split[-1]
        img = Image.open(line)

        if self.transform is not None:
            img = self.transform(img)

        return img, name

    def __len__(self):
        return self.lens


'''
 label is a vector
'''


class MultiLabelDataSet(Dataset):

    def __init__(self, txt, transform=None, target_transform=None, binary=True):
        self.transform = transform
        self.target_transform = target_transform
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.binary = binary

    def __getitem__(self, index):
        line = self.lines[index]
        split = line.split(',')  # split by comma
        img = Image.open(split[0])
        labels = []
        for i in range(1, len(split)):
            val = np.float32(split[i])
            if self.binary:
                if split[i] == '-1':
                    val = np.float32(0.)

            labels.append(val)

        if self.transform is not None:
            img = self.transform(img)

        return img, labels

    def __len__(self):
        return self.lens


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class OutlierLossL2(nn.Module):
    def __init__(self, lamda):
        super(OutlierLossL2, self).__init__()
        self.lamda = lamda

    def forward(self, score1, score2, label, gamma, strength):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        delta = label - score - gamma
        dis_loss = torch.sum(0.5 * strength * delta *
                             delta + self.lamda * strength * torch.abs(gamma))
        loss = dis_loss / bz

        return loss


class OutlierLossLogistic(nn.Module):
    def __init__(self, lamda, binary=False):
        super(OutlierLossLogistic, self).__init__()
        self.lamda = lamda
        self.binary = binary  # if target is binary (0/1)

    def forward(self, score1, score2, label, gamma, strength):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        label = label.squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        # if self.binary:
        #     logits = 1 / (1 + torch.exp(-(score + gamma)))
        #     likelihood = label * \
        #         torch.log(logits) + (1 - label) * torch.log(1 - logits)
        #     dis_loss = torch.sum(-strength * likelihood +
        #                          self.lamda * strength * torch.abs(gamma))
        # else:
        #     dis_loss = torch.sum(
        #         strength * torch.log(1 + torch.exp(-label * (score + gamma))) + 
        #         self.lamda * strength * torch.abs(gamma))
        if self.binary:
            label = label * 2 - 1
        rmargin = label * score + gamma
        dis_loss = torch.sum(
            strength * torch.log(1 + torch.exp(-rmargin)) +
            self.lamda * strength * torch.abs(gamma))
        loss = dis_loss / bz

        return loss


class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()

    def forward(self, score1, score2, label, weight):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        delta = label - score
        dis_loss = torch.sum(0.5 * delta * delta * weight)
        loss = dis_loss / bz

        return loss


class logitloss(nn.Module):
    def __init__(self):
        super(logitloss, self).__init__()

    def forward(self, score1, score2, label, weight):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        label = (label + 1) / 2
        # pdb.set_trace()
        h = 1 / (1 + torch.exp(-score))

        loss = -(label * (h + 1e-5).log() + (1 - label)
                 * (1 - h + 1e-5).log()) * weight
        loss = loss.sum() / bz

        return loss


class huberloss(nn.Module):
    def __init__(self, lamda):
        self.lamda = lamda
        super(huberloss, self).__init__()

    def forward(self, score1, score2, label, weight):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        delta = torch.abs(label - score)
        loss = torch.where(
            delta <= self.lamda, delta * delta * 0.5,
            delta * self.lamda - self.lamda * self.lamda * 0.5)
        loss = (weight * loss).sum() / bz
        return loss


class robust_logit_loss(nn.Module):
    def __init__(self, lamda):
        self.lamda = lamda
        self.eta = math.log(1 - lamda)
        self.xi = math.log((1 - lamda) / lamda)
        super(robust_logit_loss, self).__init__()

    def forward(self, score1, score2, label, weight):
        bz = label.shape[0]
        score = (score1 - score2).squeeze()
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        # target label must be +1/-1
        delta = label * score
        loss = torch.where(
            delta <= self.xi,
            self.lamda * (self.xi - delta) - self.eta,
            torch.log(1 + torch.exp(-delta)))
        loss = (weight * loss).sum() / bz
        return loss
