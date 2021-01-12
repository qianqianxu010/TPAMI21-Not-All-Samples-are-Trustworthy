from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import AgeDataSet, AgeImgDataSet, LinearLoss, logitloss, huberloss, robust_logit_loss, Identity
from torchvision import transforms
from torchvision.models import resnet50, vgg16_bn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import time
#import pdb
import os
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('-e', '--max_epoch', type=int, default=120)
parser.add_argument('-m', '--model', type=str, default='resnet',
                    choices=['resnet', 'vgg', 'alexnet'], help='model name')
parser.add_argument('--modname', type=str, default='age', help='model_name')
parser.add_argument('--test_file', type=str, default='./data/test_age.txt',
                    help='test set filename')
parser.add_argument('--val_file', type=str, default='./data/val_age.txt',
                    help='val set filename')
parser.add_argument('--img_file', type=str, default='./data/images',
                    help='img set filename')
parser.add_argument('--voted', default=False, action='store_true',
                    help='if training data are voted')
parser.add_argument('--lamda', type=float, default=0.1,
                    help='param for huber/robust_logit loss')
parser.add_argument('--loss', type=str, default='l2',
                    help='loss type l2 for linear, logit for logistic regression',
                    choices=['l2', 'logit', 'huber', 'rlogit'])
parser.add_argument('--attrib_num', type=int, default=1,
                    help='# of attributes')
parser.add_argument('--binary_label', default=False, action='store_true')
parser.add_argument('--weight_decay', type=float, default=1e-4)
args = parser.parse_args()

id2file_dict = dict()
file2id_dict = dict()
for file in os.listdir(args.img_file):
    img_id = file.split('.')[0].replace('A', '').replace(
        'a', '1').replace('b', '2').replace('c', '3')
    id2file_dict[int(img_id)] = file
    file2id_dict[file] = int(img_id)

train_file = './data/train_{}_{}.txt'.format(
    'voted' if args.voted else 'unvoted', args.modname)

DIRFILE = os.path.join(
    '.', '{}_{}_{}_wo_gamma_{:.1f}'.format(
        args.modname, args.loss,
        'voted' if args.voted else 'unvoted',
        args.lamda if args.loss in ['huber', 'rlogit'] else 0))
WRITTERPATH = os.path.join(DIRFILE, 'wo_gamma_{:}'.format(args.loss))

if not os.path.exists(DIRFILE):
    os.mkdir(DIRFILE)


# ================= load pretrained backbone =====================
print('Select pretrained model--', args.model)

if args.model == 'resnet':
    model = resnet50(pretrained=False)
    resnet50_dict = model.state_dict()
    model_dim = 2048
    train_batch_size = 32
    test_batch_size = 32
    model.fc = nn.Linear(model_dim, 62)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load('./age_estimation_resnet50.pth.tar')['state_dict']
    for k, v in state_dict.items():
        if 'num_batches_tracked' not in k:
            new_state_dict[k[7:]] = v
    resnet50_dict.update(new_state_dict)
    model.load_state_dict(resnet50_dict)
    model.fc = nn.Linear(model_dim, 1)

    model = model.cuda()
elif args.model == 'vgg':
    model = vgg16_bn(pretrained=True)
    model.classifier._modules['6'] = Identity()
    model = model.cuda()
    model_dim = 4096
    train_batch_size = 32
    test_batch_size = 32
else:
    print('please select proper model such as \'alexnet\'')
    quit()

# ================= load dataset =====================
kwargs = {'num_workers': 8, 'pin_memory': True}
train_transform = transforms.Compose(
    # [transforms.Resize((256, 256)),
    #  transforms.RandomCrop((224, 224)),
    #  transforms.RandomHorizontalFlip(),
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

# (i, j, label, attr_id, pair_id, strength)
# e.g.: for attribute 1, there are 5 comparisons between 3rd pair(i, j):
#       (i, j, 1) * 3, (i, j, -1) * 2.  Then the samples will be:
#       (i, j, 1, 1, 3, 3) * 1, (i, j, -1, 1, 3, 2) * 1
train_pair_data_with_img = AgeDataSet(txt=train_file,
                                      root_dir=args.img_file,
                                      file2id_dict=file2id_dict,
                                      id2file_dict=id2file_dict,
                                      binary=args.binary_label,
                                      load_img=True, load_rgb=True,
                                      transform=train_transform)
train_loader = DataLoader(dataset=train_pair_data_with_img,
                          batch_size=train_batch_size,
                          shuffle=True, **kwargs)
print('loading train_pair_data:', len(train_pair_data_with_img))

train_pair_data_wo_img = AgeDataSet(txt=train_file,
                                    root_dir=args.img_file,
                                    file2id_dict=file2id_dict,
                                    id2file_dict=id2file_dict,
                                    binary=args.binary_label, load_img=False)
train_loader_wo_img = DataLoader(dataset=train_pair_data_wo_img,
                                 batch_size=train_batch_size,
                                 shuffle=True, **kwargs)

train_img_data = AgeImgDataSet(txt=train_file, load_rgb=True,
                               root_dir=args.img_file,
                               file2id_dict=file2id_dict,
                               id2file_dict=id2file_dict,
                               transform=train_transform)
train_img_loader = DataLoader(dataset=train_img_data,
                              batch_size=train_batch_size,
                              shuffle=False, **kwargs)
print('loading train_img_data:', len(train_img_data))

val_pair_data_wo_img = AgeDataSet(txt=args.val_file,
                                  root_dir=args.img_file,
                                  file2id_dict=file2id_dict,
                                  id2file_dict=id2file_dict,
                                  binary=args.binary_label, load_img=False)
val_loader_wo_img = DataLoader(dataset=val_pair_data_wo_img,
                               batch_size=test_batch_size,
                               shuffle=False, **kwargs)
print('loading val_pair_data:', len(val_pair_data_wo_img))

val_img_data = AgeImgDataSet(txt=args.val_file, load_rgb=True,
                             root_dir=args.img_file,
                             file2id_dict=file2id_dict,
                             id2file_dict=id2file_dict,
                             transform=train_transform)
val_img_loader = DataLoader(dataset=val_img_data,
                            batch_size=test_batch_size,
                            shuffle=False, **kwargs)
print('loading val_img_data:', len(val_img_data))

test_pair_data_wo_img = AgeDataSet(txt=args.test_file,
                                   root_dir=args.img_file,
                                   file2id_dict=file2id_dict,
                                   id2file_dict=id2file_dict,
                                   binary=args.binary_label, load_img=False)
test_loader_wo_img = DataLoader(dataset=test_pair_data_wo_img,
                                batch_size=test_batch_size,
                                shuffle=False, **kwargs)

test_img_data = AgeImgDataSet(txt=args.test_file, load_rgb=True,
                              root_dir=args.img_file,
                              file2id_dict=file2id_dict,
                              id2file_dict=id2file_dict,
                              transform=train_transform)
test_img_loader = DataLoader(dataset=test_img_data,
                             batch_size=test_batch_size,
                             shuffle=False, **kwargs)
print('loading test_img_data:', len(test_img_data))


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.01)
        nn.init.constant_(m.bias.data, 0)


weight_init(model.fc)

# ================= loss function =====================
if args.loss == 'l2':
    loss_fn = LinearLoss()
elif args.loss == 'logit':
    loss_fn = logitloss()
elif args.loss == 'huber':
    loss_fn = huberloss(args.lamda)
elif args.loss == 'rlogit':
    loss_fn = robust_logit_loss(args.lamda)
print('ranking loss:', loss_fn)

# ================= hyper-parameters =====================
lr = args.learning_rate
print('Initial lr:', lr)
# update one part params.
# params_a = [{'params': model.fc.parameters()},
#             {'params': [param for name, param in model.named_parameters()
#                         if 'fc' not in name],
#              'lr': 1e-4, 'weight_decay': 1e-4}]
optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                        weight_decay=args.weight_decay)
# optimizer_a = optim.Adam(params_a, lr=lr)
multisteps = [30, 50, 70, 90, 140]
scheduler_a = optim.lr_scheduler.MultiStepLR(
    optimizer_a, milestones=multisteps, gamma=0.1)
# use CUDA
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
# show interval
log_interval = 20
SAVE_INTERVAL = 4
TEST_INTERVAL = 2


def train(epoch):
    model.train()

    loss = 0
    batch_cnt = 0
    for step, (_, _, img1, img2, label, _, pair_id, strength) in enumerate(train_loader):
        img1, img2, label, pair_id, strength = img1.cuda(), img2.cuda(
        ), label.cuda(), pair_id.cuda(), strength.cuda()
        optimizer_a.zero_grad()
        output1 = model(img1)
        output2 = model(img2)
        cls_loss = loss_fn(output1, output2, label, strength)

        cls_loss.backward()
        optimizer_a.step()

        loss += cls_loss.item()
        batch_cnt += 1

        # show info.
        # if step % log_interval == 0:
        #     print('Time:', time.asctime(time.localtime(time.time())),
        #           '|Epoch:', epoch,
        #           '|cls_loss:', cls_loss.item()
        #           )
    print('Time:', time.asctime(time.localtime(time.time())),
          '|Epoch:', epoch,
          '|total_loss:', loss / batch_cnt
          )
    if epoch - 1 in multisteps:
        print('adjust learning rate...')
        for param_group in optimizer_a.param_groups:
            print('lr:', param_group['lr'])

    # save params
    if epoch % SAVE_INTERVAL == 0:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer_a.state_dict()},
                   '{}/naive_{}_checkpt_{}.pkl'.format(DIRFILE, args.modname, epoch))

    return loss / batch_cnt


def get_all_scores(dataloader=train_img_loader):
    dict_scores = dict()
    with torch.no_grad():
        for step, (img, img_id, _) in enumerate(dataloader):
            img_id, img = img_id.to(device), img.to(device)
            output = model(img)
            bz = img_id.shape[0]
            for k in range(bz):
                dict_scores[int(img_id[k])] = output[k].item()
    return dict_scores


def test(epoch, set_flag='test'):
    pair_loader = {'train': train_loader_wo_img,
                   'val': val_loader_wo_img, 'test': test_loader_wo_img}
    dataloader = pair_loader[set_flag]
    model.eval()
    # test_loss = 0
    correct = 0
    batch_cnt = 0
    img_loader = {'train': train_img_loader,
                  'val': val_img_loader, 'test': test_img_loader}
    dict_scores = get_all_scores(img_loader[set_flag])
    # y_test = []
    # y_pred = []
    # y_pred_score = []
    if set_flag == 'test':
        fw = open(WRITTERPATH, 'w')
    with torch.no_grad():
        for step, (img_id1, img_id2, label, _, pair_id, strength) in enumerate(dataloader):
            img_id1, img_id2, label, pair_id, strength = img_id1.to(device), img_id2.to(
                device), label.to(device), pair_id.to(device), strength.to(device)
            bz = label.shape[0]
            output1 = torch.zeros(bz).cuda()
            output2 = torch.zeros(bz).cuda()
            for k in range(bz):  # for a batch
                output1[k] = dict_scores[int(img_id1[k])]
                output2[k] = dict_scores[int(img_id2[k])]
            pred = (output1 - output2).view_as(label)
            if set_flag == 'test':
                for k in range(bz):
                    fw.write('Pair: %d, Label: %d, Score: %.4f, Strength: %d\n' %
                             (pair_id[k], label[k], pred[k], strength[k]))
            batch_cnt += 1
            # y_pred_score.extend(pred.cpu().numpy().tolist())
            pred[pred > 0] = 1
            pred[pred <= 0] = 0 if args.binary_label else -1
            pred = pred.type(torch.cuda.LongTensor)
            label = label.type(pred.dtype)
            correct += pred.eq(label.view_as(pred)).sum().item()
            # y_test.extend(label.cpu().numpy().tolist())
            # y_pred.extend(pred.cpu().numpy().tolist())
    if set_flag == 'test':
        fw.close()
    # total_acc = accuracy_score(y_test, y_pred)
    # f1, pre, re = f1_score(y_test, y_pred), precision_score(
    #     y_test, y_pred), recall_score(y_test, y_pred)
    # auc = roc_auc_score(y_test, y_pred_score)
    # print('Test: acc:{:.4f}, f1:{:.4f}, p:{:.4f}, r:{:.4f}, auc:{:.4f}'.format(total_acc, f1, pre, re, auc))

    # test_loss /= batch_cnt
    print('{} set {}: acc:{:.4f}'.format(
        set_flag, len(dataloader.dataset), correct / len(dataloader.dataset)))


# ===================== training/testing =================
EPOCHS = args.max_epoch
last_loss = -1
for epoch in range(1, EPOCHS + 1):
    scheduler_a.step()
    train_loss = train(epoch)
    if last_loss > -1 and abs(train_loss - last_loss) < 1e-6:
        print('Test...')
        # test(epoch, 'train')
        test(epoch, 'val')
        test(epoch, 'test')
        break
    last_loss = train_loss
    if epoch % TEST_INTERVAL == 0:
        print('Test...')
        # test(epoch, 'train')
        test(epoch, 'val')
        test(epoch, 'test')
