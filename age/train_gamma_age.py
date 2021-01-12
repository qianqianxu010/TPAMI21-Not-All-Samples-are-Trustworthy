from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import AgeDataSet, AgeImgDataSet, OutlierLossL2, OutlierLossLogistic, Identity
from torchvision import transforms
from torchvision.models import resnet50, vgg16_bn, alexnet
import time
import pdb
import os
import math
import argparse

# pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--max_epoch', type=int, default=70)
parser.add_argument('--gamma_epoch', type=int, default=10)
parser.add_argument('-m', '--model', type=str, default='resnet',
                    choices=['resnet', 'vgg', 'alexnet'], help='model name')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-L', type=float, default=10)
parser.add_argument('--lamda', type=float, default=1.2,
                    help='regularization param for gamma')
parser.add_argument('--outer', type=int, default=3)
parser.add_argument('--modname', type=str, default='age', help='model_name')
parser.add_argument('--voted', default=False, action='store_true',
                    help='if training data are voted')
parser.add_argument('--test_file', type=str, default='./data/test_age.txt',
                    help='test set filename')
parser.add_argument('--val_file', type=str, default='./data/val_age.txt',
                    help='val set filename')
parser.add_argument('--img_file', type=str, default='./data/images',
                    help='img set filename')
parser.add_argument('--loss', type=str, default='l2',
                    help='loss type l2 for linear, logit for logistic regression',
                    choices=['l2', 'logit'])
parser.add_argument('--attrib_num', type=int, default=1,
                    help='# of attributes')
parser.add_argument('--binary_label', default=False, action='store_true')
parser.add_argument('--weight_decay', type=float, default=1e-4)
args = parser.parse_args()
DIRFILE = os.path.join(
    '.', '{}_{}_{}_lambda_{:.1f}'.format(
        args.modname, args.loss,
        'voted' if args.voted else 'unvoted', args.lamda))
WRITTERPATH = os.path.join(DIRFILE, 'with_gamma_{}'.format(args.loss))
print('lambda :{:2.4f}'.format(args.lamda))
all_pair_num, pairs_per_attr = 15000, 3002


id2file_dict = dict()
file2id_dict = dict()
for file in os.listdir(args.img_file):
    img_id = file.split('.')[0].replace('A', '').replace(
        'a', '1').replace('b', '2').replace('c', '3')
    id2file_dict[int(img_id)] = file
    file2id_dict[file] = int(img_id)

train_file = './data/train_{}_{}.txt'.format(
    'voted' if args.voted else 'unvoted', args.modname)


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
elif args.model == 'alexnet':
    model = alexnet(pretrained=False).cuda()
    dicts = torch.load('./face_alexnet_checkpt_80.pkl')
    model_state_dict = dicts['model_state_dict']
    model.load_state_dict(model_state_dict)
    model_dim = 4096
    train_batch_size = 128
    test_batch_size = 128
elif args.model == 'vgg':
    model = vgg16_bn(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(model_dim, 1)
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
# ])
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
print('loading test_pair_data:', len(test_pair_data_wo_img))

test_img_data = AgeImgDataSet(txt=args.test_file, load_rgb=True,
                              root_dir=args.img_file,
                              file2id_dict=file2id_dict,
                              id2file_dict=id2file_dict,
                              transform=train_transform)
test_img_loader = DataLoader(dataset=test_img_data,
                             batch_size=test_batch_size,
                             shuffle=False, **kwargs)
print('loading test_img_data:', len(test_img_data))

print('Traing sample number: {}'.format(all_pair_num))
print('Traing outlier ratio: {:.4f}'.format(
    train_pair_data_with_img.get_outlier_num() / all_pair_num))

# ============= linear classifiers for global features ================


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.01)
        # nn.init.normal_(m.bias.data, 0.01)
        nn.init.constant_(m.bias.data, 0)


weight_init(model.fc)

# ================= loss function =====================
gammas = torch.zeros((pairs_per_attr, 2))
gammas_n = torch.zeros((pairs_per_attr, 2))
gammas_old = torch.zeros((pairs_per_attr, 2))
strengths = torch.zeros((pairs_per_attr, 2))
zero = torch.zeros((1)).cuda()
lamda = args.lamda
if args.loss == 'l2':
    loss_fn = OutlierLossL2(lamda=lamda)
elif args.loss == 'logit':
    loss_fn = OutlierLossLogistic(lamda=lamda, binary=args.binary_label)
    gammas = gammas.cuda()
    gammas_n = gammas_n.cuda()
    gammas_old = gammas_old.cuda()
else:
    print('loss should either be l2 or logit, please choose again!')
    quit()
print('ranking loss:', loss_fn)

# ================= hyper-parameters =====================
lr = args.learning_rate
print('Initial lr:', lr)
# update one part params.
optimizer_a = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=args.weight_decay)
# optimizer_a = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
multisteps = [30, 50, 70, 90, 140]
scheduler_a = optim.lr_scheduler.MultiStepLR(
    optimizer_a, milestones=multisteps, gamma=0.1)
# use CUDA
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
# show interval
log_interval = 20
SAVE_INTERVAL = 10


if not os.path.exists(DIRFILE):
    os.mkdir(DIRFILE)


# for param in model.parameters():
#     param.requires_grad = False
# model.fc.weight.requires_grad = True
# model.fc.bias.requires_grad = True


def get_strengths(dataloader=train_loader_wo_img):
    with torch.no_grad():
        for step, (_, _, label, _, pair_id, strength) in enumerate(dataloader):
            bz = label.shape[0]
            for k in range(bz):  # for a batch
                strengths[int(pair_id[k]), int(
                    (label[k] + 1) / 2)] = strength[k]


def train(epoch):
    model.train()

    loss = 0
    batch_cnt = 0
    for step, (_, _, img1, img2, label, _, pair_id, strength) in enumerate(train_loader):
        img1, img2, label, pair_id, strength = img1.cuda(), img2.cuda(
        ), label.cuda(), pair_id.cuda(), strength.cuda()
        optimizer_a.zero_grad()
        bz = label.shape[0]
        output1 = model(img1)
        output2 = model(img2)
        gamma = torch.zeros(bz).cuda()
        for k in range(bz):  # for a batch
            gamma[k] = gammas[int(pair_id[k]), int((label[k] + 1) / 2)]
        cls_loss = loss_fn(output1, output2, label, gamma, strength)
        cls_loss.backward()
        optimizer_a.step()

        loss += cls_loss.item()
        batch_cnt += 1
        # print('Training cls_loss:', cls_loss.item())

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
        cptname = 'naive_{:}_checkpt_{:}.pkl'.format(args.modname, epoch)
        cptpath = os.path.join(DIRFILE, cptname)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer': optimizer_a.state_dict()}, cptpath)

    return loss / batch_cnt


def test(epoch, set_flag='test'):
    model.eval()
    test_loss = 0
    correct = 0
    batch_cnt = 0
    pair_loader = {'train': train_loader_wo_img,
                   'val': val_loader_wo_img, 'test': test_loader_wo_img}
    dataloader = pair_loader[set_flag]
    img_loader = {'train': train_img_loader,
                  'val': val_img_loader, 'test': test_img_loader}
    dict_scores = get_all_scores(img_loader[set_flag])
    if set_flag == 'test':
        fw = open(WRITTERPATH, 'w')
    with torch.no_grad():
        for step, (img_id1, img_id2, label, _, pair_id, strength) in enumerate(dataloader):
            img_id1, img_id2, label, pair_id, strength = img_id1.to(device), img_id2.to(
                device), label.to(device), pair_id.to(device), strength.to(device)
            bz = label.shape[0]
            output1 = torch.zeros(bz).cuda()
            output2 = torch.zeros(bz).cuda()
            gamma = torch.zeros(bz).cuda()
            for k in range(bz):  # for a batch
                # print(img_id1[k].item())
                output1[k] = dict_scores[int(img_id1[k])]
                output2[k] = dict_scores[int(img_id2[k])]
                if set_flag == 'train':
                    gamma[k] = gammas[int(pair_id[k]), int((label[k] + 1) / 2)]
            if set_flag == 'train':
                cls_loss = loss_fn(output1, output2,
                                   label, gamma, strength)
                test_loss += cls_loss.item()
                # print('Training cls_loss in test', cls_loss.item())
            batch_cnt += 1
            # pdb.set_trace()
            pred = output1 - output2
            # print(torch.cat((torch.unsqueeze(pred, 1), torch.unsqueeze(label, 1)), 1))
            if set_flag == 'test':
                for k in range(bz):
                    fw.write('Pair: %d, Label: %d, Score: %.4f, Strength: %d\n' %
                             (pair_id[k], label[k], pred[k], strength[k]))
            pred[pred > 0] = 1
            pred[pred <= 0] = 0 if args.binary_label else -1
            pred = pred.type(torch.cuda.LongTensor)
            label = label.type(pred.dtype)
            correct += pred.eq(label.view_as(pred)).sum().item()
    if set_flag == 'test':
        fw.close()

    if set_flag == 'train':
        test_loss /= batch_cnt
        print('train set {}: average loss:{:.4f}, acc:{:.4f}'.format(
            len(dataloader.dataset), test_loss,
            correct / len(dataloader.dataset)))
    else:
        print('{} set {}: acc:{:.4f}'.format(
            set_flag, len(dataloader.dataset),
            correct / len(dataloader.dataset)))


def logit_fun(s, gamma):
    return 1 / (1 + torch.exp(-(s + gamma)))


def solve_l1(x, lamd):
    return torch.max(torch.abs(x) - lamd, zero) * torch.sign(x)


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


def update_gammas(type, dataloader=train_loader_wo_img):
    # pdb.set_trace()
    model.eval()
    dict_scores = get_all_scores()
    if type == 'l2':
        with torch.no_grad():
            for step, (img_id1, img_id2, label, _, pair_id, strength) in enumerate(dataloader):
                label, pair_id = label.to(device), pair_id.to(device)
                bz = label.shape[0]
                for k in range(bz):  # for a batch
                    test_score1 = dict_scores[int(img_id1[k])]
                    test_score2 = dict_scores[int(img_id2[k])]
                    delta = label[k] - (test_score1 - test_score2)
                    gammas[int(pair_id[k]), int(
                        (label[k] + 1) / 2)] = solve_l1(delta, lamda)
    elif type == 'logit':
        L = args.L
        t_new = 1
        with torch.no_grad():
            for _ in range(args.gamma_epoch):
                gammas_old = gammas
                t_old = t_new
                t_new = (1 + math.sqrt(1 + 4 * t_old * t_old)) / 2
                dt = (t_old - 1) / t_new
                for step, (img_id1, img_id2, label, _, pair_id, strength) in enumerate(dataloader):
                    label, pair_id, strength = label.to(
                        device), pair_id.to(device), strength.to(device)
                    bz = label.shape[0]
                    for k in range(bz):  # for a batch
                        pair_idx = int(pair_id[k])
                        label_k = int(label[k]) if args.binary_label else int(
                            (label[k] + 1) / 2)
                        labelpn_k = 2 * label_k - 1
                        test_score1 = dict_scores[int(img_id1[k])]
                        test_score2 = dict_scores[int(img_id2[k])]
                        gamma = gammas_n[pair_idx, label_k]
                        h = logit_fun(labelpn_k * (test_score1 - test_score2), gamma)
                        # delta = gamma - (1 / L) * strength[k] * (h - 1)
                        # zeta = strength[k] * lamda / L
                        delta = gamma - (1 / L) * (h - 1)
                        zeta = lamda / L
                        gammas[pair_idx, label_k] = solve_l1(delta, zeta)
                        gammas_n[pair_idx, label_k] = gammas[pair_idx, label_k] + \
                            dt * (gammas[pair_idx, label_k] -
                                  gammas_old[pair_idx, label_k])

    else:
        print('loss could either be l2 or logit')
        quit()


# ===================== training/testing =================
EPOCHS = args.max_epoch
OUTERS = args.outer
for outer in range(OUTERS):
    last_loss = -1
    for epoch in range(1, EPOCHS + 1):
        scheduler_a.step()
        train_loss = train(epoch)
        # test(epoch, True, train_loader_wo_img)
        if last_loss > -1 and abs(train_loss - last_loss) < 1e-4:
            print('Early stopping...')
            test(epoch, 'val')
            test(epoch)
            break
        last_loss = train_loss
    print('Training loss before update gamma...')
    test(epoch, 'train')
    print('Update gamma...')
    # for epoch in range(args.gamma_epoch):
    update_gammas(args.loss)
    abs_gammas = gammas.abs()
    # print(abs_gammas.max(), abs_gammas[abs_gammas > 0].min())
    # print('Gamma', gammas)
    print('Training loss after update gamma...')
    test(epoch, 'train')
    test(epoch, 'val')
    test(epoch)

get_strengths()
cnt_outlier = 0
threshold = 1e-6 if args.loss is 'l2' else 1e-5
OUTLIERPATH = os.path.join(DIRFILE, 'outliers.txt')
GAMMAPATH = os.path.join(DIRFILE, 'gamma.mat')
with open(OUTLIERPATH, 'w') as f:
    for j in range(pairs_per_attr):
        if abs(gammas[j, 0].item()) > threshold:
            cnt_outlier += strengths[j, 0].item()
            f.write('Pair: %d, Label: %d, Gamma: %.4f\n' %
                    (j, -1, gammas[j, 0]))
        if abs(gammas[j, 1].item()) > threshold:
            cnt_outlier += strengths[j, 1].item()
            f.write('Pair: %d, Label: %d, Gamma: %.4f\n' %
                    (j, 1, gammas[j, 1]))
print('Training outlier ratio: {:.4f}%'.format(
    100 * cnt_outlier / all_pair_num))

sio.savemat(GAMMAPATH, {'gammas': gammas.cpu().numpy()})
