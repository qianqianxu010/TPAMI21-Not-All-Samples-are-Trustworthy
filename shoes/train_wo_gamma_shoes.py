from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import ShoesDataSet, ShoesImgDataSet, LinearLoss, logitloss, huberloss, robust_logit_loss, Identity
from torchvision import transforms
from torchvision.models import resnet50, vgg16_bn
import predeepscore as process

import time
#import pdb
import os
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('-e', '--max_epoch', type=int, default=100)
parser.add_argument('-m', '--model', type=str, default='resnet',
                    choices=['resnet', 'vgg', 'alexnet'], help='model name')
parser.add_argument('--modname', type=str, default='shoes', help='model_name')
parser.add_argument('--voted', default=False, action='store_true',
                    help='if training data are voted')
parser.add_argument('--test_file', type=str, default='./data/shoes_test_file.txt',
                    help='test set filename')
parser.add_argument('--val_file', type=str, default='./data/shoes_val_file.txt',
                    help='val set filename')
parser.add_argument('--img_file', type=str, default='./data/images',
                    help='img set filename')
parser.add_argument('--lamda', type=float, default=0.1,
                    help='param for huber/robust_logit loss')
parser.add_argument('--loss', type=str, default='l2',
                    help='loss type l2 for linear, logit for logistic regression',
                    choices=['l2', 'logit', 'huber', 'rlogit'])
parser.add_argument('--attrib_num', type=int, default=7,
                    help='# of attributes')
parser.add_argument('--binary_label', default=False, action='store_true')
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

train_file = './data/{}_train_{}.txt'.format(
    args.modname, 'voted' if args.voted else 'unvoted')
FILEDICT = sio.loadmat('./data/shoes_attributes.mat')['im_names'][0]

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
    model = resnet50(pretrained=True)
    model.fc = Identity()
    model = model.cuda()
    model_dim = 2048
    train_batch_size = 32
    test_batch_size = 32
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
     transforms.ToTensor()])

# (i, j, label, attr_id, pair_id, strength)
# e.g.: for attribute 1, there are 5 comparisons between 3rd pair(i, j):
#       (i, j, 1) * 3, (i, j, -1) * 2.  Then the samples will be:
#       (i, j, 1, 1, 3, 3) * 1, (i, j, -1, 1, 3, 2) * 1
train_pair_data_with_img = ShoesDataSet(txt=train_file, file_dict=FILEDICT,
                                        binary=args.binary_label, load_img=True,
                                        transform=train_transform)
train_loader = DataLoader(dataset=train_pair_data_with_img, batch_size=train_batch_size,
                          shuffle=True, **kwargs)
print('loading train_pair_data:', len(train_pair_data_with_img))

train_pair_data_wo_img = ShoesDataSet(txt=train_file, file_dict=FILEDICT,
                                      binary=args.binary_label, load_img=False)
train_loader_wo_img = DataLoader(dataset=train_pair_data_wo_img, batch_size=train_batch_size,
                                 shuffle=True, **kwargs)

val_data = ShoesDataSet(txt=args.val_file,
                        binary=args.binary_label,
                        transform=transforms.Compose(
                            [transforms.Resize((224, 224)),
                             transforms.ToTensor()]), file_dict=FILEDICT)
val_loader = DataLoader(dataset=val_data, batch_size=test_batch_size,
                        shuffle=False, **kwargs)
print('loading val_data:', len(val_data))

test_data = ShoesDataSet(txt=args.test_file,
                         binary=args.binary_label,
                         transform=transforms.Compose(
                             [transforms.Resize((224, 224)),
                              transforms.ToTensor()]), file_dict=FILEDICT)
test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size,
                         shuffle=False, **kwargs)
print('loading test_data:', len(test_data))


# ============= linear classifiers for global features ================
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.01)
        nn.init.constant_(m.bias.data, 0)


score_functions = nn.ModuleList(
    [nn.Linear(model_dim, 1).cuda() for i in range(args.attrib_num)])
for i in range(args.attrib_num):
    weight_init(score_functions[i])

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
params_a = [{'params': model.parameters()}, {
    'params': score_functions.parameters()}]
# update one part params.
optimizer_a = optim.SGD(params_a, lr=lr,
                        momentum=0.9, weight_decay=args.weight_decay)
# optimizer_a = optim.Adam(params_a, lr=lr)
multisteps = [20, 40, 60, 120]
scheduler_a = optim.lr_scheduler.MultiStepLR(
    optimizer_a, milestones=multisteps, gamma=0.1)
# use CUDA
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
# show interval
log_interval = 20
SAVE_INTERVAL = 5
TEST_INTERVAL = 5


def train(epoch):
    model.train()

    loss = 0
    batch_cnt = 0
    for step, (_, _, img1, img2, label, attr_id, pair_id, strength) in enumerate(train_loader):
        img1, img2, label, attr_id, pair_id, strength = img1.cuda(), img2.cuda(
        ), label.cuda(), attr_id.cuda(), pair_id.cuda(), strength.cuda()
        optimizer_a.zero_grad()
        bz = label.shape[0]
        output1 = model(img1)
        output2 = model(img2)
        train_score1 = torch.zeros(bz).cuda()
        train_score2 = torch.zeros(bz).cuda()
        for k in range(bz):  # for a batch
            idx = int(attr_id[k])
            train_score1[k] = score_functions[idx](output1[k])
            train_score2[k] = score_functions[idx](output2[k])
        cls_loss = loss_fn(train_score1, train_score2, label, strength)

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
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'score_funcs_dict': score_functions.state_dict(),
                    'optimizer': optimizer_a.state_dict()},
                   '{}/naive_{}_checkpt_{}.pkl'.format(DIRFILE, args.modname, epoch))
    return loss / batch_cnt


def test(epoch, set_flag='test'):
    pair_loader = {'train': train_loader,
                   'val': val_loader, 'test': test_loader}
    dataloader = pair_loader[set_flag]
    model.eval()
    score_functions.eval()
    correct = 0
    if set_flag == 'test':
        fw = open(WRITTERPATH, 'w')
    with torch.no_grad():
        for step, (_, _, img1, img2, label, attr_id, pair_id, strength) in enumerate(dataloader):
            img1, img2, label, attr_id, pair_id, strength = img1.to(device), img2.to(
                device), label.to(device), attr_id.to(device), pair_id.to(device), strength.to(device)
            output1 = model(img1)
            output2 = model(img2)
            bz = attr_id.shape[0]
            test_score1 = torch.zeros(bz).cuda()
            test_score2 = torch.zeros(bz).cuda()
            for k in range(bz):  # for a batch
                idx = int(attr_id[k])
                test_score1[k] = score_functions[idx](output1[k])
                test_score2[k] = score_functions[idx](output2[k])
            pred = test_score1 - test_score2
            if set_flag == 'test':
                for k in range(bz):
                    fw.write('Attribute: %d, Pair: %d, Label: %d, Score: %.4f, Strength: %d\n' %
                             (attr_id[k], pair_id[k], label[k], pred[k], strength[k]))
            pred[pred > 0] = 1
            pred[pred <= 0] = 0 if args.binary_label else -1
            pred = pred.type(torch.cuda.LongTensor)
            label = label.type(pred.dtype)
            correct += pred.eq(label.view_as(pred)).sum().item()
    if set_flag == 'test':
        fw.close()

    print('{} set {}: acc:{:.4f}'.format(
        set_flag, len(dataloader.dataset), correct / len(dataloader.dataset)))


# ===================== training/testing =================
EPOCHS = args.max_epoch
last_loss = -1
for epoch in range(1, EPOCHS + 1):
    scheduler_a.step()
    train_loss = train(epoch)
    if last_loss > -1 and abs(train_loss - last_loss) < 1e-5:
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
