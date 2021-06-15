import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import LeNet

from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
import data_load
import resnet
import tools

import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='output/results_cdr/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.4)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate')
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=350)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='cdr')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
args = parser.parse_args()

print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
learning_rate = args.lr

# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset=='fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 100
        args.batch_size = 32
        args.num_gradual = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.fmnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.fmnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.fmnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
    
    
    if args.dataset=='mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 100
        args.batch_size = 32
        args.num_gradual = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.mnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.mnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.mnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
        
        
    
    if args.dataset=='cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        args.batch_size = 64
        args.num_gradual = 20
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.cifar10_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.cifar10_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target)
    
    
    if args.dataset=='cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        args.batch_size = 64
        args.num_gradual = 20
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar100_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.cifar100_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.cifar100_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target)
        
    

    return train_dataset, val_dataset, test_dataset


save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)



def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()
    
    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            mask = mask * clip
            param.grad.data = mask * param.grad.data

    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model1, optimizer1, args):
    model1.train()
    train_total=0
    train_correct=0
    clip_narry = np.linspace(1-args.noise_rate, 1, num=args.num_gradual)
    clip_narry = clip_narry[::-1]
    if epoch < args.num_gradual:
        clip = clip_narry[epoch]
   
    clip = (1 - args.noise_rate)
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        data = data.cuda()
        labels = labels.cuda()
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        # Loss transfer 

        prec1, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
       
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, args.train_len//args.batch_size, prec1, loss.item()))
        
      
    train_acc1=float(train_correct)/float(train_total)
    return train_acc1


# Evaluate the Model
def evaluate(test_loader, model1):
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def main(args):
    # Data Loader (Input Pipeline)
    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.seed)
    txtfile = save_dir + "/" + model_str + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
    
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    
    
    
    # Define models
    print('building model...')
    
    if args.dataset == 'mnist':
        clf1 = LeNet()
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'fmnist':
        clf1 = resnet.ResNet50(input_channel=1, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'cifar10':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'cifar100':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=100)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
        
    clf1.cuda()
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 val_acc1 test_acc1\n')

    epoch = 0
    train_acc1 = 0
   
    
    # evaluate models with random weights
    val_acc1 = evaluate(val_loader, clf1)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc1))
    
    test_acc1 = evaluate(test_loader, clf1)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + "\n")
    val_acc_list = []
    test_acc_list = []  
    
    for epoch in range(0, args.n_epoch):
        scheduler1.step()
        print(optimizer1.state_dict()['param_groups'][0]['lr'])
        clf1.train()
        
        train_acc1 = train(train_loader, epoch, clf1, optimizer1, args)
        val_acc1 = evaluate(val_loader, clf1)
        val_acc_list.append(val_acc1)
        test_acc1 = evaluate(test_loader, clf1)
        test_acc_list.append(test_acc1)
        
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + "\n")
    id = np.argmax(np.array(val_acc_list))
    test_acc_max = test_acc_list[id]
    return test_acc_max

if __name__ == '__main__':
    best_acc = main(args)
    
