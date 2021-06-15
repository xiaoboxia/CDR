import numpy as np
import utils
import os
import numpy as np
import torch
import torchvision
from math import inf
from scipy import stats
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch


def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, dataset='mnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    
    clean_train_labels = train_labels[:, np.newaxis]
    
    if noise_type == 'symmetric':
         noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels, 
                                                                                               noise=noise_rate, 
                                                                                               random_state=random_seed, 
                                                                                               nb_classes=num_classes)
    if noise_type == 'pairflip':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_pairflip(clean_train_labels,
                                                                                          noise=noise_rate,
                                                                                          random_state=random_seed,
                                                                                          nb_classes=num_classes)
    if noise_type == 'asymmetric' and dataset == 'mnist':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_mnist(clean_train_labels,
                                                                                                    noise=noise_rate,
                                                                                                    random_state=random_seed,
                                                                                                    nb_classes=num_classes)
        
    if noise_type == 'asymmetric' and dataset == 'fmnist':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_fashionmnist(clean_train_labels,
                                                                                                    noise=noise_rate,
                                                                                                    random_state=random_seed,
                                                                                                    nb_classes=num_classes)
    
    if noise_type == 'asymmetric' and dataset == 'cifar10':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_cifar10(clean_train_labels,
                                                                                                      noise=noise_rate,
                                                                                                      random_state=random_seed,
                                                                                                      nb_classes=num_classes)
        
    if noise_type == 'asymmetric' and dataset == 'cifar100':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_cifar100(clean_train_labels,
                                                                                                       noise=noise_rate,
                                                                                                       random_state=random_seed,
                                                                                                       nb_classes=num_classes)
        
    
        
    if noise_type == 'instance' and dataset == 'mnist':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dataset_ = zip(data, targets)
        noisy_labels = get_instance_noisy_label(n=noise_rate, dataset=dataset_, labels=targets, num_classes=10, feature_size=784, norm_std=0.1, seed=random_seed)
        
        
    if noise_type == 'instance' and dataset == 'fmnist':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dataset_ = zip(data, targets)
        noisy_labels = get_instance_noisy_label(n=noise_rate, dataset=dataset_, labels=targets, num_classes=10, feature_size=784, norm_std=0.1, seed=random_seed)
        
    
    if noise_type == 'instance' and dataset == 'cifar10':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dataset_ = zip(data, targets)
        noisy_labels = get_instance_noisy_label(n=noise_rate, dataset=dataset_, labels=targets, num_classes=10, feature_size=3072, norm_std=0.1, seed=random_seed)
        
    if noise_type == 'instance' and dataset == 'cifar100':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dataset_ = zip(data, targets)
        noisy_labels = get_instance_noisy_label(n=noise_rate, dataset=dataset_, labels=targets, num_classes=100, feature_size=3072, norm_std=0.1, seed=random_seed)

    

    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels

def dataset_split_without_noise(train_images, train_labels, noise_rate, split_per=0.9, random_seed=1, num_class=196):
    total_labels = train_labels[:, np.newaxis]
    #    print(noisy_labels)
    num_samples = int(total_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = total_labels[train_set_index], total_labels[val_set_index]

    return train_set, val_set, train_labels.squeeze(), val_labels.squeeze()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean)
    print(std)
    return mean, std


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed):
    # n -> noise_rate
    # dataset -> mnist, cifar10, cifar100 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed

    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    # flip_distribution = stats.beta(a=0.01, b=(0.01 / n) - 0.01, loc=0, scale=1)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        # print(A.shape)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    #np.save("transition_matrix.npy", P)

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    # print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        # print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    #print(P)
    return np.array(new_label)


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target  