import re
from pathlib import PurePath as Path
import numpy as np
import sys
sys.path.append("../")

import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from models import MLP


use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

glob_param = {
    'pruning_perc': 90.,
    'batch_size': 128, 
    'test_batch_size': 100,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}

print("Read param", glob_param)
# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=True, 
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, 
    batch_size=glob_param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=glob_param['test_batch_size'], shuffle=True)




def generator_random(input_size = 27, output_size = 3, num_layers = 2, neurons = [64, 64]):
    n = input_size
    n_ = output_size
    k = num_layers
    N = neurons
    
    W = [10 * np.random.random([n, N[0]]) - 5.0]
    B = [10 * np.random.random([N[0]]) - 5.0]
    for i in range(1, k):
        w_i = 10 * np.random.random([N[i - 1], N[i]])
        W.append(w_i - 5.0)
        b_i = 10 * np.random.random([N[i]]) 
        B.append(b_i - 5.0)
    
    w_i = 10 * np.random.random([N[k - 1], n_]) - 5.0
    W.append(w_i)
    b_i = 10 * np.random.random([n_]) - 5.0
    B.append(b_i)

    write_to_file(input_size = n, output_size = n_, neurons = N, W = W, B = B) 
    
def write_to_file(input_size, output_size, neurons, W, B, file_name = 'network_files/neural_network_information_16'):
    eps = 0
    n = input_size
    n_ = output_size
    N = neurons
    k = len(N)

    assert len(W) == len(B)
    num = 0

    for o in range(output_size):      
        f = open(file_name + "_" + str(o), 'w')
        f.write(str(n) + '\n1\n' + str(k) + '\n')
         
        f.write(str(N[0]) + '\n')

        for l in range(0, k - 1):
            f.write(str(N[l]) + '\n')

        for i in range(N[0]):
            for j in range(n):
                f.write(str(W[0][j, i]) + '\n')
                num += (abs(W[0][j, i] > eps))
            f.write(str(B[0][i]) + '\n')
            num += (abs(B[0][i] > eps))
         
        for l in range(1, len(N)):
            for i in range(N[l]):
                for j in range(N[l-1]):
                    f.write(str(W[l][j, i]) + '\n')
                    num += (abs(W[l][j, i] > eps))
                f.write(str(B[l][i]) + '\n')
                num += (abs(B[l][i] > eps))
             
        for j in range(N[-1]):
            f.write(str(W[-1][j, o]) + '\n')
            num += (abs(W[-1][j, o] > eps))
        f.write(str(B[-1][o]) + '\n')
        num += (abs(B[-1][o] > eps))
    print("Num of nonzero parameters is {}".format(num))

    num = read_from_file(file_name + '_' + str(o))
    print("Num of nonzero parameters is actually {}".format(num))


def read_from_file(file_name):
    num = 0
    f = open(file_name, 'r')
    lines = f.readlines()
    num_layer = int(lines[2])
    
    for i in range(num_layer + 2 + 1, len(lines)):
        par = float(lines[i])
        if abs(par) == 0.0:
            print(par)
        num += (abs(par) >= 0.0)
    return num

        
def load_from_checkpoint(file_name):
    input_size = 27
    output_size = 3
    num_layers = 2
    N = [64, 64]
    W = []
    B = []

    net = Controller(input_size, output_size).to(device)
    model_source = torch.load(file_name)
    net.load_state_dict(model_source)

    for name, param in net.named_parameters():
        if param.requires_grad:
            if re.match('.*weight', name) is not None:
                W.append(param.detach().cpu().numpy().T)
            elif re.match('.*bias', name) is not None:
                B.append(param.detach().cpu().numpy())

    write_to_file(input_size = input_size, output_size = output_size, neurons = N, W = W, B = B) 

def prune_model_neuron(file_name = '../models/mlp_pretrained.pkl', rates = None):
    global glob_param
    input_size = 28 * 28
    output_size = 10
    num_layers = 2
    N = [200, 200]
    W = []
    B = []

    net = MLP()
    net.load_state_dict(torch.load(file_name))
    if torch.cuda.is_available():
        print('CUDA ensabled.')
        net.cuda()
    print("--- Pretrained network loaded ---")

    prune = 1. 
    if prune:
        glob_param['pruning_perc'] = prune
        # prune the weights
        masks = weight_prune(net, glob_param['pruning_perc'])
        net.set_masks(masks)
        print("--- {}% parameters pruned ---".format(glob_param['pruning_perc']))
        test(net, loader_test)
        
        # Retraining
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=glob_param['learning_rate'], 
                                        weight_decay=glob_param['weight_decay'])
        
        train(net, criterion, optimizer, glob_param, loader_train)
        
        
        # Check accuracy and nonzeros weights in each layer
        print("--- After retraining ---")
        test(net, loader_test)
        prune_rate(net)



    acc, avg_act_1, avg_act_2 = test(net, loader_test)
    # print(avg_act_1)
    # print(avg_act_2)

    for name, param in net.named_parameters():
        if param.requires_grad:
            if re.match('.*weight', name) is not None:
                W.append(param.detach().cpu().numpy().T)
            elif re.match('.*bias', name) is not None:
                B.append(param.detach().cpu().numpy())
            # if re.match('.*weight', name) is not None:
            #     W.append(param.detach().cpu().numpy().astype(np.float16).T)
            # elif re.match('.*bias', name) is not None:
            #     B.append(param.detach().cpu().numpy().astype(np.float16))

    for rate in rates:
        N_1 = int(200 * rate)
        N_2 = int(200 * rate)
        N_ = [input_size, N_1, N_2, output_size]

    
        N_i_idx = torch.tensor(range(input_size))
        N_o_idx = torch.tensor(range(output_size))
        N_1_idx = torch.topk(avg_act_1, N_1)[1]
        N_1_idx, _ = torch.sort(N_1_idx)
        N_1_idx = N_1_idx.cpu().numpy().astype(int)
        # print(N_1_idx)
        N_2_idx = torch.topk(avg_act_2, N_2)[1]
        N_2_idx, _ = torch.sort(N_2_idx)
        N_2_idx = N_2_idx.cpu().numpy().astype(int)
        # print(N_2_idx)
        N_idx = [N_i_idx, N_1_idx, N_2_idx, N_o_idx] 

        W_ = []
        B_ = []

        for k in range(len(N_) - 1):
            W_.append(np.zeros([N_[k], N_[k + 1]]))
            B_.append(np.zeros([N_[k + 1]]))    
            for i in range(N_[k]):
                for j in range(N_[k + 1]):
                    W_[k][i, j] = W[k][N_idx[k][i], N_idx[k + 1][j]]
            for j in range(N_[k + 1]):
                B_[k][j] = B[k][N_idx[k + 1][j]]


        write_to_file(input_size, 1, N_[1:-1], W_, B_, file_name = 'network_files/neural_network_information_16_' + str(rate))
        print("hehe")





def prune_model_weight(file_name = '../models/mlp_pretrained.pkl', prune_list = []):
    global glob_param
    input_size = 28 * 28
    output_size = 10
    num_layers = 2
    N = [200, 200]
    W = []
    B = []

    net = MLP()
    net.load_state_dict(torch.load(file_name))
    if torch.cuda.is_available():
        print('CUDA ensabled.')
        net.cuda()
    print("--- Pretrained network loaded ---")
    acc, avg_act_1, avg_act_2 = test(net, loader_test)

    for prune in prune_list:
        print("prune rate is {}%".format(prune))
        glob_param['pruning_perc'] = prune
        # prune the weights
        masks = weight_prune(net, glob_param['pruning_perc'])
        net.set_masks(masks)
        print("--- {}% parameters pruned ---".format(glob_param['pruning_perc']))
        test(net, loader_test)
        
        # Retraining
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=glob_param['learning_rate'], 
                                        weight_decay=glob_param['weight_decay'])
        
        train(net, criterion, optimizer, glob_param, loader_train)
        
        
        # Check accuracy and nonzeros weights in each layer
        print("--- After retraining ---")
        test(net, loader_test)
        prune_rate(net)


        for name, param in net.named_parameters():
            if param.requires_grad:
                if re.match('.*weight', name) is not None:
                    W.append(param.detach().cpu().numpy().T)
                elif re.match('.*bias', name) is not None:
                    B.append(param.detach().cpu().numpy())

        write_to_file(input_size = input_size, output_size = 1, neurons = N, W = W, B = B) 


if __name__ == "__main__":
    #generator_random()

    #rates = [1.0, 0.8, 0.5, 0.3, 0.1]
    #prune_model_neuron(rates = rates)
    prune_model_weight('../models/mlp_pretrained.pkl', prune_list = [0., 20, 50, 70, 90.])

