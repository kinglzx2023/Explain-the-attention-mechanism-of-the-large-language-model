#!/usr/bin/env python
# coding: utf-8



import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
torch.cuda.set_device(0)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine


input_size = 784
batch_size = 512
num_epochs = 100
learning_rate = 0.001
hidden_size = 500
number_H =3
probability = 0

address ='/home/'
address_1 = address+'cos_sim.txt'
address_2 = address+'acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file2.writelines('input_size:'+str(input_size)+'\n'+ \
                 'batch_size:'+ str(batch_size)+'\n'+\
                 'learning_rate:'+ str(learning_rate)+ '\n'+\
                 'hidden_size:' + str(hidden_size) + '\n'+\
                 'number_H:' + str(number_H) + '\n'+\
                 'dropout_probability:' + str(probability) + '\n' )
                 
def seed_torch(seed=42):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()

def cos_similarity_matrix(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out

train_datasets = dsets.MNIST(root = './Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = './Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)

class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.dropout = nn.Dropout(probability)
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        #x = self.dropout(x)
        x = self.linear(x)
        x = self.r(x)
        for i in  range(number_H):
            #x = self.dropout(x)
            x = self.linearH[i](x)
            x = self.r(x)
        out = self.out(x)
        return out

if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

file1.writelines('Initial parameters'+'\n')
for name, param in model.named_parameters():
    if name == 'linear.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_init_in.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.0.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_init_H_1.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.1.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_init_H_2.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.2.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_init_H_3.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'out.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        #features = features.to(device)
        features = Variable(features.view(-1, 28*28)).cuda()
        
        #targets = targets.to(device)
        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs= model(images)
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #weight_distribution = weights_distribution(model.parameters())
        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            #text_accuracy = round(compute_accuracy(model, test_loader).item(),2)
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    print(str(text_accuracy)+'  '+str(loss_out))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'\n')
    
file1.writelines('Trained parameters'+'\n')
for name, param in model.named_parameters():
    if name == 'linear.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_end_in.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.0.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_end_H_1.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.1.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_end_H_2.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'linearH.2.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        np.savetxt(address+'batch_size_cos_matrix_end_H_3.txt', cos_sim, fmt='%.2f')
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    if name == 'out.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim = cos_similarity_matrix(param.cpu().data)
        print(Mean(cos_sim))
        file1.writelines(str(Mean(cos_sim))+'\n')
        print('='*50)
    
file1.close() 
file2.close()



