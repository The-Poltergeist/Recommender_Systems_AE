# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:05:09 2022

@author: enoch
"""
#Stackd Autoencoder
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#%%
#reading files
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#%%
#preparing the data set
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set = np.array(training_set,dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set = np.array(test_set,dtype = 'int')

#%%
# no of users and movies
n_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
n_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#%%
#creating the matrix
def convert(data):
    new_dat = []
    for i in range(1,n_users+1):
        id_movies = data[:,1][data[:,0] == i]
        id_ratings = data[:,2][data[:,0] == i]
        ratings = np.zeros(n_movies)
        ratings[id_ratings-1] = id_ratings
        new_dat.append(list(ratings))
    return new_dat

training_set = convert(training_set)
test_set = convert(test_set)

#%%
#converting into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#%%
#Creating architecture of autoencoder
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(n_movies,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,n_movies)
        self.activation = nn.Sigmoid()
        #encoding and decoding
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay=0.5)

#%%
#training data
epochs = 200
for epoch in range(1,epochs+1):
    train_loss = 0
    count = 0.
    for id_user in range(n_users):
        input = Variable(training_set[id_user]).unsqueeze(0) #making it two dimensional
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False #only input 
            output[target == 0] = 0
            loss = criterion(output,target)
            mean_corrector = n_movies/float(torch.sum(target.data > 0)+1e-10)
            loss.backward() #direction of updating weights
            train_loss += np.sqrt(loss.data*mean_corrector)
            count += 1.
            optimizer.step()
    print('Epoch: '+str(epoch)+' Loss: '+str(train_loss/count))

#%%
test_loss = 0
count = 0.
for id_user in range(n_users):
    input = Variable(training_set[id_user]).unsqueeze(0) #making it two dimensional
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False #only input 
        output[target == 0] = 0
        loss = criterion(output,target)
        mean_corrector = n_movies/float(torch.sum(target.data > 0)+1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        count += 1.
print('Loss: '+str(test_loss/count))