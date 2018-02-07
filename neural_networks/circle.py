#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict


NUM_BATCH = 500
BATCH_SIZE = 256
PRINT_INTERVAL = 20

# A simple framework to work with pytorch
class Classifier(nn.Module):
    def __init__(self, size_in):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(size_in, 500) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 1)  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.sigmoid(out)

# may be of use to you
# returns the percentage of predictions (greater than threshold) 
# that are equal to the labels provided
def percentage_correct(pred, labels, threshold = 0.5):
    correct = 0.0

    for i in range(0, BATCH_SIZE):
        if (pred.data.numpy()[i] == labels.data.numpy()[i]):
            correct+=1

    return (correct/BATCH_SIZE) * 100;

# This code generates 2D data with label 1 if the point lies
# outside the unit circle.
def get_batch(batch_size):
    # Data has two dimensions, they are randomly generated
    data = (torch.rand(batch_size,2)-0.5)*2.5
    # square them and sum them to define the decision boundary
    # (x_1)^2 + (x_2)^2 = 1
    square = torch.mul(data,data)
    square_sum = torch.sum(square,1,keepdim=True)
    # Generate the labels
    # outside the circle is 1
    labels = square_sum>1
    return Variable(data), Variable(labels.float())

def plot_decision_boundary(data_in, preds):
    dic = defaultdict(lambda: "r")
    dic[0] = 'b'
    colour = list(map(lambda x: dic[x[0]], preds.data.numpy()>0.5))
    x = data_in.data.numpy()[:,0]
    y = data_in.data.numpy()[:,1]
    plt.clf()
    plt.scatter(x,y,c=colour)
    plt.title("Decision Boundary of a Neural Net Trained to Classify the Unit Circle")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)

def plot_percent_correct(correct, iterations):
    plt.plot(correct, iterations, 'bo')
    plt.title("Percent Correct VS Iteration Number")
    plt.show()



# Here's the spot where you'll do your batches of optimization
data, labels = get_batch(BATCH_SIZE)
model = Classifier(data.size(1))
model.train()

o = torch.optim.SGD(model.parameters(), lr = 0.001)
loss = nn.BCELoss()

iterations = []
correct = []

for i in range(NUM_BATCH):
    iterations.append(i)
    data, labels = get_batch(BATCH_SIZE)
    pred = model(data)
    correct.append(percentage_correct(pred, labels))
    error = loss(pred, labels)
    error.backward()
    o.step()

# plot decision boundary for new data
d, labels = get_batch(BATCH_SIZE)
plot_decision_boundary(d, model(d))
#plot_percent_correct(correct, iterations)




