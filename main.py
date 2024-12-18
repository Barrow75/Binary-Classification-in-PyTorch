import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

# Neural Network Classification

# 1. Make classification data

#   Make 1000 samples
n_samples = 1000

# create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(f"First 5 samples of a X:\n {X[:5]}")
print(f"First 5 samples of a y:\n {y[:5]}")

# Make a dataframe of circle data and visualize
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles)

# how many red nd blue dots
print(circles.label.value_counts())

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

# view first example of features and labels

X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shape for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# these are considered toy datasets - small eough to experiment but still sizeable enough to practice with

# TURN THE DATA INTO TENSORS AND CREATE A TRAIN AND TEST SPLIT

# data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print("print", X, X[:5])
print(y[:5])

print('\n')
# splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 20% of data will be tested
                                                    random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
print("This is used for the in_features", X_train.shape)
print("This is used for the out_features", y[:5])


# Build a model toclasify blue and red dots
# 1. Construct model
# 2. Define lost function and optimizer
# 3. Create a training and test loop

# subclasses nn.module
# create 2 nn.Linear layers that are capable of handling shape of data
# define a forward method that outlines the forward pass
# instantiate an instance of our model class and send it to a target device

# 1. Constrct a modle that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. create 2 nn.Linear layers
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features; upscale it to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  # takes in 5 features from previous layer

    # 3. forward methods that outline a forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # layer_1 -> layer_2 -> output


# 4. Instatiate an instance of our model class
model0 = CircleModelV0()
print(model0)
print('\n')
# replicate model using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)
print(model_0)
print('this is what the model looks like on the inside', model_0.state_dict())

# make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\n First 10 predictions: \n {torch.round(untrained_preds[:10])}")
print(f"\n First 10 labels: {y_test[:10]}")

print("\n")
# set up loss function and optimizer
#   which loss function or optimizer to use? - Problem specific

loss_fn = nn.BCEWithLogitsLoss()  # sigmoid activation function built in
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


# calculate accuracy - out of 100 examples, what percentage does model get right
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 3 train model
# 1. forward pass
# 2. calculate the loss
# 3 Optimizer zero grad
# 4. loss backward
# 5. optimizer step

# going from raw logits -> prediction probabilites -> prediction labels
# logits - rawoutputs of the model without being passed to activation function

# view first 5 outputs of forward pass
y_logits = model_0(X_test)[:5]
print(y_logits)

# use sigmoid activation function on our model logits to turn them into prediction probabilties
#  prediction probabilties - 0 or 1 for how likly model thinks it is a certain class
y_pred_prob = torch.sigmoid(y_logits)
print(y_pred_prob)

# Find the predicted labels
y_preds = torch.round(y_pred_prob)

# in full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)[:5]))

# check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
print(y_preds.squeeze())

# build a training and test loop
torch.manual_seed(42)

# number of epochs
epochs = 100

# build training and evfaluation loop
for epoch in range(epochs):
    model_0.train()

    # forward pass
    y_logits = model_0(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labels

    # calculate loss/ accuracy
    loss = loss_fn(y_logits,
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_preds)

    # optimizer zero grad
    optimizer.zero_grad()

    # back propgation
    loss.backward()

    # optimizer step (gradient descnet)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # calculate test loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        # print out what's happening
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss} | Test acc: {test_acc: .2f}%")

# Make predictiobs and evaluate model

# model is not learning so inspect by making predictions and visualizing
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_function.py already exists, skiping download")
else:
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()