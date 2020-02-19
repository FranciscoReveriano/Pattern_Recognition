import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns

# Inherit Linear Data
class linearRegression(torch.nn.Module):
    def __init__(self, n_features):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=n_features, out_features=1)

    def forward(self, x):
        return self.linear(x)

# Read Data
def read_data():
    # The first part is reading the dataset. With Pandas I can treat the dataset as if it was a .csv file.
    data = pd.read_csv("/home/franciscoAML/Documents/ECE_580/Homework03_Cross_Validation_and_Norms/imports-85.data", header=None)

    # At this point the dataset has no header. It is easier for me to place a header to make it easier to choose columns
    headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
               "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight",
               "engine-type",
               "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
               "peak-rpm", "city-mpg", "highway-mpg", "price"]

    # We add the headers to the table to make it readable.
    data.columns = headers

    # We can see that there are 26 columns. This means that we need to delete them. And use only the columns we have been assigned.
    # So in other words we are dropping the columns using the panda drop command function.
    data = data.drop(columns=["symboling", "normalized-losses", "make", "fuel-type", "aspiration","num-of-doors", "body-style", "drive-wheels", "engine-location","engine-type", "num-of-cylinders", "fuel-system"])

    # We now proceed to reformat the dataset by first turning all the '?' into 'NaN'
    data["price"] = pd.to_numeric(data["price"], errors='coerce')
    data["bore"] = pd.to_numeric(data["bore"], errors='coerce')
    data["stroke"] = pd.to_numeric(data["stroke"], errors='coerce')
    data["compression-ratio"] = pd.to_numeric(data["compression-ratio"], errors='coerce')
    data["horsepower"] = pd.to_numeric(data["horsepower"], errors='coerce')
    # We then proceed to drop the NaN
    data = data.dropna(subset=["price"], axis=0)
    # data = data.dropna(subset=["bore"], axis= 0)
    data = data.dropna()

    # Make a copy of the model
    Model = data.copy()
    return Model


# Read First Model
Model_1 = read_data()

# Create the X,Y Variables in our model
X = Model_1[["curb-weight", "horsepower", "city-mpg"]]
Y = Model_1["price"]

# Turn Dataframe into Torch Tensor
X_tensor = torch.tensor(X.values)
Y_tensor = torch.tensor(Y.values).reshape(-1,1)
print("X shape:", X_tensor.shape)
print("Y shape:", Y_tensor.shape)

# Numbers
Num_Features = 3                                                                                                        # takes variable 'x'
outputDim = 1                                                                                                           # takes variable 'y'
learningRate = 0.001
epochs = 200

model = linearRegression(Num_Features)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# Proceed to loop around for requested number of epochs
for epoch in range(epochs):
    # Arrange as variables
    inputs = Variable(X_tensor.float())
    labels = Variable(Y_tensor.float())

    # Clear Gradients
    optimizer.zero_grad()

    # Run Model
    outputs = model.forward(X_tensor.float())
    #print(outputs)
    # get loss for the predicted output
    loss = criterion(outputs, labels)
    # get gradients w.r.t to parameters
    loss.backward()

    #Update Parameters
    optimizer.step()


