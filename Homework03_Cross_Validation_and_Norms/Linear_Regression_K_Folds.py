import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

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

# We first set a new dataframe
Model = read_data()

# Create the X Variables in our model
X = Model[["curb-weight", "horsepower", "city-mpg"]]

# Create the Y Variable in our model
Y = Model["price"]

# Set Up The K-Folds
K = 10
kf = KFold(n_splits=10, shuffle=True)

MSE_List = []
for i in range(3):
    print("Test ",i)
    for train_index, test_index in kf.split(X):
        # Split the Dataset
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        # Turn values into tensors
        X_train_tensor = torch.tensor(x_train.values)
        X_test_tensor = torch.tensor(x_test.values)
        Y_train_tensor = torch.tensor(y_train.values).reshape(-1, 1)
        Y_test_tensor = torch.tensor(y_test.values).reshape(-1,1)

        # Set up the Model
        Num_Features = 3  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.001
        epochs = 400

        model = linearRegression(Num_Features)
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

        # Run through the model
        for j in range(epochs):
            # Clear Gradients
            optimizer.zero_grad()

            # Run Model
            outputs = model.forward(X_train_tensor.float())
            # Get Lost for Predicted Output
            loss = criterion(outputs, Y_train_tensor.float())
            # Backward Propogation
            loss.backward()
            # Update Parameters
            optimizer.step()

        # Test Prediction of the Model
        optimizer.zero_grad()
        y_hat = model(X_test_tensor.float())

        # Calculate MSE
        ## Convert to Numpy
        y_hat = y_hat.detach().numpy()
        Y_test_tensor = Y_test_tensor.numpy()
        print(y_hat[0], Y_test_tensor[0])

        ## Now I can calculate the MSE using Scikit Learning
        MSE = mean_squared_error(Y_test_tensor, y_hat)
        MSE_List.append(MSE)

print(MSE_List)