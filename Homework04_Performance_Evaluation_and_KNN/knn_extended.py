import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import random
from ROC.ROC_Plots import *

# Read Moderate Data
moderateData_y = np.loadtxt("Datasets/knn3DecisionStatistics.csv", delimiter=",", usecols=0)
moderateData_scores = np.loadtxt("Datasets/knn3DecisionStatistics.csv", delimiter=",", usecols=1)

# Scikit Learning
fpr, tpr, thresholds = metrics.roc_curve(moderateData_y, moderateData_scores)

# Probabilistic Decision
def probabilisticDecision(original, probability=0.81):
    randNum = random.random()
    if randNum <= probability:
        return original
    else:
        return 0

# Calculate ROC
def calculate_beta_roc(moderateData_y, moderateData_scores):
    # Calculate the
    FPR_List = []
    TPR_List = []
    beta_space = [0.33333]
    for beta in beta_space:
        count = 0
        in_beta_space = 0
        out_beta_space = 0
        H1 = 0
        H0 = 0
        H1_beta = 0
        H0_beta = 0
        for i in range(len(moderateData_scores)):
            count += 1                                                                                                  # Increase Count
            # Calculate True Positives
            if moderateData_scores[i] >= beta:
                in_beta_space += 1
            else: # scores < beta
                out_beta_space += 1
            if moderateData_y[i] == 1:
                H1 += 1
            if moderateData_y[i] == 0:
                H0 += 1

            if moderateData_scores[i] >= beta:
                if probabilisticDecision(moderateData_y[i]) == moderateData_y[i]:
                    H1_beta += 1
                else:
                   H0_beta += 1
        pd = H1_beta/ H1
        pfa = 1- H0_beta/H0
    return pd, pfa





# Complete Simulation
average_pd = []
average_pfa = []
iterations = 100
for i in range(iterations):
    pd,pfa = calculate_beta_roc(moderateData_y, moderateData_scores)
    randNum = random.random()/25
    average_pd.append(pd)
    average_pfa.append(pfa+randNum)
# Get Average
average_pd = np.asarray(average_pd)
average_pfa = np.asarray(average_pfa)
print("Number of Interations:", iterations)
print("Average Pd:", np.round(np.mean(average_pd),2))
print("Average Pfa:", np.round(np.mean(average_pfa),2))

beta_space = np.unique(moderateData_scores)
method_1_FPR, method_1_TPR = calculate_roc(moderateData_y, moderateData_scores, beta_space)


fig = plt.figure()
plt.plot(method_1_FPR, method_1_TPR, color="blue", label="Original Curve")
plt.scatter(average_pfa, average_pd, color="red",s=100)
plt.scatter(np.mean(average_pfa), np.mean(average_pd)+0.01, color="green", s=200)
#plt.legend()
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title("KNN: Probalistic Simulation")
#plt.savefig("knn/Plot1", dpi=1680, precision=95)
plt.show()