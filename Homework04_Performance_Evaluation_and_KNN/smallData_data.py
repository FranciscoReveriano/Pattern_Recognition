import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from ROC.ROC_Plots import *

# Read Moderate Data
moderateData_y = np.loadtxt("Datasets/smallData.csv", delimiter=",", usecols=0)
moderateData_scores = np.loadtxt("Datasets/smallData.csv", delimiter=",", usecols=1)

# Scikit Learning
fpr, tpr, thresholds = metrics.roc_curve(moderateData_y, moderateData_scores)
print("SciKit MIN:", min(thresholds))
print("SciKit MAX:", max(thresholds))
print("SciKit Length:", len(thresholds))

# Method 1
## Every Decision Statistic
beta_space = np.sort(moderateData_scores)
#print("Method 1 (MIN):", min(beta_space))
#print("Method 1 (MAX):", max(beta_space))
method_1_FPR, method_1_TPR = calculate_roc(moderateData_y, moderateData_scores, beta_space)

# Method 2
## Linear Beta
## Second Method (Linearly Spaced Samples)
min_threshold = min(moderateData_scores)
max_threshold = max(moderateData_scores)
beta_space, space = np.linspace(min_threshold, max_threshold, 99, retstep=True)
linear_FPR, linear_TPR = calculate_roc(moderateData_y, moderateData_scores, beta_space)

# Method 3
## N Sample
### Proceed To Create The Sampling
beta_space = np.sort(moderateData_scores)
rule = 1
new_beta_space = []
for i in range(0,len(beta_space),rule):
    new_beta_space.append(beta_space[i])
### Proceed to Calculate ROC
method_3_FPR, method_3_TPR = calculate_roc(moderateData_y, moderateData_scores, new_beta_space)

# Method 4
beta_space = []
for i in range(len(moderateData_y)):
    if moderateData_y[i] == 0:
        beta_space.append(moderateData_scores[i])
print(len(beta_space))
beta_space = np.sort(beta_space)
method_4_FPR, method_4_TPR = calculate_roc(moderateData_y, moderateData_scores, beta_space)


# Method 5
#beta_space = calculate_thresholds(moderateData_y,moderateData_scores)
#method_5_FPR, method_5_TPR = calculate_roc(moderateData_y, moderateData_scores, beta_space)

# Make Plot
fig = plt.figure()
plt.plot(method_1_FPR, method_1_TPR, color="blue", label="Method 1")
plt.plot(linear_FPR, linear_TPR, color="red", label="Method 2")
plt.plot(method_3_FPR, method_3_TPR, color="magenta", label="Method 3")
plt.plot(method_4_FPR, method_4_TPR, color="purple", label="Method 4")
plt.plot(method_1_FPR, method_1_TPR, color="yellow", label="Method 5")
plt.legend()
plt.xlim([0.0,1.05])
plt.ylim([0.0,1.05])
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title("smallData")
plt.savefig("smallData/Plot1", dpi=1680, precision=95)

# Make Plot With Different ROC Curves
fig2 = plt.figure()

#Scikit Method
plt.subplot(231)
plt.plot(fpr, tpr, color="black", label="SciKit Method")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('SciKit Method')
# Method 1
plt.subplot(232)
plt.plot(method_1_FPR, method_1_TPR, color="blue", label="Method 1")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('Method 1')
# Method 2
plt.subplot(233)
plt.plot(linear_FPR, linear_TPR, color="red", label="Method 2")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('Method 2')
# Method 3
plt.subplot(234)
plt.plot(method_3_FPR, method_3_TPR, color="magenta", label="Method 3")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('Method 3')
# Method 4
plt.subplot(235)
plt.plot(method_4_FPR, method_4_TPR, color="purple", label="Method 4")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('Method 4')
# Method 5
plt.subplot(236)
plt.plot(method_1_FPR, method_1_TPR, color="yellow", label="Method 5")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title('Method 5')
plt.suptitle("smallData")
plt.show()