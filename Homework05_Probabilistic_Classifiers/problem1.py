import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Read The File
path = "/home/franciscoAML/Documents/ECE_580/Homework05_Probabilistic_Classifiers/Datasets/dataSetCrossValWithKeys.csv"
df = pd.read_csv(path, names=["Fold", "Class", "x", 'y'])

# Proceed To Recreate the Folds
Fold1 = df.loc[df["Fold"] == 1]
Fold2 = df.loc[df["Fold"] == 2]

# Proceed to Drop the Fold Labels
Fold1 = Fold1.drop(["Fold"], axis=1)
#Fold1.to_csv("Fold1.csv",index=False)
Fold2 = Fold2.drop(["Fold"], axis=1)
#Fold2.to_csv("Fold2.csv", index=False)

# Proceed To Test Cross Validation 1
x1, y1 = Fold1[["x","y"]], Fold1["Class"]
x2, y2 = Fold2[["x","y"]], Fold2["Class"]

# Train the Classifier First Fold
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x1,y1)
y2_hat = clf.predict(x2)
y2_prob = clf.predict_proba(x2)
fpr, tpr, thresholds = metrics.roc_curve(y2, y2_prob[:,1])

# Train the Classifier Second Fold
clf2 = KNeighborsClassifier(n_neighbors=5)
clf2.fit(x2,y2)                                                                                                         # Train Classifier
y1_hat = clf2.predict_proba(x1)                                                                                         # Predict the values on First Fold
y1_prob = clf2.predict_proba(x1)                                                                                        # Predict Probabilities
fpr2, tpr2, thresholds2 = metrics.roc_curve(y1, y1_prob[:,1])

# Plot Differences
plt.figure()
plt.plot(fpr,tpr, c="red", label="1st Fold")
plt.plot(fpr2,tpr2, c="blue", label="2nd Fold")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title("1st & 2nd Fold Testing")
plt.legend()
plt.savefig("/home/franciscoAML/Documents/ECE_580/Homework05_Probabilistic_Classifiers/Problem_1/1st_2nd_Fold")

# Receive the ROC Curve
mean_fpr = np.mean(np.array([fpr,fpr2]), axis=0)
mean_tpr = np.mean(np.array([tpr, tpr2]), axis=0)
plt.figure()
plt.plot(mean_fpr, mean_tpr, color="red")
plt.ylabel("$P_D$")
plt.xlabel("$P_{FA}$")
plt.title("Average ROC Curve")
plt.savefig("/home/franciscoAML/Documents/ECE_580/Homework05_Probabilistic_Classifiers/Problem_1/Average_ROC_Curve")
