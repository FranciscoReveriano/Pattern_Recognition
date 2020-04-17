import numpy as np
from tqdm import tqdm

def calculate_roc(moderateData_y, moderateData_scores, beta_space):
    # Calculate the
    FPR_List = []
    TPR_List = []
    for beta in beta_space:
        # Now We Need to Get The Indexes to populate New Array
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        for i in range(len(moderateData_scores)):
            # Calculate True Positives
            # Calculate True Positives
            if moderateData_y[i] == 1 and moderateData_scores[i] >= beta:
                true_positive += 1
            # Calculate False Negative
            if moderateData_y[i] == 1 and moderateData_scores[i] < beta:
                false_negative += 1
            # Calculate False Positive
            if moderateData_y[i] == 0 and moderateData_scores[i] >= beta:
                false_positive += 1
            if moderateData_y[i] == 0 and moderateData_scores[i] < beta:
                true_negative += 1
        FPR = false_positive / (false_positive + true_negative)
        TPR = true_positive / (true_positive + false_negative)
        # Append Into List
        FPR_List.append(FPR)
        TPR_List.append(TPR)
    FPR_List = np.asarray(FPR_List)
    TPR_List = np.asarray(TPR_List)
    return FPR_List, TPR_List


def calculate_num_values(matrix_1, value):
    count = 0
    for value in matrix_1:
        if value == 0:
            count += 1
    return count

def calculate_threshold(moderateData_y, moderateData_scores, PFA_Value):
    beta = -6.0
    FPR = 0
    while FPR != PFA_Value:
        beta += 0.001
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        FPR = 0
        for i in range(len(moderateData_scores)):
            # Calculate True Positives
            if moderateData_y[i] == 1 and moderateData_scores[i] >= beta:
                true_positive += 1
            # Calculate False Negative
            if moderateData_y[i] == 1 and moderateData_scores[i] <= beta:
                false_negative += 1
            # Calculate False Positive
            if moderateData_y[i] == 0 and moderateData_scores[i] >= beta:
                false_positive += 1
            if moderateData_y[i] == 0 and moderateData_scores[i] <= beta:
                true_negative += 1
        FPR = false_positive / (false_positive + true_negative)
        FPR = round(FPR,2)
        #print(FPR, beta, false_positive)
    return beta

def calculate_num_observations(PF_Value, total_observations):
        count = 0
        PF_Count = 0
        while round(PF_Value,2) != round(PF_Count,2):
            count += 1
            PF_Count = count / total_observations
        return count

def threshold_calculation(count,moderateData_scores):
    sort_scores = np.sort(moderateData_scores)
    threshold = sort_scores[count]
    return threshold

def calculate_thresholds(moderateData_y, moderateData_scores):
    Total_Negatives = calculate_num_values(moderateData_y, 0)
    Total_Positive = calculate_num_values(moderateData_y, 1)
    FPR_Values = np.arange(0,1,0.01)
    Threshold_List = []
    for value in FPR_Values:
        count = calculate_num_observations(value, Total_Negatives)
        threshold = threshold_calculation(count, moderateData_scores )
        Threshold_List.append(threshold)
    return Threshold_List