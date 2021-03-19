import numpy as np
from sklearn.metrics import roc_auc_score
import torch 

def dice_coefficient_custom(ground_truth, prediction, n_classes:int =21, smooth:float = 0.0001):
    """
        Return dice coefficient score.
    Args:   
        ground_truth: HxW ndarray; each element contains class label from (0-20)
        prediction: HxW ndarray; each element contains predicted class label
        n_class: total number of classes; used to average score
        smooth: to avoid divide by zero problem
    """
    # init, sum over all individual class scores
    score = 0.
    # iterate for each class (0-20)
    for c in range(n_classes):
        # create 2D matrix (mask) (binary elements 0-1) for ground truth and prediction matrix
        gnd = np.zeros_like(ground_truth)
        pred = np.zeros_like(prediction)
        gnd[ground_truth == c] = 1
        pred[prediction == c] = 1
    
        # compute score
        # gnd = gnd.astype(np.bool)
        # pred = pred.astype(np.bool)
        gnd_f = gnd.flatten() # flatten to 1D
        pred_f = pred.flatten()
        intersection = (gnd_f * pred_f).sum()
        score += ((2.* intersection + smooth)/(gnd_f.sum() + pred_f.sum() + smooth))

    # average n=21
    # return
    return score/n_classes


def roc_auc_custom(ground_truth, prediction, n_classes:int =21, average:str = 'macro'):
    """
        Return roc_auc score (average for all class for each image).
    Args:   
        ground_truth: HxW ndarray; each element contains class label from (0-20)
        prediction: HxW ndarray; each element contains predicted class label
        n_class: total number of classes; used to average score
        average: type of averaging, passed to sklearn's roc_auc_score() function 
    """
    # init, sum over all individual class scores
    score = 0.
    cnt = 0
    # iterate for each class (0-20)
    for c in range(n_classes):
        # create 2D matrix (mask) (binary elements 0-1) for ground truth and prediction matrix
        gnd = np.zeros_like(ground_truth)
        pred = np.zeros_like(prediction)
        gnd[ground_truth == c] = 1
        pred[prediction == c] = 1
    
        # compute score
        gnd_f = gnd.flatten() # flatten to 1D
        pred_f = pred.flatten()
        # roc_auc need atleast one element of each class, otherwise the score is not defineds
        # due to class imbalance it can't be possible always, thus we are avoiding those instances
        try:
            score += roc_auc_score(gnd_f, pred_f, average=average) # binary 
            cnt += 1 # accurate averaging
        except ValueError: 
            pass

    # average n=21
    # return
    if cnt == 0: # no score
        return score
    else: # average
        return score/cnt

def accuracy_custom(ground_truth,prediction):
    ground_truth = ground_truth == torch.max(ground_truth)
    corr = torch.sum(prediction == ground_truth)
    tensor_size = prediction.size(0) * prediction.size(1) * prediction.size(2) * prediction.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def sensitivity_custom(ground_truth, prediction):
    # Sensitivity == Recall
    ground_truth = ground_truth == torch.max(ground_truth)

    # TP : True Positive
    # FN : False Negative
    TP = ((prediction == 1) + (ground_truth == 1)) == 2
    FN = ((prediction == 0) + (ground_truth == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def specificity_custom(ground_truth,prediction):
    ground_truth = ground_truth == torch.max(ground_truth)

    # TN : True Negative
    # FP : False Positive
    TN = ((prediction == 0) + (ground_truth == 0)) == 2
    FP = ((prediction == 1) + (ground_truth == 0)) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP
