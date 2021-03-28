# +
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix

import torch 


# -

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

def accuracy_se_sp_custom(ground_truth,prediction):
    ground_truth = ground_truth == torch.max(ground_truth)
    #corr = torch.sum(prediction == ground_truth)
    #tensor_size = prediction.size(0) * prediction.size(1) * prediction.size(2) * prediction.size(3)
    #acc = float(corr) / float(tensor_size)
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    mcm = confusion_matrix(ground_truth, prediction)
    tn = mcm[1,1]
    tp = mcm[0,0]
    fn = mcm[1,0]
    fp = mcm[0,1]
    acc = (tp + tn)/(tp+tn+fp+fn + 1e-6)
    SE = tp / (tp + fn + 1e-6)
    SP = tn / (tn + fp + 1e-6)

    #acc = float(torch.sum(TP+TN)) / (float(torch.sum(TP + TN+FP+FN)) + 1e-6)
    return [acc, SE, SP]


def sensitivity_custom(ground_truth, prediction):
    # Sensitivity == Recall
    ground_truth = ground_truth == torch.max(ground_truth)

    # TP : True Positive
    # FN : False Negative
    #TP = ((prediction == 1) + (ground_truth == 1)) == 2
    #FN = ((prediction == 0) + (ground_truth == 1)) == 2

    #SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    mcm = confusion_matrix(ground_truth, prediction)
    tn = mcm[1,1]
    tp = mcm[0,0]
    fn = mcm[1,0]
    fp = mcm[0,1]
    SE = tp / (tp + fn + 1e-6)
    return SE


def specificity_custom(ground_truth,prediction):
    ground_truth = ground_truth == torch.max(ground_truth)

    # TN : True Negative
    # FP : False Positive
    #TN = ((prediction == 0) + (ground_truth == 0)) == 2
    #FP = ((prediction == 1) + (ground_truth == 0)) == 2

    #SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    mcm = confusion_matrix(ground_truth, prediction)
    tn = mcm[1,1]
    tp = mcm[0,0]
    fn = mcm[1,0]
    fp = mcm[0,1]
    SP = tn / (tn + fp + 1e-6)
    return SP


# +
def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + 1e-6)
    avg_jacc = nanmean(jaccard)
    return avg_jacc

def iou_custom(ground_truth, prediction, n_classes:int=19):
    conf_mat = _fast_hist(ground_truth, prediction, n_classes)
    return jaccard_index(conf_mat)


# def iou_custom(ground_truth, prediction, n_classes:int=19):
#     """
#         Return mean IOU score.
#     Args:   
#         ground_truth: HxW ndarray; each element contains class label from (0-20)
#         prediction: HxW ndarray; each element contains predicted class label
#         n_class: total number of classes; used to average score
#     """
#     # init, sum over all individual class scores
#     score = 0.
#     # iterate for each class
#     for c in range(n_classes):
#         # create 2D matrix (mask) (binary elements 0-1) for ground truth and prediction matrix
#         gnd = torch.zeros_like(ground_truth)
#         pred = torch.zeros_like(prediction)
#         gnd[ground_truth == c] = 1
#         pred[prediction == c] = 1
    
#         # compute score
#         # gnd = gnd.astype(np.bool)
#         # pred = pred.astype(np.bool)
#         conf_mat = confusionmatrix_jaccard(gnd, pred, n_classes)
#         score += iou_jaccard_binary(conf_mat)

#     # average n=21
#     # return
#     return score/n_classes



