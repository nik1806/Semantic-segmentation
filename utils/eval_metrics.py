import numpy as np

def dice_coefficient(ground_truth, prediction, n_classes:int =21, smooth:float = 0.0001):
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
    