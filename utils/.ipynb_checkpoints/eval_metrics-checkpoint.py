import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix
import torch 
import pandas as pd


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
    return jaccard_index(conf_mat).numpy()



# +
#6 evaluation matrices to be used: sensitivity, specificity, accuracy, AUC, DC and IOU

def evaluate_batch(gnd_b, pred_b, cls:int = 19):
    """
        Calculate evalution scores over the batch.
    Args:   
        gnd_b: BxHxW tensor; ground truth labels; each element of matrix in B dim contains class label
        pred_b: BxCxHxW tensor; each element contains predicted class label 
                here C=19 (0-18; no. of classes); each C corresponds to probabilites for that class,
                eg. C=0 contain score at each element in matrix HxW 
    Return:
        sensitivity, specificity, accuracy, aauc_score, dice_coeeficient, IOU (Intersection over union) (averaged over batch size)
    """
    # to cpu and as numpy ndarray
    gnd_b = gnd_b.cpu()

    batch_size = gnd_b.shape[0]
    
    # extract most probable class through C-dim 
    label_b = torch.argmax(pred_b, dim=1).cpu()
    sensitivity = specificity = accuracy = auc = dice = iou = 0
    # iterate over batch elements
    for i in range(batch_size):
        gnd = gnd_b[i,:,:] 
        label = label_b[i,:,:]
        temp = accuracy_se_sp_custom(gnd, label)
        accuracy += temp[0]
        sensitivity += temp[1]
        specificity += temp[2]
        auc += roc_auc_custom(gnd.numpy(), label.numpy(), cls, average='macro')
        dice += dice_coefficient_custom(gnd.numpy(), label.numpy(), cls)
        iou += iou_custom(gnd.int(), label.int(), cls)

    return [sensitivity/batch_size, specificity/batch_size, accuracy/batch_size, auc/batch_size, dice/batch_size, iou/batch_size]


# -

def evaluation_loop(model, val_loader, epochs:int, device, task:int, cls_num:int=19):
    """
        Perform evalution on validation data for particular number of epochs.
        The evaluation metrics scores are stored for each epoch and also printed
        At the end, store all the scores in .csv for convenient analysis
    Args:
        model: neural network model to be train
        val_loader: data loader for validation set
        epochs: number of epochs(times) train the model over complete train data set
        device: device to which tensors will be allocated (in our case, from gpu 0 to 7)
        task: define for which part loop is performed and save the model and results in path for that task
        cls_num: number of classes in dataset, parameter for 'evaluate_batch' function
    Names of evaluation metrices:
        Sensitivity, Specificity, Accuracy, ROC AUC, Dice coefficient, IOU 
    """

    # list of score 
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    auc_list = []
    dice_list = []
    iou_list = []

    # For print
    print('Epochs\t Sensitivity-score Specificity-score Accuracy-score ROC-AUC-score\t Dice score\t IOU score')

    # move model to gpu
    model = model.to(device)
    # loop for original number of epochs
    for i in range(epochs):
        # load the model states
        model.load_state_dict(torch.load(f'../weights/T{task}/epoch_{i}.pth'))
        # model in evaluation model -> batchnorm, dropout etc. adjusted accordingly
        model.eval()
        # evaluation score variables to store values over each epoch
        sensitivity_score = specificity_score = accuracy_score = auc_score = dice_score = iou_score = 0


        for sample in val_loader:
            img, label = sample['image'].to(device), sample['label'].to(device)
            # deactivate autograd engine - reduce memory usage 
            with torch.no_grad(): 
                pred = model(img) # forward pass
                # evaluation
                scores = evaluate_batch(label, pred, cls=cls_num)
                # sum values
                sensitivity_score += scores[0]
                specificity_score += scores[1]
                accuracy_score += scores[2]
                auc_score += scores[3]
                dice_score += scores[4]
                iou_score += scores[5]

        print('{}\t {:.3f}\t\t\t {:.3f}\t\t {:.3f}\t\t {:.3f}\t\t {:.3f}\t\t {:.3f}'.format(i ,sensitivity_score/len(val_loader), specificity_score/len(val_loader), accuracy_score/len(val_loader), auc_score/len(val_loader), dice_score/len(val_loader), iou_score/len(val_loader)))
        # append to list (with averaged values over valid set)
        sensitivity_list.append(sensitivity_score/len(val_loader))
        specificity_list.append(specificity_score/len(val_loader))
        accuracy_list.append(accuracy_score/len(val_loader))
        auc_list.append(auc_score/len(val_loader))
        dice_list.append(dice_score/len(val_loader))
        iou_list.append(iou_score/len(val_loader))
        
    # create dictionary of score list
    eval_dict = {'sensitivity': sensitivity_list,
                'specificity': specificity_list,
                'accuracy': accuracy_list,
                'auc': auc_list,
                'dice': dice_list,
                'iou': iou_list}
    
    df_w = pd.DataFrame(eval_dict) # convert to panda's dataframe
    df_w.to_csv(f'results/T{task}_eval.csv', index=False) # save as csv

