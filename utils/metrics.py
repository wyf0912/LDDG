import numpy as np
from functools import wraps
import torch
from sklearn import metrics

def typeCheck(func):
    @wraps(func)
    def checked(pm, gt):
        if isinstance(pm, torch.Tensor):
            pm = pm.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        pm = pm.squeeze().astype('uint8')
        gt = gt.squeeze().astype('uint8')
        return func(pm, gt)

    return checked


@typeCheck
def DiceSimilarityCoefficient(pm, gt):
    """
    DSC
    :param pm: provided mask
    :param gt: ground truth
    :return: dice value
    """
    a = 2 * np.sum(np.bitwise_and(pm, gt))
    b = np.sum(pm) + np.sum(gt)
    return a / b


@typeCheck
def JaccardIndex(pm, gt):
    """
    JI
    :param pm: provided mask
    :param gt: ground truth
    :return: JI value
    """
    a = np.sum(np.bitwise_and(pm, gt))
    b = np.sum(gt) + np.sum(pm) - np.sum(np.bitwise_and(gt, pm))
    return a / b


@typeCheck
def ConformityCoefficient(pm, gt):
    """
    CC: measures the ratio between missegmented voxels and correctly segmented voxels
    :param pm: provided mask
    :param gt: ground truth
    :return: CC value
    """
    FP = np.sum(np.bitwise_and(np.bitwise_not(gt), pm))
    FN = np.sum(np.bitwise_and(gt, np.bitwise_not(pm)))
    TP = np.sum(np.bitwise_and(pm, gt))
    if TP != 0:
        return (1 - (FP + FN) / TP) * 100
    else:
        return 0  # None


@typeCheck
def Sensitivity(pm, gt):
    """
    Sensitivity or True Positive Rate (TPR): represents a methods ability to segment GM as a proportion of all correctly labelled voxels.
    :param pm: provided mask
    :param gt: ground truth
    :return: TPR value
    """
    FP = np.sum(np.bitwise_and(np.bitwise_not(gt), pm))
    FN = np.sum(np.bitwise_and(gt, np.bitwise_not(pm)))
    TP = np.sum(np.bitwise_and(pm, gt))
    return 100 * (TP / (TP + FN))


@typeCheck
def Speciﬁcity(pm, gt):
    """
    True Negative Rate (TNR): measures the proportion of correctly segmented background (non-GM) voxels
    :param pm: provided mask
    :param gt: ground truth
    :return: TNR value
    """
    FP = np.sum(np.bitwise_and(np.bitwise_not(gt), pm))
    FN = np.sum(np.bitwise_and(gt, np.bitwise_not(pm)))
    TP = np.sum(np.bitwise_and(pm, gt))
    TN = np.sum(np.bitwise_and(np.bitwise_not(pm), np.bitwise_not(gt)))
    return 100 * (TN / (TN + FP))


@typeCheck
def Precision(pm, gt):
    """
     Positive Predictive Value, (PPV): measures the degree of compromise between true and false positive.
    :param pm: provided mask
    :param gt: ground truth
    :return: PPV value
    """
    FP = np.sum(np.bitwise_and(np.bitwise_not(gt), pm))
    FN = np.sum(np.bitwise_and(gt, np.bitwise_not(pm)))
    TP = np.sum(np.bitwise_and(pm, gt))
    TN = np.sum(np.bitwise_and(np.bitwise_not(pm), np.bitwise_not(gt)))
    return 100 * (TP / (TP + FP))  # if TP + FP != 0 else 50


if __name__ == '__main__':
    # print(test(1, 2))
    pass
