import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def score_model(preds, labels, use_zero=False):   #0是否为正类
    mae = np.mean(np.absolute(preds - labels))
    corr = np.corrcoef(preds, labels)[0][1] #计算矩阵相关系数
    non_zeros = np.array(
        [i for i, e in enumerate(labels) if e != 0 or use_zero]) #获取labels中非0元素的索引，use zero=true时获取所有元素的索引
    preds = preds[non_zeros]
    labels = labels[non_zeros]
    preds = preds >= 0
    labels = labels >= 0
    f_score = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return acc, mae, corr, f_score

def score_model_3(preds, preds1, preds2):
    preds = preds >= 0
    preds1 = preds1 >= 0
    preds2 = preds2 >= 0
    lenth = len(preds)
    stacked_preds = np.stack([preds1, preds2, preds])
    true_counts = np.sum(stacked_preds == True, axis=0)
    result = true_counts >= 2
    return result
