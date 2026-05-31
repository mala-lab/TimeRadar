import numpy as np
import sklearn
import sklearn.preprocessing

def adjustment(gt, pred):
    adjusted_pred = np.array(pred)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = 1
    return adjusted_pred


import pandas as pd
def evaluate(results_storage, metrics, labels, score, **args):
    if "best_f1" in metrics:
        result = {}
        Ps, Rs, thres = sklearn.metrics.precision_recall_curve(labels, score)
        F1s = (2 * Ps * Rs) / (Ps + Rs)
        best_F1_index = np.argmax(F1s[np.isfinite(F1s)])
        best_thre = thres[best_F1_index]
        pred = (score > best_thre).astype(int)
        best_acc = sklearn.metrics.accuracy_score(labels, pred)
        result['thre_best'] = best_thre
        result['ACC_best'] = best_acc
        result['P_best'] = Ps[best_F1_index] 
        result['R_best'] = Rs[best_F1_index] 
        result['F1_best'] = F1s[best_F1_index]
        results_storage['best_f1'] = pd.DataFrame([result])
    if "f1_raw" in metrics:
        results = []
        for thre in args['f1_raw']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, pred, average="binary")
            result['thre_raw'] = thre
            result['ACC_raw'] = accuracy
            result['P_raw'] = P 
            result['R_raw'] = R 
            result['F1_raw'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_raw'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_pa" in metrics:
        results = []
        for thre in args['f1_pa']:
            result = {}
            pred = (score > thre).astype(int)
            adjusted_pred = adjustment(labels, pred)
            accuracy = sklearn.metrics.accuracy_score(labels, adjusted_pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, adjusted_pred, average="binary")
            result['thre_PA'] = thre
            result['ACC_PA'] = accuracy
            result['P_PA'] = P 
            result['R_PA'] = R 
            result['F1_PA'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_pa'] = pd.concat(results, axis=0).reset_index(drop=True)
