#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn
from .generics import (
    infer_Trange,
    has_point_anomalies,
    _len_wo_nan,
    _sum_wo_nan,
    read_all_as_events,
    convert_vector_to_events)
from ._affiliation_zone import (
    get_all_E_gt_func,
    affiliation_partition)
from ._single_ground_truth_event import (
    affiliation_precision_distance,
    affiliation_recall_distance,
    affiliation_precision_proba,
    affiliation_recall_proba)


def test_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError('Input `events` should be a list of couples')
    if not all([type(x) is tuple for x in events]):
        raise TypeError('Input `events` should be a list of tuples')
    if not all([len(x) == 2 for x in events]):
        raise ValueError(
            'Input `events` should be a list of couples (start, stop)')
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError(
            'Input `events` should be a list of couples (start, stop) with start <= stop')
    if not all([events[i][1] < events[i+1][0] for i in range(len(events) - 1)]):
        raise ValueError(
            'Couples of input `events` should be disjoint and ordered')


def pr_from_events(events_pred, events_gt, Trange):
    """
    Compute the affiliation metrics including the precision/recall in [0,1],
    along with the individual precision/recall distances and probabilities

    :param events_pred: list of predicted events, each represented by a couple
    indicating the start and the stop of the event
    :param events_gt: list of ground truth events, each represented by a couple
    indicating the start and the stop of the event
    :param Trange: range of the series where events_pred and events_gt are included,
    represented as a couple (start, stop)
    :return: dictionary with precision, recall, and the individual metrics
    """
    # testing the inputs
    test_events(events_pred)
    test_events(events_gt)

    # other tests
    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')

    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    if Trange is None:
        # Set as default, but Trange should be indicated if probabilities are used
        raise ValueError(
            'Trange should be indicated (or inferred with the `infer_Trange` function')

    E_gt = get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_partition(events_pred, E_gt)

    # Computing precision distance
    d_precision = [affiliation_precision_distance(
        Is, J) for Is, J in zip(aff_partition, events_gt)]

    # Computing recall distance
    d_recall = [affiliation_recall_distance(
        Is, J) for Is, J in zip(aff_partition, events_gt)]

    # Computing precision
    p_precision = [affiliation_precision_proba(
        Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    # Computing recall
    p_recall = [affiliation_recall_proba(
        Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(
            p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]  # math.nan
    p_recall_average = sum(p_recall) / len(p_recall)

    dict_out = dict({'precision': p_precision_average,
                     'recall': p_recall_average,
                     'individual_precision_probabilities': p_precision,
                     'individual_recall_probabilities': p_recall,
                     'individual_precision_distances': d_precision,
                     'individual_recall_distances': d_recall})
    return (dict_out)


def produce_all_results():
    """
    Produce the affiliation precision/recall for all files
    contained in the `data` repository
    :return: a dictionary indexed by data names, each containing a dictionary
    indexed by algorithm names, each containing the results of the affiliation
    metrics (precision, recall, individual probabilities and distances)
    """
    datasets, Tranges = read_all_as_events()  # read all the events in folder `data`
    results = dict()
    for data_name in datasets.keys():
        results_data = dict()
        for algo_name in datasets[data_name].keys():
            if algo_name != 'groundtruth':
                results_data[algo_name] = pr_from_events(datasets[data_name][algo_name],
                                                         datasets[data_name]['groundtruth'],
                                                         Tranges[data_name])
        results[data_name] = results_data
    return (results)


import numpy as np
def causal_cusum(score, k=None, decay=0.97, clip=None):
    """
    Causal CUSUM / 积分型持续性增强：
        s[t] = max(0, decay * s[t-1] + (score[t] - k))

    直觉：
      - score 只有“持续高于基线 k”时，s 才会累积变大
      - 短尖峰/零碎假阳性会被抑制（precision 往往明显提升）
      - 持续异常通常仍能累起来（recall 一般不至于大幅下降）

    Args:
      score: 1D array-like, anomaly score
      k: 基线(漂移项). None 时用 score 的 0.95 分位作为默认基线
      decay: 记忆衰减系数(0~1), 越接近1越强调“持续性”(更稳但响应慢)
      clip: 可选，对输出做上限裁剪，防止极端爆炸（例如 clip=1e6）

    Returns:
      out: 1D np.ndarray, CUSUM-transformed score
    """
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = score.size
    if n == 0:
        return score

    if k is None:
        k = float(np.quantile(score, 0.95))
    else:
        k = float(k)

    decay = float(decay)
    if not (0.0 <= decay <= 1.0):
        raise ValueError("decay must be in [0, 1].")

    out = np.empty_like(score)
    s = 0.0
    for t, x in enumerate(score):
        s = decay * s + (x - k)
        if s < 0.0:
            s = 0.0
        if clip is not None and s > clip:
            s = clip
        out[t] = s

    return out


def rolling_zscore(score, W=200, eps=1e-6):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = score.size
    if n == 0:
        return score
    out = np.empty_like(score)

    # 前缀和做均值/方差
    c1 = np.cumsum(score)
    c2 = np.cumsum(score * score)

    for t in range(n):
        s = max(0, t - W + 1)
        m = t - s + 1
        sum1 = c1[t] - (c1[s-1] if s > 0 else 0.0)
        sum2 = c2[t] - (c2[s-1] if s > 0 else 0.0)
        mu = sum1 / m
        var = max(sum2 / m - mu * mu, 0.0)
        std = np.sqrt(var) + eps
        out[t] = (score[t] - mu) / std

    # 只保留正异常（非常适合提升 precision）
    return np.maximum(out, 0.0)


import numpy as np

def postprocess_pred(pred, min_len=5, max_gap=3):
    """
    pred: 0/1 向量
    min_len: 预测事件最短长度，小于它的事件直接删除（提升 precision）
    max_gap: 事件内部/相邻事件间允许的最大空洞长度，<=max_gap 的 0 段会被填成 1（提升 recall / 连贯性）
    """
    pred = np.asarray(pred, dtype=np.uint8).reshape(-1)
    n = pred.size
    if n == 0:
        return pred

    # ---- 1) 填补短 gap（把 1..1 中间很短的 0..0 填成 1）----
    # 找到所有 run（变化点）
    x = pred
    change = np.flatnonzero(np.diff(np.r_[0, x, 0]))
    # ones runs: [s0,e0), [s1,e1) ... (右端点为exclusive)
    ones_s = change[0::2]
    ones_e = change[1::2]

    # 填 gap：相邻 ones 之间的间隔 <= max_gap
    for i in range(len(ones_s) - 1):
        gap = ones_s[i + 1] - ones_e[i]
        if gap > 0 and gap <= max_gap:
            pred[ones_e[i]:ones_s[i + 1]] = 1

    # ---- 2) 删除短事件（长度 < min_len 的 1 段删掉）----
    x = pred
    change = np.flatnonzero(np.diff(np.r_[0, x, 0]))
    ones_s = change[0::2]
    ones_e = change[1::2]
    for s, e in zip(ones_s, ones_e):
        if (e - s) < min_len:
            pred[s:e] = 0

    return pred.astype(int)


def rolling_zscore(score, W=200, eps=1e-6):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = score.size
    if n == 0:
        return score
    out = np.empty_like(score)

    # 前缀和做均值/方差
    c1 = np.cumsum(score)
    c2 = np.cumsum(score * score)

    for t in range(n):
        s = max(0, t - W + 1)
        m = t - s + 1
        sum1 = c1[t] - (c1[s-1] if s > 0 else 0.0)
        sum2 = c2[t] - (c2[s-1] if s > 0 else 0.0)
        mu = sum1 / m
        var = max(sum2 / m - mu * mu, 0.0)
        std = np.sqrt(var) + eps
        out[t] = (score[t] - mu) / std

    # 只保留正异常（非常适合提升 precision）
    return np.maximum(out, 0.0)

def hysteresis_binarize(score, thr_on, thr_off):
    score = np.asarray(score, dtype=np.float64)
    pred = np.zeros_like(score, dtype=np.uint8)
    on = False
    for i, s in enumerate(score):
        if not on and s >= thr_on:
            on = True
        elif on and s <= thr_off:
            on = False
        pred[i] = 1 if on else 0
    return pred


def consec_trigger(score, thr_on, thr_off, m_on=3, n_off=3):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    pred = np.zeros_like(score, dtype=np.uint8)
    on = False
    cnt_on = 0
    cnt_off = 0

    for i, s in enumerate(score):
        if not on:
            if s >= thr_on:
                cnt_on += 1
                if cnt_on >= m_on:
                    on = True
                    cnt_off = 0
            else:
                cnt_on = 0
        else:
            if s <= thr_off:
                cnt_off += 1
                if cnt_off >= n_off:
                    on = False
                    cnt_on = 0
            else:
                cnt_off = 0
        pred[i] = 1 if on else 0
    return pred


def to_quantile(score):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = score.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(score.size, dtype=np.float64)
    q = ranks / max(score.size - 1, 1)
    return q



import pandas as pd
def evaluate(results_storage, metrics, labels, score, **args):
    if "affiliation" in metrics:
        results = []
        # score = causal_cusum(score, k=np.quantile(score, 0.9), decay=0.9)
        # score = rolling_zscore(score, W=2)
        # score = to_quantile(score)
        for thre in args['affiliation']:
            result = {}
            pred = (score > thre).astype(int)
            # thr_on = thre
            # thr_off = 0.8 * thre    # 核心参数，0.6~0.9 都值得扫
            # pred = hysteresis_binarize(score, thr_on, thr_off)
            # pred = postprocess_pred(pred, min_len=3, max_gap=2)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            events_label = convert_vector_to_events(labels)
            events_pred = convert_vector_to_events(pred)
            Trange = (0, len(pred))
            affiliation_metrics = pr_from_events(events_pred, events_label, Trange)
            result['Affiliation_thre'] = thre
            result['Affiliation_ACC'] = accuracy
            result['Affiliation_P'] = P = affiliation_metrics['precision']
            result['Affiliation_R'] = R = affiliation_metrics['recall']
            result['Affiliation_F1'] = 2 * P * R / (P + R)
            results.append(pd.DataFrame([result]))
        results_storage['affiliation'] = pd.concat(results, axis=0).reset_index(drop=True)
