import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import f1
from . import affiliation
from . import auc_vus

import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)      # 或 None
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

class Evaluator():
    def __init__(self, gt, anomaly_score, save_path):
        """
        input:
            gt: np.ndarray[int],
            anomaly_score: np.ndarray[float],
            save_path: str
        """
        self.gt = gt
        self.anomaly_score = anomaly_score
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def evaluate(self, metrics, merge=False, verbose=True, **metrics_args):
        """
        support metric: 'affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa', 'best_f1'

        input:
            metrics: List[str], e.g. ['affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa']
            metrics_args: Dict[str, args], e.g. {'affiliation': [0.01, 0.02], 'f1_raw': [0.1, 0.2], ..., 'sliddingWindow': 100}
            merge: bool, if True: merge all results from different metrics
        output:
            results_storage: Dict[str, Dict[str, List[float]]]
        """
        results_storage = {}
        # metrics
        f1.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        affiliation.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        auc_vus.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)

        # save
        self._save_csv(results_storage, metrics, merge=merge, verbose=verbose)
        return results_storage
    
    def _save_csv(self, results_storage, metrics, merge, verbose):
        if merge:
            df = pd.concat([pd.DataFrame.from_dict(results_storage[metric]) for metric in metrics], axis=1)
            if verbose: print(df)
            df.to_csv(f'{self.save_path}/_results.csv', index=0)
        else:
            for metric in metrics:
                df = pd.DataFrame.from_dict(results_storage[metric])
                if verbose: print(df)
                df.to_csv(f'{self.save_path}/_{metric}.csv', index=0)

    def find_thres(self, method, verbose=True, **args):
        """
        support method: 'prior_anomaly_rate', 'spot'  

        input:  
            method: str   
            args:  
            method == 'prior_anomaly_rate': require pAR: List[float]  
            method == 'spot': require init_score: np.ndarray[float]; q: List[float]  
        output:  
            thresholds: List[float]  
        
        example:  
            thres = evaluator.find_thres(method='prior_anomaly_rate', pAR=[0.05, 0.1])  
            thres = evaluator.find_thres(method='spot', init_score=init_score, q=[0.1, 0.2])  
        """
        if method == 'prior_anomaly_rate':
            thresholds = [np.percentile(self.anomaly_score, 100 * (1-pAR)) for pAR in args['pAR']]
            self._save_thres_info(args['pAR'], thresholds, method, verbose)
        elif method == 'spot':
            from .spot import SPOT
            thresholds = []
            for q in args['q']:
                s = SPOT(q)
                s.fit(args['init_score'], self.anomaly_score)
                s.initialize(verbose=False)
                ret = s.run()
                thresholds.append(np.mean(ret['thresholds']))
            self._save_thres_info(args['q'], thresholds, method, verbose)

        return thresholds
    
    def _save_thres_info(self, arg1, arg2, method, verbose):
        thres_info = pd.DataFrame(np.stack([arg1, arg2], axis=1), columns=['hyper-parameter', 'threshold'])
        if verbose: print(thres_info)
        thres_info.to_csv(f'{self.save_path}/_{method}_thres_info.csv', index=0)
    
    def _find_borders(self, gt):
        """
        return anomaly intervals: [[start1, end1]...[startn, endn]]
        """
        borders = []
        s = 0
        while True:
            if gt[s]==1:
                e = s
                while gt[e]==1:
                    e += 1
                    if e == len(gt): break
                borders.append([s, e-1])
                s = e+1
            else:
                s += 1
            if s >= len(gt): break
        return borders
    
    def vis_anomaly_intervals_1(self, series=None, start=None, end=None):
        if start is None: start = 0
        if end is None: end = len(self.gt)

        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            series = series[start:end]
        
        gt = self.gt[start:end]
        anomaly_score = self.anomaly_score[start:end]

        borders = self._find_borders(gt)
        n_anomalies = len(borders)

        vis_path = os.path.join(self.save_path, "vis")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    
        for c in range(nc):
            fig, ax1 = plt.subplots(figsize=(20, 5))

            # series
            if series is not None:
                ax1.plot(range(start, end), series[:, c], linewidth=1, label='series', color='#5861AC')
                ax1.set_ylabel('Series value')

            # anomaly score
            ax2 = ax1.twinx()
            ax2.plot(range(start, end), anomaly_score, linewidth=0.2, label='anomlay score', color='#72C3A3')
            ax2.fill_between(range(start, end), anomaly_score, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
            ax2.set_ylabel('Anomaly score')
            ax2.set_xlabel('Time step')

            # abnormly interval
            for i in range(n_anomalies):
                if i == 0: 
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+1+start, ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.3)
                else:
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+1+start, ymin=0, ymax=1, facecolor='r', alpha=0.2)
            
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
            plt.savefig(f'{vis_path}/vis_{start}-{end}_c{c}.pdf', bbox_inches='tight')
            plt.clf()

    def vis_anomaly_intervals_2(self, series=None, max_span=100, max_anomalies=1):
        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            
        borders = self._find_borders(self.gt)
        n_anomalies = len(borders)

        vis_path = os.path.join(self.save_path, "vis_abnorm")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        for c in range(nc):
            for i in range(n_anomalies):
                start = 0
                end = len(self.anomaly_score)
                if i:
                    start = borders[i-1][1]+1
                if i+1 < n_anomalies:
                    end = borders[i+1][0]
                
                if max_span is not None:
                    start = max(start, borders[i][0]-max_span)
                    end = min(end, borders[i][1]+max_span)

                fig, ax1 = plt.subplots()

                # series
                if series is not None:
                    series_i = series[start:end, c]
                    ax1.plot(range(start, end), series_i, linewidth=1, label='series', color='#5861AC')
                    ax1.set_ylabel('Series value')

                # anomaly score
                ax2 = ax1.twinx()
                score_i = self.anomaly_score[start:end]
                ax2.plot(range(start, end), score_i, linewidth=0.2, color='#72C3A3')
                ax2.fill_between(range(start, end), score_i, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
                ax2.set_ylabel('Anomaly score')
                ax2.set_xlabel('Time step')

                # abnormly interval
                plt.axvspan(xmin=borders[i][0], xmax=borders[i][1]+1, ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.2)

                fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
                plt.savefig(f'{vis_path}/{i}_{borders[i][0]}-{borders[i][1]}_c{c}.pdf', bbox_inches='tight')
                plt.clf()

                if max_anomalies is None:
                    continue
                if i == max_anomalies-1: break
                