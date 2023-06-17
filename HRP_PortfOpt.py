import seaborn as sns
import pandas as pd
import numpy as np


def get_columns_seriation(returns):

        matrix_cov = returns.cov()

        dendogram = sns.clustermap(matrix_cov, method='ward', metric='euclidean')

        seriation = dendogram.dendrogram_col.reordered_ind
        columns_seriation = returns.columns[seriation]
        return columns_seriation


class HRP():
    
    def __init__(self, returns):
        self.returns = returns
        self._columns_s = get_columns_seriation(returns)
    

    def calculate_hrp(self):
        
        matrix_cov = self.returns.cov()
        
        # weights and parities
        # weights starting with value 1
        weights = pd.Series(1, index=self._columns_s)
        parities = [self._columns_s]

        while len(parities) > 0:
            # clusters instance
            parities = [cluster[start:end] 
                         for cluster in parities
                          for start, end in ((0, len(cluster) // 2),(len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]

            # Iteration over parities
            for subcluster in range(0, len(parities), 2):

                cluster_l = parities[subcluster]
                cluster_r = parities[subcluster + 1]

                matrix_cov_l = matrix_cov[cluster_l].loc[cluster_l]
                inv_diag = 1 / np.diag(matrix_cov_l.values)
                w_cluster_l = inv_diag / np.sum(inv_diag)
                vol_cluster_l = np.dot(w_cluster_l, np.dot(matrix_cov_l, w_cluster_l))

                matrix_cov_r = matrix_cov[cluster_r].loc[cluster_r]
                inv_diag = 1 / np.diag(matrix_cov_r.values)
                w_cluster_r = inv_diag  / np.sum(inv_diag)
                vol_cluster_r = np.dot(w_cluster_r, np.dot(matrix_cov_r, w_cluster_r))

                factor = 1 - vol_cluster_l / (vol_cluster_l + vol_cluster_r)

                weights[cluster_l] *= factor
                weights[cluster_r] *= 1 - factor

        return weights
    
    