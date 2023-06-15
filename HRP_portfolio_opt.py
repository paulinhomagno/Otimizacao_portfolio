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
        
        # Inicialização de pesos
        pesos = pd.Series(1, index=self._columns_s)
        paridades = [self._columns_s]

        while len(paridades) > 0:
            # Instanciação de clusters
            paridades = [cluster[inicio:fim] 
                         for cluster in paridades
                          for inicio, fim in ((0, len(cluster) // 2),(len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]

            # Iteração entre paridades
            for subcluster in range(0, len(paridades), 2):

                cluster_esquerdo = paridades[subcluster]
                cluster_direito = paridades[subcluster + 1]

                matriz_cov_esquerda = matrix_cov[cluster_esquerdo].loc[cluster_esquerdo]
                inversa_diagonal = 1 / np.diag(matriz_cov_esquerda.values)
                pesos_cluster_esquerdo = inversa_diagonal / np.sum(inversa_diagonal)
                vol_cluster_esquerdo = np.dot(pesos_cluster_esquerdo, np.dot(matriz_cov_esquerda, pesos_cluster_esquerdo))

                matriz_cov_direita = matrix_cov[cluster_direito].loc[cluster_direito]
                inversa_diagonal = 1 / np.diag(matriz_cov_direita.values)
                pesos_cluster_direito = inversa_diagonal  / np.sum(inversa_diagonal)
                vol_cluster_direito = np.dot(pesos_cluster_direito, np.dot(matriz_cov_direita, pesos_cluster_direito))

                fator_alocacao = 1 - vol_cluster_esquerdo / (vol_cluster_esquerdo + vol_cluster_direito)

                pesos[cluster_esquerdo] *= fator_alocacao
                pesos[cluster_direito] *= 1 - fator_alocacao

        return pesos
    
    