"""
Dejar que el algoritmo busque el numero de categorias

"""

import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    dataset = pd.read_csv('data/candy.csv')

    print(dataset.head(10))

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    print(meanshift.labels_)
    print(max(meanshift.labels_))

    print('='*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_

    print(dataset)
