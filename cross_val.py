import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

if __name__ == '__main__':
    dataset = pd.read_csv('data/felicidad.csv')

    print(dataset)

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    # Prueba rapida
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')

    print(score)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)
