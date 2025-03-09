import pandas as pd
from sklearn import datasets

# Funktion zur Berechnung der Mittelwerte
def calcMeans(dataframe):
    return dataframe.apply(lambda x: x.mean())

# Test mit kleinem DataFrame
test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("Test DataFrame Mean:")
print(calcMeans(test_df))
print(test_df.describe())

# Iris-Datensatz laden und Funktion anwenden
iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
mean_values = calcMeans(iris_X)
print("\nIris DataFrame Mean:")
print(mean_values)
print(iris_X.describe())