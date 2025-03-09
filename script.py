import pandas as pd
from sklearn import datasets

# func Mean
def calcMeans(dataframe):
    return dataframe.apply(lambda x: x.mean())

# DataFrame
test_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
print("Test DataFrame Mean:")
print(calcMeans(test_df))
print(test_df.describe())

# Load and print
iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
mean_values = calcMeans(iris_X)
print("\nIris DataFrame Mean:")
print(mean_values)
print(iris_X.describe())