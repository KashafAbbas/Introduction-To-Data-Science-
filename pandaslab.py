import pandas as pd
import numpy as np

iris = pd.read_csv('Iris.csv')


missing_data = iris.isnull().sum()
print("Missing values per column:\n", missing_data)

numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

dups=iris.duplicated()
print(iris[dups])

iris['sepal_area'] = iris['SepalLengthCm'] * iris['SepalWidthCm']
iris['petal_area'] = iris['PetalLengthCm'] * iris['PetalWidthCm']
iris['total_area'] = iris['sepal_area'] + iris['petal_area']
print("Added sepal_area, petal_area, and total_area columns.")
print('Total area',iris['total_area'])


iris.dropna(inplace=True)
print("Dropped rows with any remaining missing values.")

species_mapping = {species: idx for idx, species in enumerate(iris['Species'].unique())}
iris['species_numeric'] = iris['Species'].map(species_mapping)
print("Converted species column to numeric format:", species_mapping)

grouped_sum = iris.groupby('Species').sum(numeric_only=True)
print("Grouped aggregation (sum of numeric columns by species):\n", grouped_sum)


iris_long = pd.melt(
    iris,
    id_vars=['Species', 'species_numeric'],
    value_vars=numeric_columns + ['sepal_area', 'petal_area', 'total_area'],
    var_name='attribute',
    value_name='value'
)
print("Reshaped dataset into long format:\n", iris_long.head())

print("Final dataset after all tasks:\n", iris.head())








