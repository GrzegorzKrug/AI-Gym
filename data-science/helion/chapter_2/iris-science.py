import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option('max_columns', 15)


df_iris = pd.read_csv('iris.csv', header=None)

print(f"Head: {df_iris.head(2)}")
print()
print(f"Tail: {df_iris.tail(2)}")

df_iris.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
print(f"Columns: {df_iris.columns}")

print(f"Target\n{df_iris['target']}")

mask = df_iris['sepal-length'] > 6
iris_cut = df_iris[df_iris['sepal-length'] > 6]
# iris_cut = df_iris[mask]

print(f"Length is changed by: {len(df_iris) - len(iris_cut)}, "
      f"it is {round((len(df_iris) - len(iris_cut))/len(df_iris), 3)}%")
print(f"mean:\n ", df_iris.groupby(['target']).mean())
print(f"sorted:\n", df_iris.sort_values(['sepal-width']).head())
print(df_iris.groupby(['target']).describe())

color = {'setosa': 'b', 'versicolor': 'r', 'virginica': 'g'}
colors = [color[x] for x in df_iris['target']]
fig1 = df_iris.plot(kind='scatter', x='petal-width', y='petal-length', c=colors, s=50, marker='s')
## df_iris.plot(kind='hist', alpha=0.5, bins=20)

iris = datasets.load_iris()
# print(iris.data.T)
cov_data = np.corrcoef(iris.data.T)
print(cov_data)

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

_x = _y = np.arange(4)
_xx, _yy = np.meshgrid(_x, _y)
xx, yy = _xx.ravel(), _yy.ravel()

_cov_data = cov_data.ravel()

bottom = np.zeros_like(xx) + min(_cov_data)
_cov_data -= min(_cov_data)
width = depth = .25


def get_color_with_interval(xx, n=5):
    _cols = ['r', 'm', 'm', 'b', 'c', 'y', 'y', 'g']
    n = int(n) - 1
    if n < 2 or n > len(_cols):
        raise ValueError(f"N must be in range 2 and {len(_cols)}: {n}")

    xx = xx - min(xx)
    interval = max(xx) / (n-1)
    edges = [interval * _n for _n in range(1, n-1)]
    colors = []

    for x in xx:
        _set = False
        for c, edg in zip(_cols, edges):
            if x < edg:
                colors.append(c)
                _set = True
                break
        if not _set:
            colors.append(_cols[n-1])

    return colors


colors = get_color_with_interval(_cov_data)
ax.bar3d(xx, yy, bottom, width, depth, _cov_data, shade=True, color=colors)
# plt.show()
X = df_iris[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].to_numpy()
y = df_iris['target'].to_numpy()
# print(y)

df_train, df_valid, y_train, y_valid = train_test_split(X, y, test_size=0.7)
# df_test, df_valid, y_test, y_valid = train_test_split(df_valid, y_valid, test_size=0.5)

print(len(df_train), len(y_train))
# print(y_train)
classifier = DecisionTreeClassifier(max_depth=2)

classifier.fit(df_train, y_train)
resp = classifier.predict(df_valid)

# fig1 = plt.figure()
colors = [color[x] for x in resp]
fig1.scatter(df_valid[:, 3], df_valid[:, 2], c=colors, s=20, marker='o')
plt.show()

