## clustering algorithm test - iasl

## library import
import numpy as np
import matplotlib.pyplot as plt

## lidar data loading
#data_raw = np.loadtxt('points_portland.txt', delimiter=',')
data = np.loadtxt('filtered_portland.txt')

## lidar data refining
#data = data_raw[data_raw[:, 2] > 300]
#print(data)

## clustering algorithm test

##k-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=15)
kmeans.fit(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('K-means')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=kmeans.labels_, s=20, alpha=0.5, cmap='rainbow')
plt.show()

##DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=18, min_samples=4)
dbscan.fit(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('DBSCAN')
ax.scatter(x, y, z, c=dbscan.labels_, s=20, alpha=0.5, cmap='rainbow')
plt.show()

##Birch
from sklearn.cluster import Birch
birch = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
birch.fit(data)
birch.labels_ = birch.predict(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Birch')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=birch.labels_, s=20, alpha=0.5, cmap='rainbow')
plt.show()

##MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
minibatchkmeans = MiniBatchKMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10,  random_state=0)
minibatchkmeans.labels_ = minibatchkmeans.fit_predict(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('MiniBatchKMeans')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=minibatchkmeans.labels_, s=20, alpha=0.5, cmap='rainbow')
plt.show()

#GaussianMixture
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=15, covariance_type='full', random_state=100)
gmm.fit(data)
gmm.labels_ = gmm.predict(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('GaussianMixture')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=gmm.labels_, s=20, alpha=0.5, cmap='rainbow')
plt.show()

##OPTICS
from sklearn.cluster import OPTICS
optics_clustering = OPTICS(min_samples=3).fit(data)
label = optics_clustering.labels_
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('OPTICS')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=label, s=20, alpha=0.5, cmap='rainbow')
plt.show()

##MeanShift
from sklearn.cluster import MeanShift
meanshift = MeanShift()
meanshift.fit(data)
meanshift_labels = meanshift.fit_predict(data)
# 3차원 데이터 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('MeanShift')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.scatter(x, y, z, c=meanshift_labels, s=20, alpha=0.5, cmap='rainbow')
plt.show()

