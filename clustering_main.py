import numpy as np
import open3d as o3d
import pptk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import convex_hull_plot_2d
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import math

# data open
# data = np.loadtxt('filtered_portland.txt')
data = pd.read_csv('../input/filtered_points_300_sandiego.csv') # 엑셀 파일을 pandas를 통해 읽음 103809x3 행렬이 나옴 여기서 3이 xyz 좌표
# print(data)
data = np.array(data) # pandas 데이터를 numpy로 바꾸는데 103809x3행렬로 묶이게 됨. 여기서 데이터는 행끼리 묶임
# print(data)
data_3d = data[:, :3] #[행, 열] [lower:upper:step]이니까 일단 행은 데이터를 다 뽑고 :3은 열의 2번째  index까지 즉 xyz를 뽑는거를 의미한다
# print(data_3d) #xyz좌표 뽑기
# vis = pptk.viewer(data_3d)  #viewer를 통해 포인트 클라우드를 볼 수 있음
pcd = o3d.geometry.PointCloud()  #포인트 클라우드 생성
pcd.points = o3d.utility.Vector3dVector(data_3d)  # float64 numpy array를 (n,3) open3d포맷으로 바꿔줌
# o3d.visualization.draw_geometries([pcd])
#
# birch algorithm
# from sklearn.cluster import Birch
# birch = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
# birch.fit(data)
# birch.labels_ = birch.predict(data)
# labels = birch.labels_
# labels = pd.DataFrame(labels)
#
#
## dbscan algorithm
# from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(ps=18, min_samples=4)
# dbscan.fit(data)
# labels = dbscan.labels_
# labels = pd.DataFrame(labels)
# print(labels)

# kmeans algorithm
kmeans = KMeans(n_clusters=20) # 군집 중심점의 개수
kmeans.fit(data_3d) # 포인트 클라우드를 주어진 알고리즘 적용
labels = kmeans.labels_ # 라벨링한 것[...](행)형태로 저장
# print(labels)
labels = pd.DataFrame(labels)   # pandas 형태로 각 점에 라벨링
# print(labels)

# # data + label(x,y,z,label)
label = labels.to_numpy() #[[]...[]]^T(열)로 라벨링 값을 저장
# print(label)
data_label = np.append(data_3d, label, axis=1)  # data_3d 데이터에 label를 붙임 axis=1은 열을 추가해서 붙인다는 것
# print(data_label)   # (x,y,z,label)를 얻음
df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label']) # pandas형태로 저장
df_np = df_pd.to_numpy()
# cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k']
# 라벨링 색깔
# num = 0
# print(df_np)
# print(max(df_np[:,3][:]))
def least_square(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # print(mean_x)
    a = ((x - mean_x)*(y - mean_y)).sum() / ((x-mean_x)**2).sum()
    # print(a)
    return a

fig = plt.figure()
for n in range(int(max(df_pd['label']))+1):
    n0 = df_pd[df_pd['label'] == n]  # df['label']에서 ==n을 만족하는 df를 n0에 저장
    # print(n0)
    n_xyz = n0[['x', 'y', 'z']]  # [[]]인 이유는 x,y를 하나의 데이터 좌표로 받아 와야 함
    # print(n_xy)
    n_xyz = n_xyz.to_numpy()  # xy 좌표 pandas를 numpy로 바꿈
    # num += 1
    n_x = n_xyz[:, 0]
    n_y = n_xyz[:, 1]
    n_z = n_xyz[:, 2]
    # print(n_x)

    a = least_square(n_x, n_y)
    theta = math.atan(a)
    # print(theta)
    n_x_new = n_x*math.cos(theta) - n_y*math.sin(theta)
    n_y_new = n_x*math.sin(theta) + n_y*math.cos(theta)
    n_x_new_max = max(n_x_new)
    n_y_new_max = max(n_y_new)
    n_x_new_min = min(n_x_new)
    n_y_new_min = min(n_y_new)

    n_x_new1 = np.linspace(n_x_new_min, n_x_new_max, 10)
    n_y_new1 = np.linspace(n_y_new_min, n_y_new_max, 10)
    # print('x :', n_x_new1)
    # print('y :', n_y_new1)
    n_x_rect, n_y_rect = np.meshgrid(n_x_new1, n_y_new1)
    n_x_rect1 = n_x_rect * math.cos(theta) + n_y_rect * math.sin(theta)
    n_y_rect1 = - n_x_rect * math.sin(theta) + n_y_rect * math.cos(theta)
    # print('x :', n_x_rect)
    # print('y :', n_y_rect)

    plt.scatter(n_x, n_y)
    plt.scatter(n_x_rect1, n_y_rect1)

plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for n in range(int(max(df_pd['label']))+1):
#     n0 = df_pd[df_pd['label'] == n]  # df['label']에서 ==n을 만족하는 df를 n0에 저장
#     # print(n0)
#     n_xyz = n0[['x', 'y', 'z']]  # [[]]인 이유는 x,y를 하나의 데이터 좌표로 받아 와야 함
#     # print(n_xy)
#     n_xyz = n_xyz.to_numpy()  # xy 좌표 pandas를 numpy로 바꿈
#     # print(n_xyz)
#     # print('number = ', num)
#     # num += 1
#     n_x = n_xyz[:, 0]
#     n_y = n_xyz[:, 1]
#     n_z = n_xyz[:, 2]
#     ax.scatter(n_x, n_y, n_z)
#
# # ax = fig.gca( proj3d='3d')
# # ax.scatter(n_x, n_y, n_z)
# plt.show()
