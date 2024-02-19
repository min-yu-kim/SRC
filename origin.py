import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import pptk
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def least_square(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    a = ((x - mean_x) * (y - mean_y)).sum() / ((x - mean_x) ** 2).sum()
    return a
def rotate(point, angle, pivot=(0, 0)):
    x, y = point
    px, py = pivot
    angle = math.radians(angle)
    # print(angle)
    qx = px + math.cos(angle) * (x - px) - math.sin(angle) * (y - py)
    qy = py + math.sin(angle) * (x - px) + math.cos(angle) * (y - py)
    return [qx, qy]
def create_grid_coord(coords, grid_size, theta):
    rect = coords
    x, y = zip(*rect)  # x, y 좌표 추출

    if max(x) - min(x) < grid_size or max(y) - min(y) < grid_size:
        return [rect]

    dx = grid_size  # x 좌표 간격
    dy = grid_size  # y 좌표 간격

    # 그리드 생성
    grid_coords = []
    for i in range(int((max(x) - min(x)) / dx)):
        for j in range(int((max(y) - min(y)) / dy)):
            # 그리드 좌표 계산
            x1 = min(x) + i * dx
            x2 = min(x) + (i + 1) * dx
            y1 = min(y) + j * dy
            y2 = min(y) + (j + 1) * dy

            # 그리드 좌표 저장
            grid_coords.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # 회전축을 기준으로 좌표 회전
    center_x = (max(x) + min(x)) / 2  # x 중심점
    center_y = (max(y) + min(y)) / 2  # y 중심점
    pivot = (center_x, center_y)  # 회전축
    angle = math.degrees(theta)
    # angle = math.degrees(math.atan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0]))  # 각도 계산
    grid_coords = [[rotate(point, angle, pivot=pivot) for point in coord] for coord in grid_coords]
    print(grid_coords)
    return grid_coords

cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki',
        'greenyellow', 'chartreuse', 'limegreen', 'lime',
        'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
        'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta',
        'mistyrose', 'lightskyblue', 'aquamarine', 'paleturquoise', 'lavenderblush',
        'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
        'mediumpurple', 'salmon', 'wheat', 'powderblue',
        'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
        'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue',
        'goldenrod', 'lightslategray', 'lightgray', 'limegreen',
        'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen', 'mediumaquamarine',
        'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab',
        'darkkhaki', 'khaki', 'peachpuff', 'rosybrown', 'slategray',
        'darkslategray', 'mediumblue', 'navy', 'mediumslateblue', 'rebeccapurple',
        'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']

# data open
# data = np.loadtxt('filtered_portland.txt')
data = pd.read_csv('filtered_DBSCAN_portland.csv')
# data = pd.read_csv('../input/filtered_points_portland.csv', delimiter=',') # 테스트용 가벼운 데이터

# print(data)
data = np.array(data)
# print(data)
data_3d = data[:, :3]
# print(data_3d) #xyz좌표 뽑기
# vis = pptk.viewer(data_3d)
pcd = o3d.geometry.PointCloud()  # 포인트 클라우드 생성
pcd.points = o3d.utility.Vector3dVector(data_3d)  # float64 numpy array를 (n,3) open3d포맷으로 바꿔줌
#o3d.visualization.draw_geometries([pcd])

# birch algorithm
# from sklearn.cluster import Birch
# birch = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
# birch.fit(data)
# birch.labels_ = birch.predict(data)
# labels = birch.labels_
# labels = pd.DataFrame(labels)
data_2d = data[:, :2]
label = data[:, 3]
# from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(eps=18, min_samples=5)
# dbscan.fit(data_2d)
# labels = dbscan.labels_
# from sklearn.cluster import MeanShift
# meanshift = MeanShift()
# meanshift.fit(data_2d)
# meanshift_labels = meanshift.fit_predict(data_2d)
#
fig = plt.figure(figsize=(10, 10))
x = data_2d[:, 0]
y = data_2d[:, 1]
# plt.title('DBSCAN')
# plt.scatter(x, y, c=label, s=20, alpha=0.5, cmap='rainbow')
# plt.savefig('plot_portland_final.png')
# plt.show()


# # data + label(x,y,z,label)
label = pd.DataFrame(label)
label = label.to_numpy() # [[]...[]]^T(열)로 라벨링 값을 저장
# np.savetxt('data.csv', meanshift.labels_, delimiter=',')
# print(label)
data_label = np.append(data_3d, label, axis=1)  # data_3d 데이터에 label를 붙임 axis=1은 열을 추가해서 붙인다는 것
# print(data_label)   # (x,y,z,label)를 얻음
df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])  # pandas형태로 저장
df_np = df_pd.to_numpy()

# fig = plt.figure(figsize=(10, 10))
fig = go.Figure()
# ax = fig.add_subplot(111, projection='3d')

for n in range(int(max(df_pd['label'])) + 1):
    n0 = df_pd[df_pd['label'] == n]  # df['label']에서 ==n을 만족하는 df를 n0에 저장
    # print(n0)
    n_xyz = n0[['x', 'y', 'z']]  # [[]]인 이유는 x,y를 하나의 데이터 좌표로 받아 와야 함
    # print(n_xy)
    n_xyz = n_xyz.to_numpy()  # xy 좌표 pandas를 numpy로 바꿈
    n_x = n_xyz[:, 0]
    n_y = n_xyz[:, 1]
    n_z = n_xyz[:, 2]
    # print(n_x)
    z_max = np.max(n_z)
    n0 = df_pd[df_pd['label'] == n]  # df['label']에서 ==n을 만족하는 df를 n0에 저장
    # print(n0)
    n_xy = n0[['x', 'y']]  # [[]]인 이유는 x,y를 하나의 데이터 좌표로 받아 와야 함
    # print(n_xy)
    n_xy = n_xy.to_numpy()  # xy 좌표 pandas를 numpy로 바꿈
    # print(n_xy)
    # print(n_xy)



    hull = ConvexHull(n_xy)  # convex hull로 주어진 점의 외접하게 다각형을 그림
    coord = n_xy[hull.vertices]
    hull = ConvexHull(n_xy)
    hull_points = n_xy[hull.vertices]


    cov = np.cov(n_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    slope = largest_eigenvector[1] / largest_eigenvector[0]
    theta = np.arctan(slope)
    # a = least_square(n_x, n_y)  # 최소자승법으로 구한 기울기

    # theta = math.atan(eigenvectors)
    # theta = np.arctan(a)# 기울기 각도
    # print(theta)
    if theta > np.pi /2:
        theta = -theta
    n_x_new = n_x * math.cos(theta) + n_y * math.sin(theta)  # 주어진 점들을 변환 행렬를 이용해서 좌표이동
    n_y_new = -n_x * math.sin(theta) + n_y * math.cos(theta)
    #
    n_x_new_max = max(n_x_new)  # 여기서 변환된 좌표의 최대값 최솟값을 구함
    n_y_new_max = max(n_y_new)
    n_x_new_min = min(n_x_new)
    n_y_new_min = min(n_y_new)
    # 여기서 부터는 최댓값이랑 최솟값을 역변환해서 실제 포인트들을 외접하는 사각형의 꼭짓점을 구함
    n_x_ver1 = n_x_new_max * math.cos(theta) - n_y_new_max * math.sin(theta)
    n_y_ver1 = n_x_new_max * math.sin(theta) + n_y_new_max * math.cos(theta)
    n_x_ver2 = n_x_new_max * math.cos(theta) - n_y_new_min * math.sin(theta)
    n_y_ver2 = n_x_new_max * math.sin(theta) + n_y_new_min * math.cos(theta)
    n_x_ver3 = n_x_new_min * math.cos(theta) - n_y_new_max * math.sin(theta)
    n_y_ver3 = n_x_new_min * math.sin(theta) + n_y_new_max * math.cos(theta)
    n_x_ver4 = n_x_new_min * math.cos(theta) - n_y_new_min * math.sin(theta)
    n_y_ver4 = n_x_new_min * math.sin(theta) + n_y_new_min * math.cos(theta)
    # print(n_x_ver1)
    plt.scatter(n_x, n_y)
    #
    coords = [[n_x_ver1, n_y_ver1], [n_x_ver2, n_y_ver2], [n_x_ver4, n_y_ver4], [n_x_ver3, n_y_ver3]]
    # print(coords)
    points = np.array(coords)
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o')  # 좌표들을 plot합니다.
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')  # 볼록 껍질의 각 변
    grids = create_grid_coord(points, 200, theta)
    # print(grids)
    # print(grids[:, 0])

    # plt.scatter(x, y, c=label, s=20, alpha=0.5, cmap='rainbow')
    # plt.scatter()
    max_heights = []
    for i, grid in enumerate(grids):
        max_height = -np.inf
        for vertex in grid:
            # vertex 좌표와 가장 가까운 pcd_data의 좌표 인덱스 찾기
            dist = np.linalg.norm(data_3d[:, :2] - vertex, axis=1)
            closest_idx = np.argmin(dist)
            # 가장 가까운 pcd_data 좌표의 높이가 현재까지의 최대 높이보다 높을 경우 최대 높이 갱신
            height = data_3d[closest_idx, 2]
            if height > max_height:
                max_height = height
        # 최대 높이를 max_heights 리스트에 추가
        max_heights.append(max_height)
        # print(max_height)

        grid_matrix = np.zeros((len(grid), 3))
        for j, vertex in enumerate(grid):
            # print(vertex)
            grid_matrix[j, :] = [vertex[0], vertex[1], max_heights[i]]

            for k, l in zip(grid_matrix[:, 0], grid_matrix[:, 1]):
            #     # plt.scatter(k, l, c='k', s=20)
                if k != 0 or l != 0:
                    # xx, yy = np.meshgrid(k, l)
                    # plt.scatter(xx, yy, c='k')
                    plt.scatter(k, l, c='k', s=20)
        # print(grid_matrix)
        # print(grid_matrix[:, 0])
        # plt.scatter(x, y, c=label, s=20, alpha=0.5, cmap='rainbow')

        new_arr = []
        for i in range(0, len(grid_matrix), 2):
            if i + 1 < len(grid_matrix):
                new_arr.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
            else:
                new_arr.append(np.array([grid_matrix[i], grid_matrix[0]]))
            # print(new_arr)
        new_arr2 = []
        for i in range(1, len(grid_matrix), 2):
            if i + 1 < len(grid_matrix):
                new_arr2.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
            else:
                new_arr2.append(np.array([grid_matrix[i], grid_matrix[0]]))

        traces = []
        for arr in new_arr:
            vertices1 = [(x, y, 250) for x, y, _ in arr]
            vertices2 = [(x, y, z) for x, y, z in arr]
            vertices = np.vstack((vertices1, vertices2))
            x, y, z = zip(*vertices)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n], showscale=False))

        for arr in new_arr2:
            vertices1 = [(x, y, 250) for x, y, _ in arr]
            vertices2 = [(x, y, z) for x, y, z in arr]
            vertices = np.vstack((vertices1, vertices2))
            x, y, z = zip(*vertices)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n], showscale=False))

        vertices = [(x, y, 250) for x, y, _ in grid_matrix]
        vertices2 = [(x, y, z) for x, y, z in grid_matrix]
        x, y, z = zip(*vertices)
        # fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='green', opacity=0.20)])
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
        x, y, z = zip(*vertices2)
        # fig = go.Figure(data=[go.Mesh3d(x=x2, y=y2, z=z2, color='green', opacity=0.20)])
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
# fig.show()
plt.show()