import function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


class Building:
    def __init__(self, label, df_pd, cmap, grid_size):
        self.label = label
        self.df_pd = df_pd
        self.cmap = cmap
        self.grid_size = grid_size
        self.rotated_points = None

    def process(self):
        n0 = self.df_pd[self.df_pd['label'] == self.label]
        n_xy = n0[['x', 'y']]
        plt.scatter(self.df_pd.loc[self.df_pd['label'] == self.label, 'x'], self.df_pd.loc[self.df_pd['label'] == self.label, 'y'], color=self.cmap[self.label], s=2, alpha=0.5)
        theta = self.pca(n_xy)
        coords = self.matrix_rotate(theta, n_xy)
        rotated_coords = self.grid_rotate(coords, theta)
        grid_coords = self.grid_generate(rotated_coords)
        rotated_points = self.grid_rotate_to_origin(rotated_coords, grid_coords, -theta)
        selected_grids = self.select_intersecting_grids(rotated_points, x, y)
        self.plot_3d(self.label, fig, data_3d, selected_grids)

    def pca(self, n_xy):
        n_xy = n_xy.to_numpy()
        # plt.scatter(n_x, n_y)
        cov = np.cov(n_xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        largest_eigenvalue_index = np.argmax(eigenvalues)
        largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
        slope = largest_eigenvector[1] / largest_eigenvector[0]
        theta = np.arctan(slope)
        return theta

    def matrix_rotate(self, theta, n_xy):
        n_xy = n_xy.to_numpy()
        n_x = n_xy[:, 0]
        n_y = n_xy[:, 1]
        if theta > np.pi / 2:
            theta = -theta
        n_x_new = n_x * math.cos(theta) + n_y * math.sin(theta)
        n_y_new = -n_x * math.sin(theta) + n_y * math.cos(theta)
        n_x_new_max = max(n_x_new)
        n_y_new_max = max(n_y_new)
        n_x_new_min = min(n_x_new)
        n_y_new_min = min(n_y_new)

        n_x_ver1 = n_x_new_max * math.cos(theta) - n_y_new_max * math.sin(theta)
        n_y_ver1 = n_x_new_max * math.sin(theta) + n_y_new_max * math.cos(theta)
        n_x_ver2 = n_x_new_max * math.cos(theta) - n_y_new_min * math.sin(theta)
        n_y_ver2 = n_x_new_max * math.sin(theta) + n_y_new_min * math.cos(theta)
        n_x_ver3 = n_x_new_min * math.cos(theta) - n_y_new_max * math.sin(theta)
        n_y_ver3 = n_x_new_min * math.sin(theta) + n_y_new_max * math.cos(theta)
        n_x_ver4 = n_x_new_min * math.cos(theta) - n_y_new_min * math.sin(theta)
        n_y_ver4 = n_x_new_min * math.sin(theta) + n_y_new_min * math.cos(theta)
        coords = [[n_x_ver1, n_y_ver1], [n_x_ver2, n_y_ver2], [n_x_ver4, n_y_ver4], [n_x_ver3, n_y_ver3]]
        return coords

    def grid_rotate(self, coords, theta):
        points = np.array(coords)
        hull = ConvexHull(points)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # for simplex in hull.simplices:
        #   plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        x, y = zip(*coords)
        center_x = (max(x) + min(x)) / 2  # x 중심점
        center_y = (max(y) + min(y)) / 2  # y 중심점
        pivot = (center_x, center_y)
        # origin = np.array(coords[0])
        coords = np.array(coords) - pivot

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rotated_coords = np.dot(coords, rotation_matrix)
        rotated_coords += pivot

        # print(rotated_coords)
        # rotated_points = np.array(rotated_coords)
        # hull2 = ConvexHull(rotated_points)
        # # plt.plot(points[:, 0], points[:, 1], 'o')
        # for simplex in hull2.simplices:
        #     plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], 'r-')

        # plt.axis('equal')
        # plt.show()
        return rotated_coords

    def grid_generate(self, rotated_coords):
        # rotated_coords = rotated_coords.transpose()
        x = [coord[0] for coord in rotated_coords]
        y = [coord[1] for coord in rotated_coords]

        start_x = min(x)
        start_y = min(y)

        if max(x) - min(x) < grid_size or max(y) - min(y) < grid_size:
            rotated_coords = rotated_coords.tolist()
            return [rotated_coords]
        dx = grid_size  # x 좌표 간격
        dy = grid_size  # y 좌표 간격
        grid_coords = []
        # 그리드 생성
        for i in range(int((max(x) - start_x) / dx) + 1):
            for j in range(int((max(y) - start_y) / dy) + 1):
                # 그리드 좌표 계산
                x1 = start_x + i * dx
                x2 = start_x + (i + 1) * dx
                y1 = start_y + j * dy
                y2 = start_y + (j + 1) * dy

                # 그리드 좌표 저장
                grid_coords.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        grid_coords = np.array(grid_coords)
        # for grid in grid_coords:
        #     hull3 = ConvexHull(grid)
        #     for simplex in hull3.simplices:
        #         plt.plot(grid[simplex, 0], grid[simplex, 1], 'r-')

        # plt.axis('equal')
        # plt.show()
        return grid_coords

    def grid_rotate_to_origin(self, rotated_coords, grid_coords, theta):
        # grid_points = np.array(grid_coords)
        # hull = ConvexHull(points)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # for simplex in hull.simplices:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        x, y = zip(*rotated_coords)
        center_x = (max(x) + min(x)) / 2  # x 중심점
        center_y = (max(y) + min(y)) / 2  # y 중심점
        pivot = (center_x, center_y)
        # grid_coords = np.array(grid_coords)
        rotated_points = []  # 회전된 좌표를 저장할 리스트
        for grid in grid_coords:
            grid_points = np.array(grid) - pivot

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            rotated_coords = np.dot(grid_points, rotation_matrix)
            rotated_coords += pivot
            rotated_coords = np.array(rotated_coords)
            # print(rotated_coords)

            hull3 = ConvexHull(rotated_coords)
            for simplex in hull3.simplices:
                plt.plot(rotated_coords[simplex, 0], rotated_coords[simplex, 1], 'b-')

            # rotated_points.append(rotated)
            rotated_points.append(rotated_coords.tolist())

        # rotated_coords.append(rotated_coords)
        # plt.axis('equal')
        # plt.show()
        print(rotated_points)
        return rotated_points

    def select_intersecting_grids(self, rotated_points, x, y):
        selected_grids = []
        for grid in rotated_points:
            grid_points = np.array(grid)
            # x, y 좌표를 이용하여 그리드와 겹치는 영역이 있는지 확인합니다.
            inside = np.logical_and(x > grid_points[:, 0].min(), x < grid_points[:, 0].max())
            inside = np.logical_and(inside, y > grid_points[:, 1].min())
            inside = np.logical_and(inside, y < grid_points[:, 1].max())
            # 겹치는 영역이 있으면 그리드를 선택합니다.
            if np.sum(inside) > 0:
                selected_grids.append(grid)
        return selected_grids

    def plot_3d(self, n, fig, data_3d, rotated_points):
        cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow',
                'chartreuse',
                'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
                'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
                'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
                'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
                'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
                'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
                'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
                'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy',
                'mediumslateblue',
                'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
        max_heights = []
        for i, grid in enumerate(rotated_points):
            max_height = -np.inf
            for vertex in grid:
                dist = np.linalg.norm(data_3d[:, :2] - vertex, axis=1)
                closest_idx = np.argmin(dist)
                height = data_3d[closest_idx, 2]
                if height > max_height:
                    max_height = height
            max_heights.append(max_height)
            # print(max_heights)
            # print(max(max_heights))
            # max_value = max(max_heights)
            # print(max_value)
            # print(grid)
            grid_matrix = np.zeros((len(grid), 3))

            for j, vertex in enumerate(grid):
                # print(grid_matrix)
                # print(vertex)
                grid_matrix[j, :] = [vertex[0], vertex[1], max_heights[i]]
                for k, l in zip(grid_matrix[:, 0], grid_matrix[:, 1]):
                    if k != 0 or l != 0:
                        plt.scatter(k, l, c='k', s=2)
            # plt.scatter(x, y, c=label, s=20, alpha=0.5, cmap='rainbow')

            new_arr = []
            for i in range(0, len(grid_matrix), 2):
                if i + 1 < len(grid_matrix):
                    new_arr.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
                else:
                    new_arr.append(np.array([grid_matrix[i], grid_matrix[0]]))

            new_arr2 = []
            for i in range(1, len(grid_matrix), 2):
                if i + 1 < len(grid_matrix):
                    new_arr2.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
                else:
                    new_arr2.append(np.array([grid_matrix[i], grid_matrix[0]]))

            surface_points = []
            for arr in new_arr:
                vertices1 = [(x, y, 0) for x, y, _ in arr]
                vertices2 = [(x, y, z) for x, y, z in arr]
                vertices = np.vstack((vertices1, vertices2))
                surface_points += vertices.tolist()
                x, y, z = zip(*vertices)
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n],
                                        showscale=False))

            for arr in new_arr2:
                vertices1 = [(x, y, 0) for x, y, _ in arr]
                vertices2 = [(x, y, z) for x, y, z in arr]
                vertices = np.vstack((vertices1, vertices2))
                surface_points += vertices.tolist()
                x, y, z = zip(*vertices)
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n],
                                        showscale=False))

            vertices1 = [(x, y, 0) for x, y, _ in grid_matrix]
            vertices2 = [(x, y, z) for x, y, z in grid_matrix]
            vertices = np.vstack((vertices1, vertices2))
            surface_points += vertices.tolist()
            x, y, z = zip(*vertices1)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
            x, y, z = zip(*vertices2)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
        return surface_points

class DataAnalysis:
    def __init__(self, data, eps, min_samples, algorithm, city, grid_size):
        self.data = np.array(data)
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.city = city
        self.grid_size = grid_size

    def process_data(self):
        data_3d = self.data[:, :3]
        data_2d = self.data[:, :2]

        label = self.DBSCAN(self.data, self.eps, self.min_samples, self.algorithm, self.city)

        label = pd.DataFrame(label)
        label = label.to_numpy()

        data_label = np.append(data_3d, label, axis=1)
        df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
        df_np = df_pd.to_numpy()

        x = data_2d[:, 0]
        y = data_2d[:, 1]

        # fig = go.Figure()

        cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow',
                'chartreuse',
                'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
                'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
                'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
                'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
                'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
                'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
                'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
                'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy',
                'mediumslateblue',
                'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']

        for n in range(int(max(df_pd['label'])) + 1):
            building = Building(n, df_pd, cmap, self.grid_size)
            building.process()

    def DBSCAN(self, data, eps, min_samples, algorithm, city):
        cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow',
                'chartreuse',
                'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
                'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
                'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
                'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
                'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
                'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
                'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
                'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy',
                'mediumslateblue',
                'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
        # cmap = ['navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy' ,'navy', 'navy', 'navy', 'navy']
        data = np.array(data)
        data_2d = data[:, :2]
        # start = time.time()
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data_2d)
        label = dbscan.labels_
        # print("time: ", time.time() - start)
        label = pd.DataFrame(label)
        label = label.to_numpy()
        data_label = np.append(data, label, axis=1)
        df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
        df_np = df_pd.to_numpy()
        np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
        for n in range(int(max(df_pd['label']))):
            plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                        s=2, alpha=0.5)
        plt.axis('equal')
        plt.title(f'{algorithm}_{city}')
        # plt.savefig(f'plot_{algorithm}_{city}.png')
        # plt.show()
        return label

eps = 18
min_samples = 4
algorithm = "DBSCAN"
city = "portland"
grid_size = 1000

# Create an instance of DataAnalysis and call process_data
data = pd.read_csv('../input/filtered_points_portland.csv', delimiter=',') # 테스트용 가벼운 데이터
data = np.array(data)
data_3d = data[:, :3]
data_2d = data[:, :2]
label = function.DBSCAN(data, eps, min_samples, algorithm, city)
label = pd.DataFrame(label)
label = label.to_numpy()
data_label = np.append(data_3d, label, axis=1)
df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
df_np = df_pd.to_numpy()
x = data_2d[:, 0]
y = data_2d[:, 1]
result = []
result2 = []
select_coord = []

fig = go.Figure()

point1 = np.array([753674, 1379505, 420])
point_portland = np.array([754396, 1380805, 280])
point2 = np.array([753953, 1380978, 100])
point_portland_2 = np.array([758918, 1385675, 280])
point3 = np.array([753503, 1380555, 130])
r = 40

cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']

analysis = DataAnalysis(data, eps, min_samples, algorithm, city, grid_size)
analysis.process_data()

fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
# fig.show()

plt.show()


