import function
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import sqlite3

import random
import math
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None


def get_distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def get_nearest_node(tree, point):
    nearest_node = tree[0]
    min_distance = get_distance(nearest_node, point)
    for node in tree:
        distance = get_distance(node, point)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node


def steer(nearest_node, sampled_point, step_size):
    distance = get_distance(nearest_node, sampled_point)
    if distance >= step_size:
        theta = math.atan2(sampled_point.y - nearest_node.y,
                           sampled_point.x - nearest_node.x)
        new_x = nearest_node.x + step_size * math.cos(theta)
        new_y = nearest_node.y + step_size * math.sin(theta)
        return Node(new_x, new_y)
    else:
        return sampled_point


def is_collision_free(nearest_node, new_node, obstacles):
    new_point = Point(new_node.x, new_node.y)
    for obstacle in obstacles:
        if obstacle.contains(new_point):
            return False
    return True


def rrt(start, goal, obstacles, x_max, y_max, step_size, max_iterations):
    tree = [start]
    for i in range(max_iterations):
        if random.random() < 0.1:
            sampled_point = goal
        else:
            sampled_x = random.random() * x_max
            sampled_y = random.random() * y_max
            sampled_point = Node(sampled_x, sampled_y)
        nearest_node = get_nearest_node(tree, sampled_point)
        new_node = steer(nearest_node, sampled_point, step_size)
        if is_collision_free(nearest_node, new_node, obstacles):
            new_node.parent = nearest_node
            tree.append(new_node)
            if get_distance(new_node, goal) < step_size:
                goal.parent = new_node
                return tree  # Return the entire tree
    return None


def plot_rrt(tree, start, goal, obstacles, path=None):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='r', edgecolor='none')

    # Plot tree edges
    # for node in tree:
    #     if node.parent:
    #         ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'k-', linewidth=1)

    # Plot start and goal
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'bo', markersize=10, label='Goal')

    # Plot path if available
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
        plt.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
        print(path_x)
        print(path_y)
    ax.set_aspect('equal', 'box')
    # ax.set_xlim(0, x_max)
    # ax.set_ylim(0, y_max)
    # ax.legend()
    plt.show()

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
refining_height = 280
city = "portland"
# data = pd.read_csv(f'points_{city}.csv', delimiter=',') # 실제 데이터
data = pd.read_csv('../../input/filtered_points_portland.csv', delimiter=',') # 테스트용 가벼운 데이터
sampled_obstacles = []

# ### refining
# filtered_points = function.refining(data, refining_height)

# filtered_points.to_csv(f'filtered_points_{refining_height}_{city}.csv', index=False)
# print("refining finish")
# print("\n")
#
# # nb_neighbors는 한 점의 근처 이웃의 수. 이 값이 높을수록 이상치 탐지 능력은 상승.
# # std_ratio는 표준 편차 비율. 이 값은 일종의 허용 한계값으로 이 값보다 큰 이상치가 식별.
# # 이 값을 높일수록 이상치로 간주되는 데이터 증가.
# nb_neighbors = 40
# std_ratio = 8
# inliers_df = function.remove_outlier(filtered_points, nb_neighbors, std_ratio)
# # print(inliers_df)
# inliers_df.to_csv(f"filtered_{city}.csv", index=False)
# print("remove outlier finish")
# print("\n")

###clustering
# data = pd.read_csv(f"filtered_{city}.csv")
# # data = inliers_df
algorithm = "DBSCAN"
#
# cluster = 10
# function.kmeans(data, cluster, algorithm, city)
eps = 18
min_samples = 4
function.DBSCAN(data, eps, min_samples, algorithm, city)
# function.MiniBatchKMeans(data, cluster, algorithm, city)
# function.GaussianMixture(data, cluster, algorithm, city)
# OPTICS_sample = 3
# function.OPTICS(3, data, algorithm, city)
# # function.MeanShift(data, algorithm, city)
# print("clustering finish")
# print("\n")

###주석 살리기
point1 = np.array([753674, 1379505, 420])
point_portland = np.array([754396, 1380805, 280])
point2 = np.array([753953, 1380978, 100])
point_portland_2 = np.array([758918, 1385675, 280])
point3 = np.array([753503, 1380555, 130])
r = 40
grid_size = 1000
algorithm = "DBSCAN"

# data = pd.read_csv(f'filtered_{algorithm}_{city}.csv')
# data = pd.read_csv('DBSCAN_portland.csv')
data = np.array(data)
data_3d = data[:, :3]
data_2d = data[:, :2]
# label = data[:, 3]
label = function.DBSCAN(data, eps, min_samples, algorithm, city)
label = pd.DataFrame(label)
label = label.to_numpy()
data_label = np.append(data_3d, label, axis=1)
df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
df_np = df_pd.to_numpy()
x = data_2d[:, 0]
y = data_2d[:, 1]
# print(x)
fig = go.Figure()
# # new_coord = [[753809.6626206175, 1380455.7890325333, 0], [753809.5400000003, 1380455.75, 0], [753809.4066456945, 1380456.168932259, 0], [753809.5292663118, 1380456.2079647924, 0], [753809.6626206175, 1380455.7890325333, 288.32], [753809.5400000003, 1380455.75, 288.32], [753809.4066456945, 1380456.168932259, 288.32], [753809.5292663118, 1380456.2079647924, 288.32]]
result = []
result2 = []
select_coord = []
result_df = pd.DataFrame(columns=['n', 'coords', 'theta'])

# print(label)
# pos_list = function.make_pos(point2, point3)
# pos_list2 = function.make_pos(point1, point3)
# function.plot_corridors(point2, point3, r, fig)
# function.plot_corridors(point1, point3, r, fig)
obstacles = []  # Initialize list to store Polygon obstacles

for n in range(int(max(df_pd['label'])) + 1):
    n0 = df_pd[df_pd['label'] == n]
    n_xy = n0[['x', 'y']]
    plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n], s=2, alpha=0.5)
    theta = function.pca(n_xy)
    coords = function.matrix_rotate(theta, n_xy)
    # print(coords)
    new_data = {
        'n': n,
        'coords': coords,
        'theta': theta
    }

    result_df = result_df.append(new_data, ignore_index=True)

    # print(coords)
    # function.convex_hull_plot(coords)
    rotated_coords = function.grid_rotate(coords, theta)
    grid_coords = function.grid_generate(rotated_coords, grid_size)
    rotated_points = function.grid_rotate_to_origin(rotated_coords, grid_coords, -theta)
    # print(rotated_points)
    # print("\n")
    obstacle_polygon = Polygon(rotated_points[0])  # Assuming the first sublist contains the coordinates
    obstacles.append(obstacle_polygon)
    selected_grids = function.select_intersecting_grids(rotated_points, x, y)
    # print(rotated_coords)
    output_coords = []

    # 주어진 좌표를 원하는 형식으로 변환
    # for i in range(len(rotated_points)):
    #     x = int(rotated_coords[0][i])
    #     y = int(rotated_coords[i][1])
    #     output_coords.append((x, y))

    for sublist in rotated_points:
        for coord in sublist:
            x = int(coord[0])
            y = int(coord[1])
            output_coords.append((x, y))

    # 변환된 좌표 출력
    # print(output_coords)
    #print(selected_grids)
    # function.plot_3d(n, fig, data_3d, selected_grids)

    # print(rotated_points)

    # for obstacle in rotated_points:
        # print(obstacle)
        # print(obstacle[0])
        # print(obstacle + [obstacle[0]])
        # print('\n')
        # sampled_obstacle = function.sample_obstacle(obstacle + [obstacle[0]], sampling_distance=20)
        # sampled_obstacles.append(sampled_obstacle)
    # print(sampled_obstacles)
#
#
    # for selected_grid in selected_grids:
    #     function.convex_hull_plot(selected_grid)
    #     max_val = function.max_find(data_3d, selected_grid)
    # #    #print(max_val2)
    #     select_coords = [[coord[0], coord[1], 0] for coord in selected_grid] + [[coord[0], coord[1], max_val] for coord in selected_grid]
    # #    #print(select_coords)
    #     crush = function.is_collision(select_coords, r, pos_list)
    #     crush_pos = crush[1]
    # #    #print(crush_pos)
    #     result += crush_pos
    #
    #     crush2 = function.is_collision(select_coords, r, pos_list2)
    #     crush_pos2 = crush2[1]
    #     result2 += crush_pos2
    # print(result)
    # max_value = function.plot_3d(n, fig, data_3d, selected_grids)
    # max_val = function.max_find(data_3d, coords)
    # # print(max_val)
    # # print(surface_points)
    # # print('\n')
    # new_coords = [[coord[0], coord[1], 0] for coord in coords] + [[coord[0], coord[1], max_val] for coord in coords]
    # print(new_coords)
    # point1 = np.array([754396, 1380805, 280])

    # print("coords finish")
    #pos = function.make_pos(point2, point3)
    #pos_list = pos[0]
    #crush = function.is_collision(new_coords, r, pos_list)
    #crush_pos = crush[1]
    #function.plot_corridor_crush(r, fig, crush_pos)
        # print(crush_pos)
    #result += crush_pos
#
#
#     #crush2 = function.is_collision(select_coord, r, pos_list2)
#     #crush_pos2 = crush2[1]
#         # print(crush_pos)
#     #result2 += crush_pos2
#
#     # print(normal_pos)
#     # print(crush[1])
#     # function.plot_corridor2(r, fig, crush_pos)
#     # another = pos_list - crush_pos
#     # print(another)
#     # pos2 = function.make_pos(point2, point3)
#     # pos_list2 = pos2[0]
#     # crush2 = function.is_collision(new_coords, r, pos_list2)
#     # crush_pos2 = crush2[1]
#     # function.plot_corridor2(r, fig, crush_pos2)
#
#     # print(len(pos[0]))
#     # print(len(crush[1]))
#     # function.plot_corridors(p1, p2, r, fig, new_coords)
# # p1 = np.array([754396, 1380805, 280])
# # p2 = np.array([753953, 1380978, 300])
# # p3 = np.array([753503, 1380555, 430])
# # r = 40
# print(len(result))
# print(len(result2))
#
# #function.plot_corridors(point1, point3, r, fig)
# function.plot_corridor_crush(r, fig, result)
# function.plot_corridor_crush(r, fig, result2)
# # function.plot_corridor_normal(r, fig, result2)
# # function.plot_corridor(surface_points, p1, p2, r, fig)
# #function.plot_corridor(p1, p2, r, fig)
# #function.plot_corridor(p2, p3, r, fig)
# # function.plot_corridor(surface_points, p2, p3, r, fig)
#
# print(sampled_obstacles)
# sampled_obstacles = np.array(sampled_obstacles)
# print(len(sampled_obstacles))
# print(sampled_obstacles)
# print(rotated_coords_list)

# Example usage
# start = Node(2, 2)
# goal = Node(9, 9)
start = Node(753000, 1380400)
goal = Node(754200, 1381200)
# Define polygon obstacles
# obstacle1 = Polygon([(2, 3), (4, 7), (7, 5), (5, 2)])
# obstacle2 = Polygon([(8, 2), (10, 6), (13, 4), (11, 1)])
# obstacles = [obstacle1, obstacle2]

# print(obstacles)
x_max = 764200
y_max = 1481200
step_size = 10
max_iterations = 1000

tree = rrt(start, goal, obstacles, x_max, y_max, step_size, max_iterations)

if tree:
    print("Path found.")
    # Reconstruct path
    current_node = tree[-1]  # Access the last node in the tree
    path = [current_node]
    print(path)
    while current_node.parent is not None:
        path.append(current_node.parent)
        current_node = current_node.parent
    path.reverse()

    plot_rrt(tree, start, goal, obstacles, path)
else:
    print("No path found.")

plt.show()

