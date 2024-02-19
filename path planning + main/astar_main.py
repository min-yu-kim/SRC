import function
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import random
from shapely.geometry import Polygon, Point
import matplotlib.patches as patches
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import heapq
PI = np.pi

class Env:
    def __init__(self, obs, x_range, y_range):
        self.x_range = x_range  # size of background
        self.y_range = y_range
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = obs

    def update_obs(self, obs):
        self.obs = obs

    # def obs_map(self):
    #     """
    #     Initialize obstacles' positions
    #     :return: map of obstacles
    #     """
    #
    #     x = self.x_range
    #     y = self.y_range
    #     obs = set()
    #
    #     for i in range(x):
    #         obs.add((i, 0))
    #     for i in range(x):
    #         obs.add((i, y - 1))
    #
    #     for i in range(y):
    #         obs.add((0, i))
    #     for i in range(y):
    #         obs.add((x - 1, i))
    #
    #     for i in range(10, 21):
    #         obs.add((i, 15))
    #     for i in range(15):
    #         obs.add((20, i))
    #
    #     for i in range(15, 30):
    #         obs.add((30, i))
    #     for i in range(16):
    #         obs.add((40, i))
    #
    #     print(obs)
    #
    #     return obs

class Plotting:
    def __init__(self, xI, xG, obs, x_range, y_range):
        self.xI, self.xG = xI, xG
        self.env = Env(obs, x_range, y_range)
        self.obs = self.env.obs

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name):
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        # plt.show()

    def animation_lrta(self, path, visited, name):
        self.plot_grid(name)
        cl = self.color_list_2()
        path_combine = []

        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])
            plt.pause(0.2)
            self.plot_path(path[k])
            path_combine += path[k]
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)
        # plt.show()

    def animation_ara_star(self, path, visited, name):
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()

        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])
            self.plot_path(path[k], cl_p[k], True)

        # plt.show()

    def animation_bi_astar(self, path, v_fore, v_back, name):
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)
        self.plot_path(path)
        # plt.show()

    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]
        #
        # plt.plot(self.xI[0], self.xI[1], "bs")
        # plt.plot(self.xG[0], self.xG[1], "gs")
        # plt.plot(obs_x, obs_y, "sk")
        # plt.title(name)
        # plt.axis("equal")

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            # plt.plot(x[0], x[1], color=cl, marker='o')
            # plt.gcf().canvas.mpl_connect('key_release_event',
            #                              lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40
            #
            # length = 15


    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")


    def plot_visited_bi(self, v_fore, v_back):
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        for k in range(max(len_fore, len_back)):
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1], linewidth='3', color='gray', marker='o')
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth='3', color='cornflowerblue', marker='o')

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])



    @staticmethod
    def color_list():
        cl_v = ['silver',
                'wheat',
                'lightskyblue',
                'royalblue',
                'slategray']
        cl_p = ['gray',
                'orange',
                'deepskyblue',
                'red',
                'm']
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        cl = ['silver',
              'steelblue',
              'dimgray',
              'cornflowerblue',
              'dodgerblue',
              'royalblue',
              'plum',
              'mediumslateblue',
              'mediumpurple',
              'blueviolet',
              ]
        return cl

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, obs, x_range, y_range, step_size):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.step_size = step_size

        self.Env = Env(obs, x_range, y_range)  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self, step_size):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s, step_size):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e, step_size):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e, step_size)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e, step_size):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s, step_size):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s, step_size):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        neighbors = []

        for u in self.u_set:
            neighbor = (s[0] + u[0] * step_size, s[1] + u[1] * step_size)
            if not self.is_collision(s, neighbor):
                neighbors.append(neighbor)

        return neighbors

        # return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def planner_main(s_start, s_goal, x_range, y_range, obs, step_size):
    astar = AStar(s_start, s_goal, "euclidean", obs, x_range, y_range, step_size)
    plot = Plotting(s_start, s_goal, obs, x_range, y_range)

    path, visited = astar.searching(step_size)
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


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
grid_size = 100
algorithm = "DBSCAN"
grid_size_bound = 100

s_start = (752800, 1380200)
x_range = 754400
y_range = 1381600
step_size = 10  # 그리드 간격

# 그리드 생성
x_grid = np.arange(s_start[0], s_start[0] + x_range, step_size)
y_grid = np.arange(s_start[1], s_start[1] + y_range, step_size)


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
bound_obstacles = []
for n in range(int(max(df_pd['label'])) + 1):
    n0 = df_pd[df_pd['label'] == n]
    # path planning에서 추가
    n_xy = n0[['x', 'y']]
    x_max = n0['x'].max()
    y_max = n0['y'].max()
    x_min = n0['x'].min()
    y_min = n0['y'].min()
    bound_box = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    # print(bound_box)
    bound_box = np.array(bound_box)
    # print(bound_box)
    # print(bound_box)
    grid_bound_box = function.grid_generate_bound(bound_box, x_grid, y_grid, step_size)

    #
    # selected_grids_bound = function.select_intersecting_grids(grid_bound_box, x_grid, y_grid)
    #
    #
    # print(grid_bound_box)
    # bound_box = np.array(selected_grids_bound)

    bound_obstacles.append(grid_bound_box)



    from scipy.spatial import ConvexHull

    hull = ConvexHull(bound_box)
    for simplex in hull.simplices:
        plt.plot(bound_box[simplex, 0], bound_box[simplex, 1], 'b-')


    # function.get_bounding_box(n_xy)
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
    # print(rotated_coords)
    grid_coords = function.grid_generate(rotated_coords, grid_size)
    rotated_points = function.grid_rotate_to_origin(rotated_coords, grid_coords, -theta)
    # print(rotated_points)
    # obstacle = function.convert_to_xywh(rotated_points)
    # print(obstacle)
    # print(rotated_points)
    # print("\n")
    # obstacle_polygon = Polygon(rotated_points[0])  # Assuming the first sublist contains the coordinates

    selected_grids = function.select_intersecting_grids(rotated_points, x, y)
    # print(selected_grids)
    # print('\n')
    obstacles.append(rotated_points)
    # print(selected_grids)
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

# print(obstacles)


# x_range = (752000, 764200)
# y_range = (1380000, 1481200)
# s_start = (752800, 1380200)
s_goal = (754400, 1381200)
# x_range = 754400  # size of background
# y_range = 1381600
# obs_rectangle = obstacles
# print(obs_rectangle)
# Car(0, 0, 1, 2, 60)
# planner_main(x_start, x_goal)
# plt.show()
# print(obstacles)
# s_start = (5, 5)
# s_goal = (45, 25)

# step_size = 10
# plt.show()

# obs = {(24, 30), (40, 3), (30, 26), (33, 30), (50, 11), (4, 0),
#        (4, 30), (19, 30)}
obs = function.transform_coordinates(bound_obstacles)
planner_main(s_start, s_goal, x_range, y_range, obs, step_size)
plt.show()
# print(obstacles)
# plt.show()

