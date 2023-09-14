import function
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import sqlite3


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
data = pd.read_csv('../input/filtered_points_portland.csv', delimiter=',') # 테스트용 가벼운 데이터
#
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

# #data = pd.read_csv(f'filtered_{algorithm}_{city}.csv')
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
for n in range(int(max(df_pd['label'])) + 1):
    n0 = df_pd[df_pd['label'] == n]
    n_xy = n0[['x', 'y']]
    plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n], s=2, alpha=0.5)
    theta = function.pca(n_xy)
    coords = function.matrix_rotate(theta, n_xy)

    new_data = {
        'n': n,
        'coords': coords,
        'theta': theta
    }

    result_df = result_df.append(new_data, ignore_index=True)

    #print(coords)
    # function.convex_hull_plot(coords)
    rotated_coords = function.grid_rotate(coords, theta)
    grid_coords = function.grid_generate(rotated_coords, grid_size)
    rotated_points = function.grid_rotate_to_origin(rotated_coords, grid_coords, -theta)
    #print(rotated_points)
    selected_grids = function.select_intersecting_grids(rotated_points, x, y)
    #print(selected_grids)
    function.plot_3d(n, fig, data_3d, selected_grids)
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
    #print(result)
    #max_value = function.plot_3d(n, fig, data_3d, selected_grids)
    #max_val = function.max_find(data_3d, coords)
    # print(max_val)
    # print(surface_points)
    # print('\n')
    #new_coords = [[coord[0], coord[1], 0] for coord in coords] + [[coord[0], coord[1], max_val] for coord in coords]
    # print(new_coords)
    # point1 = np.array([754396, 1380805, 280])
#
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
#print(result_df)
fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
# fig.show()
plt.axis('equal')
# plt.title(f'{algorithm}_{city}')
#plt.savefig(f'plot_{algorithm}_{city}.png')
# plt.show()

conn = sqlite3.connect('mydata.db')
c = conn.cursor()

# 테이블 생성
# c.execute('''CREATE TABLE result (n int, coords text, theta text)''')

# 각 행을 데이터베이스에 저장
# for index, row in result_df.iterrows():
#     c.execute("INSERT INTO result (n, coords, theta) VALUES (?, ?, ?)", (row['n'], str(row['coords']), str(row['theta'])))
#
# conn.commit()
c.execute("SELECT * FROM result")
rows = c.fetchall()
for row in rows:
    print(row)

conn.close()