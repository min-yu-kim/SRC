# def make_pos(point1, point2):
#     t = np.linspace(0, 1, 100)
#     x = point1[0] + t * (point2[0] - point1[0])
#     pos_list = []
#     pos_index = []
#     for i in range(len(x)):
#         pos = point1 + (point2 - point1) * (i + 1) / (len(x))
#         pos_index.append(i+1)
#         pos_list.append(pos)
#     return [pos_list, pos_index]
#
# def is_collision(new_coords, r, pos_list):
#     collision = False
#     crush_pos = []
#     faces = [(0, 1, 2, 3),
#              (4, 5, 6, 7),
#              (0, 1, 5, 4),
#              (2, 3, 7, 6),
#              (1, 2, 6, 5),
#              (0, 3, 7, 4)]
#     for pos in pos_list:
#         for face in faces:
#             p1 = np.array(new_coords[face[0]])
#             p2 = np.array(new_coords[face[1]])
#             p3 = np.array(new_coords[face[2]])
#             v1 = np.array(p2) - np.array(p1)
#             v2 = np.array(p3) - np.array(p1)
#             n = np.cross(v1, v2)
#             dist = np.dot(n, np.array(pos) - p1) / np.linalg.norm(n)
#             # print(dist)
#             if abs(dist) <= 2 * r:
#                 collision = True
#                 if pos.tolist() not in crush_pos:
#                     crush_pos.append(pos.tolist())
#     return [collision, crush_pos]
#
# def plot_corridor2(r, fig, crush_pos):
#     for pos in crush_pos:
#         theta = np.linspace(0, 2 * np.pi, 100)
#         phi = np.linspace(0, np.pi, 100)
#         x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
#         y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
#         z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
#         colors = np.zeros_like(z_sphere)
#         color_val = "red"
#         fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
#                                       colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))
#
# import numpy as np
# import plotly.graph_objects as go
# new_coords = [[753809.6626206175, 1380455.7890325333, 0], [753809.5400000003, 1380455.75, 0], [753809.4066456945, 1380456.168932259, 0], [753809.5292663118, 1380456.2079647924, 0], [753809.6626206175, 1380455.7890325333, 288.32], [753809.5400000003, 1380455.75, 288.32], [753809.4066456945, 1380456.168932259, 288.32], [753809.5292663118, 1380456.2079647924, 288.32]]
# # new_coords = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]
#
# point1 = np.array([754396, 1380805, 280])
# point2 = np.array([753953, 1380978, 300])
# r = 10
#
# pos = make_pos(point1, point2)
# # print(pos[1])
# # print(pos[0])
# pos_list = pos[0]
# # print(pos_list)
# crush = is_collision(new_coords, r, pos_list)
# crush_pos = crush[1]
# # print(crush[0])
# print(crush[1])
# print(len(crush[1]))
#
# fig = go.Figure()
# plot_corridor2(r, fig, crush_pos)
#
# fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
# fig.show()

for i in range(4):
    list = [[i]]
    print(list)

result = []
for i in range(4):
    temp_list = [[i]]
    result += temp_list
print(result)

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt

# generate some random 3D coordinates
np.random.seed(1)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull

# 샘플 데이터 생성
coords = np.random.rand(10, 3)

# 3D Convex Hull 계산
hull = ConvexHull(coords[:, :2])

# 시각화
fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], s=20, color='red')
ax.add_patch(Polygon(coords[hull.vertices, :2], fill=None, alpha=1, edgecolor='red'))

plt.show()

