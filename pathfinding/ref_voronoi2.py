import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

# 비행금지 구역 다각형 정의
forbidden_polygon = np.array([(2, 2), (2, 6), (6, 6), (6, 2)])

# Voronoi 다이어그램 생성을 위한 점 생성
points = np.random.uniform(0, 10, (50, 2))  # 50개의 무작위 점 생성

# 비행금지 구역 내부의 점 제거
def point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

points = [point for point in points if not point_inside_polygon(point, forbidden_polygon)]

# Voronoi 다이어그램 생성
vor = Voronoi(points)

# Voronoi 다이어그램 플롯
plt.figure(figsize=(8, 8))

# 비행금지 구역 플롯
plt.fill(*zip(*forbidden_polygon), 'gray', alpha=0.5)

# Voronoi 다이어그램 엣지 플롯
for region in vor.regions:
    if -1 not in region and len(region) > 0:
        plt.fill(*zip(*[vor.vertices[i] for i in region]), alpha=0.5)

# Voronoi 다이어그램 점 플롯
plt.plot(points[:, 0], points[:, 1], 'ko')

# 그래프 출력
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
