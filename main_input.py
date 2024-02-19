import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def convex_hull_plot(coords):
    points = np.array(coords)
    hull = ConvexHull(points)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

def sample_obstacle(obstacle, sampling_distance=1):
    # 샘플링된 점들을 저장할 리스트
    sampled_points = []

    for i in range(len(obstacle) - 1):
        x1, y1 = obstacle[i]
        x2, y2 = obstacle[i + 1]

        # 두 점 사이의 거리 계산
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # 샘플링 간격에 따라 점 생성
        num_samples = int(distance / sampling_distance)
        if num_samples > 0:
            x_step = (x2 - x1) / num_samples
            y_step = (y2 - y1) / num_samples
            for j in range(num_samples):
                x = x1 + j * x_step
                y = y1 + j * y_step
                sampled_points.append((x, y))

    return sampled_points


# 장애물 샘플링 예제
obstacle = [(753406, 1380642), (753501, 1380604), (753431, 1380425), (753335, 1380463), (753406, 1380642)]
sampled_obstacle = sample_obstacle(obstacle, sampling_distance=10)

# 장애물과 샘플링된 점 시각화
obstacle_x = [x for x, y in obstacle]
obstacle_y = [y for x, y in obstacle]

sampled_x = [x for x, y in sampled_obstacle]
sampled_y = [y for x, y in sampled_obstacle]

points = np.array(sampled_obstacle)
hull = ConvexHull(points)
# plt.plot(points[:, 0], points[:, 1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(obstacle_x, obstacle_y, 'ro-', label='Obstacle')  # 장애물
plt.plot(sampled_x, sampled_y, 'bo-', label='Sampled Points')  # 샘플링된 점들
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

