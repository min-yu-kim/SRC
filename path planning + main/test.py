import numpy as np
import matplotlib.pyplot as plt

# 입력 좌표 데이터
coordinates = np.array([
    [1, 2],
    [1, 5],
    [2, 5],
    [2, 2]
])

# 좌표 중에서 최소값을 찾아 시작 지점으로 설정
min_x = np.min(coordinates[:, 0])
min_y = np.min(coordinates[:, 1])

# 시작 지점과 범위 설정
s_start = (0, 0)
x_range = 10
y_range = 10
step_size = 3  # 적절한 크기로 조정하여 배경 그리드를 설정

# 그리드 생성
x_grid = np.arange(s_start[0], s_start[0] + x_range, step_size)
y_grid = np.arange(s_start[1], s_start[1] + y_range, step_size)

# 그림 초기화
fig, ax = plt.subplots()

# 전체 배경 그리드를 검은 색으로 그리기
for x in x_grid:
    for y in y_grid:
        plt.plot(x, y, 'ko')

# 각 좌표의 (x, y)에서부터 가장 가까운 그리드 점을 빨간 색으로 그리기
for coord in coordinates:
    x_coord, y_coord = coord

    # 좌표에서 가장 가까운 그리드 점 찾기
    nearest_grid_x = min(filter(lambda x: x <= min_x, x_grid), key=lambda x: abs(x - min_x))
    nearest_grid_y = min(filter(lambda y: y <= min_y, y_grid), key=lambda y: abs(y - min_y))

    plt.plot(nearest_grid_x, nearest_grid_y, 'ro')

# 직사각형 그리기
rect_x = [coordinates[0, 0], coordinates[1, 0], coordinates[2, 0], coordinates[3, 0], coordinates[0, 0]]
rect_y = [coordinates[0, 1], coordinates[1, 1], coordinates[2, 1], coordinates[3, 1], coordinates[0, 1]]
plt.plot(rect_x, rect_y, 'b-')

# 그림 표시
plt.show()
