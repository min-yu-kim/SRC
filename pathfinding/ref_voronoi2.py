import numpy as np
import matplotlib.pyplot as plt


def generate_obstacles():
    # 장애물 생성 (다각형의 꼭짓점)
    obstacle1 = [(2, 3), (4, 7), (7, 5), (5, 2)]
    obstacle2 = [(8, 2), (10, 6), (13, 4), (11, 1)]

    return [obstacle1, obstacle2]


def is_point_valid(point, obstacles):
    # 점이 유효한지 확인 (장애물과 충돌하지 않는지)
    for obstacle in obstacles:
        if is_point_inside_polygon(point, obstacle):
            return False
    return True


def is_point_inside_polygon(point, polygon):
    # 점이 다각형 안에 있는지 확인
    x, y = point
    poly = np.array(polygon)
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def generate_random_point():
    # 무작위로 점 생성
    return [np.random.uniform(0, 15), np.random.uniform(0, 8)]


def nearest_neighbor(tree, point):
    # 가장 가까운 이웃 찾기
    distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in tree]
    return tree[np.argmin(distances)]


def new_point(q_near, q_rand, delta):
    # 새로운 점 생성
    direction = np.array(q_rand) - np.array(q_near)
    direction_normalized = direction / np.linalg.norm(direction)
    q_new = np.array(q_near) + delta * direction_normalized
    return q_new.tolist()


def rrt(start, goal, obstacles, max_iter=1000, delta=0.5):
    tree = [start]

    for _ in range(max_iter):
        q_rand = generate_random_point()
        q_near = nearest_neighbor(tree, q_rand)
        q_new = new_point(q_near, q_rand, delta)

        if is_point_valid(q_new, obstacles):
            tree.append(q_new)

            # 만약 목표에 도달하면 경로 반환
            if np.linalg.norm(np.array(q_new) - np.array(goal)) < delta:
                path = [goal]
                current_node = q_new
                while current_node != start:
                    path.append(nearest_neighbor(tree, current_node))
                    current_node = nearest_neighbor(tree, current_node)
                path.reverse()
                return path

    return None


def plot_environment(obstacles, start, goal, path=[]):
    # 환경 그리기
    fig, ax = plt.subplots()

    for obstacle in obstacles:
        obstacle.append(obstacle[0])  # 다각형을 닫기 위해 첫 번째 꼭짓점 추가
        poly = np.array(obstacle)
        ax.plot(poly[:, 0], poly[:, 1], 'r-', linewidth=2)

    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'k-', linewidth=2, label='Path')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    start = [1, 1]
    goal = [14, 7]
    obstacles = generate_obstacles()

    path = rrt(start, goal, obstacles)

    if path:
        print("Path found:", path)
        plot_environment(obstacles, start, goal, path)
    else:
        print("Path not found.")
