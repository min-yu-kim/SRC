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
    for node in tree:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'k-', linewidth=1)

    # Plot start and goal
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'bo', markersize=10, label='Goal')

    # Plot path if available
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.legend()
    plt.show()


# Example usage
start = Node(2, 2)
goal = Node(9, 9)

# Define polygon obstacles
obstacle1 = Polygon([(2, 3), (4, 7), (7, 5), (5, 2)])
obstacle2 = Polygon([(8, 2), (10, 6), (13, 4), (11, 1)])
obstacles = [obstacle1, obstacle2]

x_max = 15
y_max = 15
step_size = 0.5
max_iterations = 1000

tree = rrt(start, goal, obstacles, x_max, y_max, step_size, max_iterations)

if tree:
    print("Path found.")
    # Reconstruct path
    current_node = tree[-1]  # Access the last node in the tree
    path = [current_node]
    while current_node.parent is not None:
        path.append(current_node.parent)
        current_node = current_node.parent
    path.reverse()

    plot_rrt(tree, start, goal, obstacles, path)
else:
    print("No path found.")
