"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt

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

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

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

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
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

        # step_size = 1
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


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    x_range = 51  # size of background
    y_range = 31
    obs = {(24, 30), (40, 3), (30, 26), (0, 20), (0, 7), (20, 7), (0, 10), (13, 30), (18, 30), (33, 30), (50, 11), (4, 0), (9, 0), (50, 22), (40, 13), (30, 0), (35, 0), (15, 30), (40, 0), (30, 23), (20, 30), (0, 17), (20, 9), (0, 4), (20, 4), (50, 1), (50, 12), (29, 30), (34, 30), (50, 27), (6, 0), (30, 29), (40, 10), (11, 0), (16, 0), (14, 15), (30, 16), (37, 0), (42, 0), (0, 30), (20, 14), (0, 1), (43, 30), (50, 2), (48, 30), (50, 17), (9, 30), (50, 28), (40, 7), (30, 30), (11, 15), (13, 0), (18, 0), (0, 27), (39, 0), (20, 3), (44, 0), (18, 15), (0, 14), (49, 0), (50, 7), (39, 30), (44, 30), (50, 18), (40, 4), (5, 30), (30, 27), (10, 30), (25, 30), (0, 21), (46, 30), (0, 24), (15, 0), (20, 0), (25, 0), (0, 11), (46, 0), (50, 8), (50, 23), (40, 14), (40, 1), (30, 20), (0, 18), (6, 30), (21, 30), (20, 10), (26, 30), (0, 5), (20, 5), (1, 0), (0, 8), (22, 0), (27, 0), (50, 13), (32, 0), (30, 15), (50, 24), (40, 11), (35, 30), (40, 30), (30, 17), (20, 15), (1, 30), (0, 2), (22, 30), (50, 3), (3, 0), (8, 0), (49, 30), (50, 14), (29, 0), (10, 15), (34, 0), (50, 29), (40, 8), (31, 30), (30, 18), (36, 30), (0, 28), (17, 15), (20, 12), (0, 15), (2, 30), (50, 4), (50, 19), (5, 0), (45, 30), (10, 0), (50, 30), (31, 0), (30, 24), (40, 5), (36, 0), (0, 22), (41, 0), (11, 30), (0, 25), (16, 30), (47, 30), (13, 15), (20, 1), (0, 12), (50, 9), (50, 20), (40, 15), (7, 0), (40, 2), (30, 21), (12, 0), (17, 0), (0, 19), (38, 0), (20, 11), (43, 0), (0, 6), (7, 30), (48, 0), (12, 30), (27, 30), (20, 6), (0, 9), (32, 30), (50, 10), (50, 25), (14, 30), (40, 12), (30, 22), (19, 15), (0, 16), (41, 30), (14, 0), (20, 8), (19, 0), (0, 3), (24, 0), (45, 0), (50, 0), (23, 30), (28, 30), (50, 15), (50, 26), (40, 9), (30, 28), (16, 15), (30, 19), (0, 29), (15, 15), (37, 30), (20, 13), (50, 21), (0, 0), (42, 30), (21, 0), (26, 0), (50, 5), (47, 0), (3, 30), (8, 30), (50, 16), (40, 6), (30, 25), (0, 23), (12, 15), (0, 26), (20, 2), (17, 30), (0, 13), (38, 30), (2, 0), (50, 6), (23, 0), (28, 0), (33, 0), (4, 30), (19, 30)}
    step_size = 1

    astar = AStar(s_start, s_goal, "euclidean", obs, x_range, y_range, step_size)
    plot = Plotting(s_start, s_goal, obs, x_range, y_range)

    path, visited = astar.searching(step_size)
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
    plt.show()

