import math
import heapq
import numpy as np


class AStarPlanner:
    def __init__(self):
        pass

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def _init_planning(self, ox, oy, resolution=1.0, rr=1.0, save_process=False):
        """
                Initialize grid map for a star planning

                ox: x position list of Obstacles [m]
                oy: y position list of Obstacles [m]
                resolution: grid resolution [m]
                rr: robot radius[m]
                """

        self.resolution = resolution
        self.rr = rr
        self.save_process = save_process

        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    def planning(self, sx, sy, gx, gy, ox, oy, resolution=1.0, rr=1.0, save_process=False):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        # print("planning...")
        self._init_planning(ox, oy, resolution, rr, save_process)

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        searchx, searchy = [], []

        while True:
            if len(open_set) == 0:
                # print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # show graph
            if self.save_process:  # pragma: no cover
                searchx.append(self.calc_grid_position(current.x, self.min_x))
                searchy.append(self.calc_grid_position(current.y, self.min_y))

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Found goal!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        pathx, pathy = self.calc_final_path(goal_node, closed_set)

        return pathx, pathy, searchx, searchy

    def planning2(self, sx, sy, gx, gy, ox, oy, resolution=1.0, rr=1.0, save_process=False):
        """
        A star path search with only searching the closest nine grids

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        # print("planning...")
        self._init_planning(ox, oy, resolution, rr, save_process)

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        searchx, searchy = [], []

        while True:
            if len(open_set) == 0:
                # print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Found goal!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            new_open_set = dict()
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 0, c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    new_open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        new_open_set[n_id] = node
                    else:
                        new_open_set[n_id] = open_set[n_id]
            open_set = new_open_set
        #             open_set[c_id] = current

        pathx, pathy = self.calc_final_path(goal_node, closed_set)

        return pathx, pathy, searchx, searchy

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # out-of-boundary check
        if node.x >= len(self.obstacle_map) or \
                node.y >= len(self.obstacle_map[0]):
            return False
        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


class GridAStarPlanner:
    class Node:
        def __init__(self, x, y, g_cost, h_cost, parent):
            self.x = x
            self.y = y
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.parent = parent

        def f_cost(self):
            return self.g_cost + self.h_cost

        def __lt__(self, other):
            return self.f_cost() < other.f_cost()

    def __init__(self):
        pass

    def planning(self, matrix, start_x, start_y, target_x, target_y, simple=False):
        # define allowed moves (up, down, left, right, up-left, up-right, down-left, down-right)
        # moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        moves = [(1,0), (0,1), (-1,0), (0,-1), (-1,-1), (-1,1), (1,-1), (1,1)]
        # moves = [(0,1), (1,0), (0,-1), (-1,0), (-1,-1), (1,-1), (-1,1), (1,1)]

        # initialize start node and open list

        start_node = self.Node(start_x, start_y, 0, 0, None)
        open_list = []
        heapq.heappush(open_list, start_node)
        # initialize closed list
        closed_list = set()
        # search for path
        while open_list:
            current_node = heapq.heappop(open_list)
            if simple:
                open_list = []
            # check if current node is target node
            if current_node.x == target_x and current_node.y == target_y:
                # construct path by following parent nodes back to start node
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return list(reversed(path))
            # generate neighbor nodes
            for move in moves:
                x = current_node.x + move[0]
                y = current_node.y + move[1]
                # check if neighbor node is valid and not in closed list
                if 0 <= x < len(matrix[0]) and 0 <= y < len(matrix) and matrix[y][x] == 0. and (
                x, y) not in closed_list:
                    if simple:
                        g_cost = 0.
                    else:
                        g_cost = current_node.g_cost + 1

                    h_cost = np.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)  # Euler distance heuristic

                    neighbor_node = self.Node(x, y, g_cost, h_cost, current_node)
                    # check if neighbor node is in open list
                    for open_node in open_list:
                        if neighbor_node.x == open_node.x and neighbor_node.y == open_node.y and neighbor_node.f_cost() >= open_node.f_cost():
                            break
                    else:
                        heapq.heappush(open_list, neighbor_node)
            # add current node to closed list
            closed_list.add((current_node.x, current_node.y))
        # target not found
        return None





