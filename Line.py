from collections import deque
import numpy as np
import itertools

class Line:
    avg_over_last_n = 5
    avg_over_last_n_weights = list(range(1, avg_over_last_n + 1))
    max_base_x_shift = 100
    min_curve_radius_m = 500

    def __init__(self):
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # recently removed line pixels in case we need to revert
        self.allx_removed = None
        self.ally_removed = None

        # recent & avg starting position of the line
        self.starting_x = None
        self.starting_x_avg = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.radius_of_curvature_avg = None

    def add_and_get_radius(self, possible_radius_m):
        if self.radius_of_curvature is not None:
            if possible_radius_m < self.min_curve_radius_m:
                possible_radius_m = self.radius_of_curvature_avg
            self.radius_of_curvature.append(possible_radius_m)
            self.radius_of_curvature_avg = np.average(self.radius_of_curvature, None, self.avg_over_last_n_weights)
        else:
            self.radius_of_curvature = deque([possible_radius_m for i in range(0, self.avg_over_last_n)], self.avg_over_last_n)
            self.radius_of_curvature_avg = possible_radius_m
        return int(self.radius_of_curvature_avg)


    def add_and_get_starting_x(self, possible_x):
        if self.starting_x is not None:
            if abs(self.starting_x_avg - possible_x) > self.max_base_x_shift:
                possible_x = self.starting_x_avg
            self.starting_x.append(possible_x)
            self.starting_x_avg = np.average(self.starting_x, None, self.avg_over_last_n_weights)
        else:
            self.starting_x = deque([possible_x for i in range(0, self.avg_over_last_n)], self.avg_over_last_n)
            self.starting_x_avg = possible_x
        return int(self.starting_x_avg)

    def append_x_y_points(self, x, y):
        if self.allx is not None:
            self.allx_removed = self.allx.popleft()
            self.allx.append(x)

            self.ally_removed = self.ally.popleft()
            self.ally.append(y)
        else:
            self.allx = deque([x for i in range(0, self.avg_over_last_n)], self.avg_over_last_n)
            self.ally = deque([y for i in range(0, self.avg_over_last_n)], self.avg_over_last_n)
        return [list(itertools.chain.from_iterable(self.allx)), list(itertools.chain.from_iterable(self.ally))]

    def pop_bad_x_y_points(self):
        if self.allx is not None:
            self.allx.appendleft(self.allx_removed)
            self.ally.appendleft(self.ally_removed)
