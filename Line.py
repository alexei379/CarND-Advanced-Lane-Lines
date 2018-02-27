import numpy as np
import itertools

class Line:
    avg_over_last_n = 7
    avg_over_last_n_weights = list(range(1, avg_over_last_n + 1))
    max_base_x_shift = 100
    min_curve_radius_m = 400

    def __init__(self):
        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []

        # recently removed line pixels in case we need to revert
        self.allx_removed = None
        self.ally_removed = None

        # recent & avg starting position of the line
        self.starting_x = np.array([])
        self.starting_x_avg = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = np.array([])
        self.radius_of_curvature_avg = None

    def add_and_get_radius(self, possible_radius_m):
        if self.radius_of_curvature.size >= self.avg_over_last_n:
            if possible_radius_m < self.min_curve_radius_m:
                possible_radius = self.radius_of_curvature_avg
            self.radius_of_curvature = np.append(self.radius_of_curvature[-self.avg_over_last_n + 1:], possible_radius_m)
            self.radius_of_curvature_avg = np.average(self.radius_of_curvature, None, self.avg_over_last_n_weights)
        else:
            self.radius_of_curvature = np.array([possible_radius_m for i in range(0, self.avg_over_last_n)])
            self.radius_of_curvature_avg = possible_radius_m
        return int(self.radius_of_curvature_avg)


    def add_and_get_starting_x(self, possible_x):
        if self.starting_x.size >= self.avg_over_last_n:
            if abs(self.starting_x_avg - possible_x) > self.max_base_x_shift:
                possible_x = self.starting_x_avg
            self.starting_x = np.append(self.starting_x[-self.avg_over_last_n + 1:], possible_x)
            self.starting_x_avg = np.average(self.starting_x, None, self.avg_over_last_n_weights)
        else:
            self.starting_x = np.array([possible_x for i in range(0, self.avg_over_last_n)])
            self.starting_x_avg = possible_x
        return int(self.starting_x_avg)

    def append_x_y_points(self, x, y):
        if len(self.allx) >= self.avg_over_last_n:
            self.allx_removed = self.allx[0]
            self.allx = self.allx[-self.avg_over_last_n + 1:]
            self.allx.append(x)

            self.ally_removed = self.ally[0]
            self.ally = self.ally[-self.avg_over_last_n + 1:]
            self.ally.append(y)
        else:
            self.allx = [x for i in range(0, self.avg_over_last_n)]
            self.ally = [y for i in range(0, self.avg_over_last_n)]
        return [list(itertools.chain.from_iterable(self.allx)), list(itertools.chain.from_iterable(self.ally))]

    def pop_bad_x_y_points(self):
        del self.allx[-1]
        self.allx.insert(0, self.allx_removed)

        del self.ally[-1]
        self.ally.insert(0, self.ally_removed)