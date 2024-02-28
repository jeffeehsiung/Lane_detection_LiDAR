# data loading
import numpy as np

from scipy.spatial import distance
from scipy.optimize import fsolve
from sklearn.metrics import mean_squared_error



class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))
    
    def cost(up_lane_coeffs, down_lane_coeffs, x_range):
        """
        Calculate the cost as the mean squared error between the true lane width (assumed constant)
        and the measured lane width across a range of x-values.

        Args:
        - up_lane_coeffs (array): Coefficients for the polynomial representing the upper lane boundary.
        - down_lane_coeffs (array): Coefficients for the polynomial representing the lower lane boundary.
        - x_range (array): Range of x-values to evaluate the lane width.

        Returns:
        - float: The cost calculated as mean squared error.
        """
        # Define polynomial functions for the upper and lower lanes
        poly_up = np.poly1d(up_lane_coeffs)
        poly_down = np.poly1d(down_lane_coeffs)

        # Calculate derivatives for the lower lane polynomial
        poly_down_deriv = np.polyder(poly_down)

        # Vectorized calculation of y-values for lower and upper lanes
        y_down = poly_down(x_range)
        y_deriv_down = poly_down_deriv(x_range)

        # Initialize measured intervals
        interval_measured = []

        # Calculate perpendicular distances for each x in x_range
        for x, y, dy in zip(x_range, y_down, y_deriv_down):
            # Define the perpendicular line at (x, y) of the lower lane
            perp_slope = -1 / dy if dy != 0 else np.inf
            y_intercept = y - perp_slope * x

            # Define a function for the intersection with the upper lane
            def intersection_fun(x_intersect):
                return perp_slope * x_intersect + y_intercept - poly_up(x_intersect)

            # Use fsolve to find the intersection point
            x_intersect = fsolve(intersection_fun, x)[0]

            # Calculate the perpendicular distance
            y_intersect = perp_slope * x_intersect + y_intercept
            dist = distance.euclidean([x_intersect, y_intersect], [x, y])
            interval_measured.append(dist)

        # True interval is assumed to be constant (e.g., 3 meters)
        interval_truth = np.full_like(x_range, 3)

        # Calculate MSE as the cost
        cost = mean_squared_error(interval_truth, interval_measured)
        return cost
