# data loading
import numpy as np

from scipy.spatial import distance
from scipy.optimize import fsolve
from sklearn.metrics import mean_squared_error

import warnings
from numpy import RankWarning
warnings.simplefilter('ignore', RankWarning)


class PolynomialRegression:
    def __init__(self, degree=3, coeffs=None, costs=None):
        self.degree = degree
        self.coeffs = coeffs
        # define self cost
        self.costs = costs

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
    
    def cost(self, left_lane_coeffs, right_lane_coeffs, x_range, parallelism_weight=100):
        """
        Calculate the cost by considering both the mean squared error between the true lane width
        and the measured lane width across a range of x-values, and the difference in slopes
        between the two lanes to ensure parallelism.

        Args:
        - left_lane_coeffs (array): Coefficients for the polynomial representing the left lane boundary.
        - right_lane_coeffs (array): Coefficients for the polynomial representing the right lane boundary.
        - x_range (array): Range of x-values to evaluate the lane width.
        - parallelism_weight (float): Weight for the parallelism penalty term.

        Returns:
        - float: The total cost.
        """
        if left_lane_coeffs is None or right_lane_coeffs is None:
            return float('inf')  # Return a high cost to indicate an invalid state
        # Define polynomial functions for the left and right lanes
        poly_left = np.poly1d(left_lane_coeffs)
        poly_right = np.poly1d(right_lane_coeffs)

        # Calculate derivatives for the left and right lane polynomial
        poly_left_deriv = np.polyder(poly_left)
        poly_right_deriv = np.polyder(poly_right)

        # Vectorized calculation of y-values and slopes for left and right lanes
        y_right = poly_right(x_range)
        y_deriv_right = poly_right_deriv(x_range)
        y_deriv_left = poly_left_deriv(x_range)

        # Initialize measured intervals and slope differences
        interval_measured = []
        slope_differences = []

        # Calculate perpendicular distances for each x in x_range
        for x, y, dy_right, dy_left in zip(x_range, y_right, y_deriv_right, y_deriv_left):
            # calculate the shortest distance(orthogonal distance) between the two lines
            dist = np.abs(poly_left(x) - poly_right(x)) / np.sqrt(1 + poly_left_deriv(x)**2)
            interval_measured.append(dist)
            
            # Calculate the slope difference
            slope_difference = np.abs(dy_right - dy_left)
            slope_differences.append(slope_difference)

        # True interval is assumed to be in between 3 to 3.75 meters, or 3
        interval_truth = np.full_like(x_range, (3+3.75)/2)
        # Calculate MSE for intervals as the cost
        cost_intervals = mean_squared_error(interval_truth, interval_measured)

        # Calculate the mean of slope differences as the parallelism penalty
        parallelism_penalty = np.mean(slope_differences)
    
        # Total cost combines interval cost and parallelism penalty
        total_cost = cost_intervals + parallelism_weight * parallelism_penalty
        
        # set the cost
        self.costs = total_cost
        
        return total_cost
