# data loading
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
from bisect import bisect_right

from PolynomialRegression import *

import warnings
from numpy import RankWarning
warnings.simplefilter('ignore', RankWarning)

class ParallelPolynomialRegression:
    X_left_length = None
    X_right_length = None
    lane_min_samples = None
    
    def __init__(self, degree=3, left_coeffs=None, right_coeffs=None, random_state=None):
        self.degree = degree
        self.left_coeffs = left_coeffs
        self.right_coeffs = right_coeffs
        self.random_state = None
        
    @classmethod
    def set_static_variables(cls, X_left_length, X_right_length, lane_min_samples):
        """
        Set the static variables for the class.

        Args:
            X_left_length (_type_): _description_
            X_right_length (_type_): _description_
            lane_min_samples (_type_): _description_
        """
        cls.X_left_length = X_left_length
        cls.X_right_length = X_right_length
        cls.lane_min_samples = lane_min_samples
    
    def determine_scaler(self, X):
        """
        Determine the scaler based on the length of the data.
        
        Args:
        - X (array): The input data.
        
        Returns:
        - float: The scaler value.
        """
        # Define thresholds and values for the scaler in a sorted order
        thresholds = [3000, 2000, 1000, 700, 500, 400, 300, 200]
        values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5]
        index = bisect_right(thresholds, len(X))
        return values[index] if index < len(thresholds) else 1.0

        
    def fit(self, X, y):
        X_left = X[:ParallelPolynomialRegression.X_left_length]
        X_right = X[ParallelPolynomialRegression.X_left_length:]
        y_left = y[:ParallelPolynomialRegression.X_left_length]
        y_right = y[ParallelPolynomialRegression.X_left_length:]
        
        # if any of the shapes are empty, print a warning
        if len(X_left) == 0 or len(X_right) == 0:
            #  skip the iteration
            print("Warning: X_left or X_right is empty. Skipping iteration.")
            return
     
        iteration = 0
        max_iterations = 100
        lucky_number = 7
        prev_error = float('inf')
        left_lane_coeffs = np.zeros(self.degree + 1)
        right_lane_coeffs = np.zeros(self.degree + 1)
        
        # define scaler from 0.05, 0.1, 0.15, 0.2, to 0.25 depending on the length of the data from above 3000, 2000, 1000, 700, 500 from the left lane
        # define scaler for left lane
        left_scaler = self.determine_scaler(X_left)

        # define scaler for right lane
        right_scaler = self.determine_scaler(X_right)
        
        poly_regression = PolynomialRegression(self.degree)

        while iteration <= max_iterations:
            try:
                if len(X_left) >=  ParallelPolynomialRegression.lane_min_samples:
                    # Use random sampling for data points in each grid cell
                    idx = np.random.randint(len(X_left), size= int(max((left_scaler * len(X_left)), ParallelPolynomialRegression.lane_min_samples)))
                    X_left_rd = X_left[idx]
                    y_left_rd = y_left[idx]
                    # Fit the left lane
                    model_left = RANSACRegressor(poly_regression, 
                                                min_samples = int(min(lucky_number, ParallelPolynomialRegression.lane_min_samples*0.65)), 
                                                max_trials = 10000, 
                                                random_state=0)
                    model_left.fit(X_left_rd, y_left_rd)
                    left_lane_coeffs = model_left.estimator_.get_params()["coeffs"]
                    min_x = X_left_rd.min()
                    max_x = X_left_rd.max()
                
                if len(X_right) >=  ParallelPolynomialRegression.lane_min_samples:
                    # Use random sampling for data points in each grid cell
                    idx = np.random.randint(len(X_right), size= int(max((right_scaler * len(X_right)), ParallelPolynomialRegression.lane_min_samples)))
                    X_right_rd = X_right[idx]
                    y_right_rd = y_right[idx]
                    # Fit the right lane
                    model_right = RANSACRegressor(poly_regression,
                                                min_samples = int(min(lucky_number, ParallelPolynomialRegression.lane_min_samples*0.65)),
                                                max_trials = 10000,
                                                random_state=0)
                    model_right.fit(X_right_rd, y_right_rd)
                    right_lane_coeffs = model_right.estimator_.get_params()["coeffs"]
                    
                    if min_x > X_right_rd.min():
                        min_x = X_right_rd.min()
                    if max_x < X_right_rd.max():
                        max_x = X_right_rd.max()
                
                # current_error = model_left.estimator_.score(X_left_rd, y_left_rd) + model_right.estimator_.score(X_right_rd, y_right_rd)
                current_error = poly_regression.cost(left_lane_coeffs, right_lane_coeffs, np.linspace(min_x, max_x, 100))
                # Update best model based on error
                if current_error < prev_error:
                    prev_error = current_error
                    self.left_coeffs = left_lane_coeffs
                    self.right_coeffs = right_lane_coeffs
            except ValueError as e:
                # Handle the case where RANSAC cannot find a valid consensus set
                print(f"Warning: RANSAC did not find a valid consensus set. Iteration {iteration} skipped.")
                continue
                
            iteration += 1
        
            
    def get_params(self, deep=False):
        return {'left_coeffs': self.left_coeffs, 'right_coeffs': self.right_coeffs}

    def set_params(self,random_state=None):
        self.random_state = random_state

    def predict(self, X):
        """
        Predict the lane width based on the polynomial coefficients.

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        X_left = X[:ParallelPolynomialRegression.X_left_length]
        X_right = X[ParallelPolynomialRegression.X_left_length:]
        
        poly_eqn_left = np.poly1d(self.left_coeffs)
        poly_eqn_right = np.poly1d(self.right_coeffs)
        
        y_hat_left = poly_eqn_left(X_left.ravel())
        y_hat_right = poly_eqn_right(X_right.ravel())
        
        y_hat = np.concatenate((y_hat_left, y_hat_right))
        
        return y_hat

    def score(self, X, y):
        """
        Calculate the total cost of the model by combining the mean squared error and the custom cost.
        
        Args:
        - X (array): The input data.
        - y (array): The target values.
        
        Returns:
        - float: The total cost.
        
        """
        X_left = X[:ParallelPolynomialRegression.X_left_length]
        X_right = X[ParallelPolynomialRegression.X_left_length:]
        # Check if X_left or X_right is empty
        if X_left.size == 0 or X_right.size == 0:
            # Handle the case where there's not enough data to score
            # This could involve returning a default high cost or another appropriate value
            mse = float('inf')  # or some other appropriate action
        else:
            mse = mean_squared_error(y, self.predict(X))
            
        x_range = np.linspace(X_left.min(), X_left.max(), 100)
        custom_cost = self.cost(x_range = x_range, parallelism_weight=100)
        
        total_cost = mse + custom_cost
        # total_cost = mse
        
        return total_cost
    
    def cost(self,x_range, parallelism_weight=100):
        """
        Calculate the cost by considering both the mean squared error between the true lane width
        and the measured lane width across a range of x-values, and the difference in slopes
        between the two lanes to ensure parallelism.

        Args:
        - x_range (array): Range of x-values to evaluate the lane width.
        - parallelism_weight (float): Weight for the parallelism penalty term.

        Returns:
        - float: The total cost.
        """
        if self.left_coeffs is None or self.right_coeffs is None:
            return float('inf')  # Return a high cost to indicate an invalid state
        
        poly_regression = PolynomialRegression(degree=self.degree)
        total_cost = poly_regression.cost(self.left_coeffs, self.right_coeffs, x_range, parallelism_weight)
        
        return total_cost
    


