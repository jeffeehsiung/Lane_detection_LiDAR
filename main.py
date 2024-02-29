# data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data preprocessing
import open3d as o3d
import os
import time

from sklearn.linear_model import RANSACRegressor
from DataPreprocessSystem import *
from LaneDetectionSystem import *
from LaneMarker import *
from DataVisualizer import *
from PolynomialRegression import *

import warnings
from numpy import RankWarning
warnings.simplefilter('ignore', RankWarning)
 

if __name__ == "__main__":
    # Example of using the system
    folder_path = './pointclouds'

    data_preprocess_system = DataPreprocessSystem(folder_path)
    lane_detection_system = LaneDetectionSystem()
    lane_marker = LaneMarker()
    data_visualizer = DataVisualizer()

    # Example of preprocessing a point cloud
    pointclouds_with_attributes = data_preprocess_system.load_pointclouds_with_attributes(folder_path)  # Assuming you're working with the first point cloud

    # create a list of filename which with extensoin .bin
    file_name = [os.path.basename(file) for file in os.listdir(folder_path) if file.endswith('.bin')]

    for i, (pcd, attributes) in enumerate(pointclouds_with_attributes):
        # filter the point cloud based on the z-axis
        selected_indices = data_preprocess_system.z_filter(pcd)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[selected_indices])
        attributes = attributes[selected_indices]
        
        #  explore the parameter space and visualize the results
        eps_values = np.arange(0.01, 0.9, 0.03)  # Example range for eps
        min_samples_values = range(5, 30, 5)  # Example range for min_samples

        best_score = float('-inf')
        best_params = {'eps': None, 'min_samples': None}

        for eps in eps_values:
            for min_samples in min_samples_values:
                num_clusters, noise_ratio = lane_detection_system.evaluate_clustering(pcd, eps, min_samples)
                
                # Calculate the clustering score
                current_score = lane_detection_system.score_clustering(num_clusters, noise_ratio)
                
                # Update best parameters if current score is better
                if current_score > best_score:
                    best_score = current_score
                    best_params['eps'] = eps
                    best_params['min_samples'] = min_samples

        print(f"Best parameters: {best_params}, Best score: {best_score}")

        eps = best_params['eps']  # Use tuned eps value
        min_samples = best_params['min_samples']  # Use tuned min_samples value
        
        # ground_pcd, non_ground_pcd, ground_attributes, non_ground_attributes= lane_detection_system.segment_ground_plane(pcd, attributes, distance_threshold=(eps*0.005), ransac_n=5, num_iterations=1000)
        
        # visualize the segmentation result
        # data_visualizer.visualize_lane_detection(pcd, non_ground_pcd)
        
        # pcd = non_ground_pcd
        # attributes = non_ground_attributes
        
        # Perform DBSCAN and update attributes
        best_labels = lane_detection_system.cluster_with_dbscan(pcd, eps, min_samples)
        updated_attributes = np.hstack((attributes, best_labels.reshape(-1, 1)))  # Append labels as a new column
        # Learn intensity threshold from clusters    
        intensity_threshold = lane_detection_system.learn_intensity_threshold(updated_attributes, percentage=0.7)
        
        # Filter clusters based on intensity and geometric shape
        selected_indices = lane_detection_system.intensity_filter(updated_attributes, intensity_threshold)
        
        # Extract filtered points for visualization or further processing
        filtered_points = np.asarray(pcd.points)[selected_indices]
        
        # calculate the delta of the points before and after filtering
        delta = len(np.asarray(pcd.points)) - len(filtered_points)
        
        print(f"Filtered {delta} points from {len(np.asarray(pcd.points))} to {len(filtered_points)}")
        
        # Create a new point cloud object for filtered points, if needed
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_attributes = updated_attributes[selected_indices]
        
        data_visualizer.visualize_lane_detection(pcd, filtered_pcd)
        # Replace the original pcd and attributes with the filtered ones
        pointclouds_with_attributes[i] = (filtered_pcd, filtered_attributes)
        num_lanes, y_peaks_coordinates = lane_detection_system.find_number_of_lanes(filtered_pcd, filtered_attributes, percentile = 1, min_num_peaks=2)
        
        # normalize the point cloud
        # filtered_pcd = data_preprocess_system.scaler_transform(filtered_pcd)
        # cluster the point cloud into lanes
        num_slopes = lane_detection_system.optimize_k_means(filtered_pcd, max_n_clusters = num_lanes + 1, visualize=True)
        print(f"Number of lanes: {num_slopes}") 
        # cluster the point cloud into lanes using k-means
        kmeans = KMeans(n_clusters=num_slopes, random_state=10, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(np.asarray(filtered_pcd.points)[:, :2])
        
        # update the attributes signigying slope labels
        filtered_attributes = np.hstack((filtered_attributes, cluster_labels.reshape(-1, 1)))  # Append labels as a new column
        
        # calculate the slope of each slope cluster
        slopes = lane_detection_system.calculate_slope(filtered_pcd, filtered_attributes, num_slopes)
        print(f"Slopes: {slopes}")
        
        # delete the point cloud and attributes with the slope orthogonal to the x-axis 
        filtered_pcd, filtered_attributes = lane_detection_system.delete_orthogonal_slope(filtered_pcd, filtered_attributes, slopes)
        
        
        
        # grid_dict = lane_marker.create_grid_dict(min_x, max_x)
        grid_dict = lane_marker.create_grid_dict(filtered_pcd, filtered_attributes)
        # conver pointcloud to np.array
        filtered_pcd_array = np.asarray(filtered_pcd.points)
        min_x = np.floor(np.min(filtered_pcd_array[:, 0])).astype(int)
        max_x = np.ceil(np.max(filtered_pcd_array[:, 0])).astype(int) 
        data_in_grid = lane_marker.filter_lidar_data_by_grid(filtered_pcd_array, grid_dict)
        
        poly_degree = 3
        # shape of lidar_data
        n = filtered_pcd_array.shape[1]
        
        data_repres_left = np.empty((0, n))
        data_repres_right = np.empty((0, n)) 

        iteration = 0
        max_iter = 1000
        # prev_error = 1000
        prev_error = float('inf')
        best_coeffs_pair_left = None
        best_coeffs_pair_right = None
        
        start = time.time()
        while iteration <= max_iter:
            # Adjust the loop to iterate through grid_dict keys directly
            for grid_cell_coord, data_points in data_in_grid.items():
                y_offset, x = grid_cell_coord
                if len(data_points) >= min_samples:
                    for point in data_points:
                        if point[1] > y_offset:  # If the point's y coordinate is greater than y_center, it's on the left
                            data_repres_left = np.append(data_repres_left, [point], axis=0)
                        else:  # Otherwise, it's on the right
                            data_repres_right = np.append(data_repres_right, [point], axis=0)
            
            # The following processing is based on sorted x values for consistency
            # Preprocess Data: Sort based on x-coordinate
            data_repres_left = data_repres_left[data_repres_left[:, 0].argsort()]
            data_repres_right = data_repres_right[data_repres_right[:, 0].argsort()]
            
            # Ensure enough points for fitting
            if len(data_repres_left) >= poly_degree + 1 and len(data_repres_right) >= poly_degree + 1:
                X_left = data_repres_left[:, 0].reshape(-1, 1)
                y_left = data_repres_left[:, 1]
                X_right = data_repres_right[:, 0].reshape(-1, 1)
                y_right = data_repres_right[:, 1]
                # Create scalers for X and y (if y scaling is desired)
                scaler_X_left = StandardScaler().fit(X_left)
                scaler_X_right = StandardScaler().fit(X_right)
                # mu_X for X before scaling
                mu_X_left = scaler_X_left.mean_
                mu_X_right = scaler_X_right.mean_
                # sigma_X for standard deviation of X before scaling
                sigma_X_left = scaler_X_left.scale_
                sigma_X_right = scaler_X_right.scale_
                # Scale the data
                X_left_scaled = scaler_X_left.transform(X_left)
                X_right_scaled = scaler_X_right.transform(X_right)
                                
                # Polynomial Fitting with RANSAC
                model_left = RANSACRegressor(PolynomialRegression(degree = poly_degree),
                                             min_samples = 7,
                                             max_trials = 10000,
                                             random_state=0)
                model_left.fit(X_left_scaled, y_left)
                left_lane_coeffs = model_left.estimator_.get_params(deep = True)["coeffs"]
                
                model_right = RANSACRegressor(PolynomialRegression(degree = poly_degree),
                                              min_samples = 7,
                                              max_trials = 10000,
                                              random_state=0)
                model_right.fit(X_right_scaled, y_right)
                right_lane_coeffs = model_right.estimator_.get_params(deep = True)["coeffs"] # corresponds to coefficients for order from highest to lowest
                
                # Model Evaluation (this is a placeholder for whatever metric you use)
                current_error = PolynomialRegression.cost(left_lane_coeffs, right_lane_coeffs, np.linspace(min_x, max_x, num=100))
                
                # Update best model based on error
                if current_error < prev_error:
                    prev_error = current_error
                    best_coeffs_pair_left = left_lane_coeffs
                    best_coeffs_pair_right = right_lane_coeffs
                
            iteration += 1
        
        # convert the coefficients back to the original scale
        alpha_left_3 = best_coeffs_pair_left[3] / sigma_X_left**3
        alpha_left_2 = best_coeffs_pair_left[2] / sigma_X_left**2
        alpha_left_1 = best_coeffs_pair_left[1] / sigma_X_left
        alpha_left_0 = best_coeffs_pair_left[0] - (alpha_left_1 * mu_X_left) + (alpha_left_2 * mu_X_left**2) - (alpha_left_3 * mu_X_left**3)
        best_coeffs_pair_left = np.array([alpha_left_3, alpha_left_2, alpha_left_1, alpha_left_0])
        alpha_right_3 = best_coeffs_pair_right[3] / sigma_X_right**3
        alpha_right_2 = best_coeffs_pair_right[2] / sigma_X_right**2
        alpha_right_1 = best_coeffs_pair_right[1] / sigma_X_right
        alpha_right_0 = best_coeffs_pair_right[0] - (alpha_right_1 * mu_X_right) + (alpha_right_2 * mu_X_right**2) - (alpha_right_3 * mu_X_right**3)
        best_coeffs_pair_right = np.array([alpha_right_3, alpha_right_2, alpha_right_1, alpha_right_0])
        best_coeffs_pair = np.append(best_coeffs_pair_left, best_coeffs_pair_right, axis = 0)

        # Convert the best_coeffs_pair to a 2D array with 4 columns
        best_coeffs_pair = best_coeffs_pair.reshape(-1, 4)
        # Ensure the output directory exists
        output_dir = 'sample_output_test'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the coefficients to a text file
        lidar_txt_name = file_name[i].replace('.bin', '.txt')
        # save both left and right lane coefficients as two rows in the text file
        np.savetxt(os.path.join(output_dir, lidar_txt_name), best_coeffs_pair, delimiter=';', fmt='%.15e')        
        
        # # fh = open(f'scene{i}', 'bw')
        # # # save the point cloud to file as float32
        # # np.asarray(filtered_pcd.points).astype('float32').tofile(fh)
    