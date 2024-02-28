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
                
                # Polynomial Fitting with RANSAC
                model_left = RANSACRegressor(PolynomialRegression(degree = poly_degree),
                                             min_samples = int(min_samples*0.5),
                                             max_trials = 10000,
                                             random_state=0)
                model_left.fit(X_left, y_left)
                left_lane_coeffs = model_left.estimator_.get_params(deep = True)["coeffs"]
                
                model_right = RANSACRegressor(PolynomialRegression(degree = poly_degree),
                                              min_samples = int(min_samples*0.5),
                                              max_trials = 10000,
                                              random_state=0)
                model_right.fit(X_right, y_right)
                right_lane_coeffs = model_right.estimator_.get_params(deep = True)["coeffs"]
                
                # Model Evaluation (this is a placeholder for whatever metric you use)
                current_error = PolynomialRegression.cost(left_lane_coeffs, right_lane_coeffs, np.linspace(min_x, max_x, num=100))
                
                # Update best model based on error
                if current_error < prev_error:
                    prev_error = current_error
                    best_coeffs_pair_left = left_lane_coeffs
                    best_coeffs_pair_right = right_lane_coeffs
                
            iteration += 1
            
        best_coeffs_pair = np.append(best_coeffs_pair_left, best_coeffs_pair_right, axis = 0)


        # start = time.time()
        # while iteration <= max_iter:
        #     for y in (-1, 1):
        #         for x in range(min_x, max_x + 1):
        #             if y == -1:
        #                 if len(data_in_grid[y, x]) >= min_samples:
        #                     idx = np.random.randint(len(data_in_grid[y, x]), size = poly_degree)
        #                     data_repres_left = np.append(data_repres_left, data_in_grid[y, x][idx, :], axis = 0)
        #                 elif len(data_in_grid[y, x]) == 0:
        #                     pass
        #                 else:
        #                     idx = np.random.randint(len(data_in_grid[y, x]), size = len(data_in_grid[y,x]))
        #                     data_repres_left = np.append(data_repres_left, data_in_grid[y, x][idx, :], axis = 0)

        #             elif y == 1:
        #                 if len(data_in_grid[y, x]) >= min_samples:
        #                     idx = np.random.randint(len(data_in_grid[y, x]), size = poly_degree)
        #                     data_repres_right = np.append(data_repres_right, data_in_grid[y, x][idx, :], axis = 0)
        #                 elif len(data_in_grid[y, x]) == 0:
        #                     pass
        #                 else:
        #                     idx = np.random.randint(len(data_in_grid[y, x]), size = len(data_in_grid[y,x]))
        #                     data_repres_right = np.append(data_repres_right, data_in_grid[y, x][idx, :], axis = 0)
                            
        #     left_grids_x = np.sort(data_repres_left, axis = 0)[:, 0] 
        #     left_grids_y = np.sort(data_repres_left, axis = 0)[:, 1]
        #     right_grids_x = np.sort(data_repres_right, axis = 0)[:, 0] 
        #     right_grids_y = np.sort(data_repres_right, axis = 0)[:, 1]

        #     ransac_up = RANSACRegressor(PolynomialRegression(degree = poly_degree),
        #                             min_samples = int(min_samples*0.5),    
        #                             max_trials = 10000,
        #                             random_state=0)
        #     ransac_up.fit(np.expand_dims(left_grids_x, axis=1), left_grids_y)
        #     left_grids_y_pred = ransac_up.predict(np.expand_dims(left_grids_x, axis=1))
        #     left_lane_coeffs= ransac_up.estimator_.get_params(deep = True)["coeffs"]

        #     ransac_down = RANSACRegressor(PolynomialRegression(degree = poly_degree), 
        #                             min_samples = int(min_samples*0.5),    
        #                             max_trials = 10000,
        #                             random_state=0)
        #     ransac_down.fit(np.expand_dims(right_grids_x, axis=1), right_grids_y)
        #     right_grids_y_pred = ransac_down.predict(np.expand_dims(right_grids_x, axis=1))
        #     right_lane_coeffs = ransac_down.estimator_.get_params(deep = True)["coeffs"]
            
        #     ego_lane_coeffs_pair = np.append(right_lane_coeffs, left_lane_coeffs, axis = 0) 
        #     curr_error = PolynomialRegression.cost(left_lane_coeffs, right_lane_coeffs, np.linspace(min_x, max_x, num=100))      
            
        #     if curr_error < prev_error:
        #         prev_error = curr_error 
        #         best_coeffs_pair = ego_lane_coeffs_pair

        #     iteration += 1

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
    