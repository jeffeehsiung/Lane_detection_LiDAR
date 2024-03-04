# data loading
import numpy as np
import matplotlib.pyplot as plt

class LaneMarker:
    def __init__(self) -> None:
        pass

    def create_grid_dict(self, pcd, attributes, slopes_dict, max_lane_width=3.9, num_lanes = 2, visualize=False):
        """
        Creates a grid dictionary for a given range of x-coordinates, considering the slope of each lane segment.

        Parameters:
        - pcd: Open3D point cloud object.
        - attributes: Attributes corresponding to each point in the point cloud.
        - slopes_dict: A dictionary with the slope and intercept for each segment.
        - max_lane_width: The maximum width of a lane.
        - visualize: Whether to visualize the grid bounds.
        
        Returns:
        - dict: A dictionary with grid coordinates as keys and adjusted grid bounds as values based on the slope.
        """
        grid_dict = {}
        segment_dict = {}
        # Convert Open3D point cloud to NumPy array
        points = np.asarray(pcd.points)
        # sort the points and intensities based on the x-coordinate
        sorted_indices = np.argsort(points[:, 0])
        points = points[sorted_indices]
        attributes = attributes[sorted_indices]
        
        # Calculate the range of x-coordinates
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        step = 1
        line_of_sight = 10  # The distance ahead of the vehicle to consider for the grid bounds
        
        # Process slopes_dict to exclude slopes outside certain conditions
        mean_slope = np.mean([slope for _, (slope, _) in slopes_dict.items()])
        std_slope = np.std([slope for _, (slope, _) in slopes_dict.items()])
        mean_intercept = np.mean([intercept for _, (_, intercept) in slopes_dict.items()])
        std_intercept = np.std([intercept for _, (_, intercept) in slopes_dict.items()])
        # remove slopes that are outside of the probability distribution and are greater than 35 degrees: tan(35) = 0.7
        z_score_threshold = 1.645 # 80% of the probability distribution
        slopes_dict_copy = slopes_dict.copy()
        
        slopes_dict = {segment: (slope, intercept) for segment, (slope, intercept) in slopes_dict.items() if mean_slope - z_score_threshold * std_slope < slope < mean_slope + z_score_threshold * std_slope and abs(slope) < 0.7}
        print(f"Removed slopes: {[(segment, (slope, intercept)) for segment, (slope, intercept) in slopes_dict_copy.items() if mean_slope - z_score_threshold * std_slope > slope or slope > mean_slope + z_score_threshold * std_slope or abs(slope) > 0.7]}")
        # # further remove segments in the slopes dictionary where intercept is outside of 80% of the probability distribution
        slopes_dict = {segment: (slope, intercept) for segment, (slope, intercept) in slopes_dict.items() if mean_intercept - z_score_threshold * std_intercept < intercept < mean_intercept + z_score_threshold * std_intercept}
        print(f"Removed intercepts: {[(segment, (slope, intercept)) for segment, (slope, intercept) in slopes_dict_copy.items() if mean_intercept - z_score_threshold * std_intercept > intercept or intercept > mean_intercept + z_score_threshold * std_intercept]}")
    
        # Define segments with their slopes and intercepts
        for label in np.unique(attributes[:, -1]):
            if label in slopes_dict:
                segment_slope, segment_intercept = slopes_dict[label]
                cluster_mask = attributes[:, -1] == label
                cluster_points = points[cluster_mask]
                segment_start, segment_end = cluster_points[:, 0].min(), cluster_points[:, 0].max()
                if len(cluster_points) > 1:
                    segment_dict[(segment_start, segment_end)] = (segment_slope, segment_intercept)
        
        # Start defining grids
        x_current = min_x
        previous_slope, previous_intercept = 0, 0
        while x_current < max_x:
            applicable_segments = [(start_end, s_i) for start_end, s_i in segment_dict.items() if start_end[0] <= x_current <= start_end[1]]        
            if np.negative(line_of_sight) <= x_current <= line_of_sight:
                segment_slope = mean_slope
                segment_intercept = 0
            elif not applicable_segments:  # No segment directly matches x_current
                # Use average of previous and next segment's slope and intercept
                try:
                    next_segment = min([(start_end, s_i) for start_end, s_i in segment_dict.items() if start_end[0] > x_current], key=lambda x: x[0][0])
                    segment_slope = (previous_slope + next_segment[1][0]) / 2
                    segment_intercept = (previous_intercept + next_segment[1][1]) / 2
                except ValueError:  # No next segment. Use the average of previous and mean slope and intercept
                    segment_slope = (previous_slope + mean_slope) / 2
                    segment_intercept = (previous_intercept + mean_intercept) / 2
            
            elif len(applicable_segments) > 1:  # Multiple segments match
                # Choose segment with slope closest to previous segment's slope
                segment_slope, segment_intercept = min(applicable_segments, key=lambda x: abs(x[1][0] - previous_slope))[1]
            else:  # Exactly one segment matches
                segment_slope, segment_intercept = applicable_segments[0][1]
            
            # Update previous slope and intercept for next iteration
            previous_slope, previous_intercept = segment_slope, segment_intercept
            
            # Define grid bounds with slope consideration
            lanes_vertical_bin_mean_y = segment_slope * (x_current) + (segment_intercept/num_lanes)
            # mediate with the center of the lane
            lanes_vertical_bin_mean_y = (lanes_vertical_bin_mean_y) / 2
            
            grid_x_low_bound = x_current - line_of_sight
            grid_x_up_bound = x_current + line_of_sight
            grid_y_low_bound = min(np.negative(max_lane_width), lanes_vertical_bin_mean_y - max_lane_width * 1.15)
            grid_y_up_bound = max(max_lane_width, lanes_vertical_bin_mean_y + max_lane_width * 1.15)
                
            grid_dict[(lanes_vertical_bin_mean_y, x_current)] = [grid_x_up_bound, grid_x_low_bound, grid_y_up_bound, grid_y_low_bound]
            
            x_current += step
        
        # plot the points and on top of that plot the grid bounds and mark the lanes_vertical_bin_mean_y and x for each grid
        if visualize:
            fig, ax = plt.subplots()
            ax.scatter(points[:, 0], points[:, 1], c='b', label='Point Cloud')
            for grid_coord, grid_bounds in grid_dict.items():
                x_up, x_low, y_up, y_low = grid_bounds
                ax.plot([x_low, x_up], [y_up, y_up], c='r')
                ax.plot([x_up, x_up], [y_up, y_low], c='r')
                ax.plot([x_up, x_low], [y_low, y_low], c='r')
                ax.plot([x_low, x_low], [y_low, y_up], c='r')
                lanes_vertical_bin_mean_y, x = grid_coord
                ax.scatter(x, lanes_vertical_bin_mean_y, c='g')
                ax.text(x, lanes_vertical_bin_mean_y, f'{x:.2f}, {lanes_vertical_bin_mean_y:.2f}', fontsize=8)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Grid Bounds')
            plt.show()
        
        return grid_dict


    def filter_lidar_data_by_grid(self, lidar_data, grid_dict):
        """
        Filters LiDAR data points based on the grid definitions, assigning points to the corresponding grid cells.

        Args:
        - lidar_data (numpy.ndarray): The LiDAR data points array.
        - grid_dict (dict): The dictionary with grid coordinates as keys and grid bounds as values.

        Returns:
        - dict: A dictionary mapping each grid cell to the LiDAR data points within that cell.
        """
        data_in_grid = {}
        for grid_cell_coord, bounds in grid_dict.items():
            x_upper_bound, x_lower_bound, y_upper_bound, y_lower_bound = bounds
            # Filter LiDAR data points within the current grid cell bounds
            filtered_data = lidar_data[(lidar_data[:, 0] <= x_upper_bound) &
                                    (lidar_data[:, 0] > x_lower_bound) &
                                    (lidar_data[:, 1] <= y_upper_bound) &
                                    (lidar_data[:, 1] > y_lower_bound)]
            data_in_grid[grid_cell_coord] = filtered_data
        return data_in_grid
    


