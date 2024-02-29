# data loading
import numpy as np
import matplotlib.pyplot as plt

class LaneMarker:
    def __init__(self) -> None:
        pass

    
    def fit_lane_polynomials(self, lane_groups, reference_y=0):
        """
        Fits cubic polynomials to each lane and selects the lanes directly to the left and right of the reference point (y = 0).

        Parameters:
        - lane_groups: A dictionary containing lane points, where keys are (bin_index, lane_index) and values are numpy arrays of points.
        - reference_y: The Y-coordinate of the reference point to determine the lanes to the left and right.

        Returns:
        - lane_fits: A dictionary with keys 'left' and 'right', containing the polynomial coefficients of the lanes directly to the left and right of the reference point.
        """
        closest_left_lane = None
        closest_right_lane = None
        
        min_left_distance = float('inf')
        min_right_distance = float('inf')
        
        lanes = {}
        for (bin_index, lane_index), points in lane_groups.items():
            if lane_index not in lanes:
                lanes[lane_index] = points
            else:
                # Concatenate points from the same lane across different bins
                lanes[lane_index] = np.vstack((lanes[lane_index], points))

            
        # find the left and right lanes based on the reference point
        for lane_index, points in lanes.items():
            avg_y = np.mean(points[:, 1]) # Average Y-coordinate of the lane
            if avg_y < 0:  # Lane is to the left of y = 0
                distance = abs(avg_y)
                if distance < min_left_distance:
                    min_left_distance = distance
                    closest_left_lane = lane_index
            elif avg_y > 0:  # Lane is to the right of y = 0
                distance = abs(avg_y)
                if distance < min_right_distance:
                    min_right_distance = distance
                    closest_right_lane = lane_index
        
        lane_fits = {'left': None, 'right': None}

        # Fit the closest lane on each side, if any
        if closest_left_lane:
            coefs, residuals, _, _, _ = np.polyfit(lanes[closest_left_lane][:, 0], lanes[closest_left_lane][:, 1], 3, full=True)
            lane_fits['left'] = {'coefs': coefs, 'residuals': residuals}

        if closest_right_lane:
            coefs, residuals, _, _, _ = np.polyfit(lanes[closest_right_lane][:, 0], lanes[closest_right_lane][:, 1], 3, full=True)
            lane_fits['right'] = {'coefs': coefs, 'residuals': residuals}
        return coefs, residuals
    
    
    def create_grid_dict(self, pcd, attributes, num_lanes, max_lane_width=3.9):
        """
        Creates a grid dictionary for a given range of x-coordinates, considering the slope of each lane segment.

        Parameters:
        - pcd: Open3D point cloud object.
        - attributes: Attributes corresponding to each point in the point cloud.

        Returns:
        - dict: A dictionary with grid coordinates as keys and adjusted grid bounds as values based on the slope.
        """
        grid_dict = {}

        # Convert Open3D point cloud to NumPy array
        points = np.asarray(pcd.points)
        # sort the points and intensities based on the x-coordinate
        sorted_indices = np.argsort(points[:, 0])
        points = points[sorted_indices]
        attributes = attributes[sorted_indices]
        
        # Calculate the range of x-coordinates
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        line_of_sight = 1 # The distance ahead of the vehicle to consider for the grid bounds
        
        # Initialize the dictionary to store slopes for each segment
        slopes_dict = {}
        
        # Based on the attributes slope labels, find for each label cluster the x range
        for label in np.unique(attributes[:, -1]):
            # Filter points that belong to the current cluster
            cluster_mask = attributes[:, -1] == label
            cluster_points = points[cluster_mask]
            # Calculate the range of x-coordinates for the current cluster
            segment_start, segment_end = cluster_points[:, 0].min(), cluster_points[:, 0].max()
            if len(cluster_points) > 1:
                    # Calculate the slope for the current segment using linear regression (np.polyfit)
                    x_values = cluster_points[:, 0]
                    y_values = cluster_points[:, 1]
                    slope, intercept = np.polyfit(x_values, y_values, 1)[:2]
                    # Store the slope in the dictionary with the segment range as the key, and with the slope
                    slopes_dict[(segment_start, segment_end)] = (slope, intercept)
        
        x_current = min_x
        while x_current < max_x:
            # Determine the segment slope; find the slope and intercept entry that x_current falls into the range of the segment
            for (segment_start, segment_end), (current_slope, intercept) in slopes_dict.items():
                if segment_start <= x_current <= segment_end:
                    break
            else:
                # If x_current is not within any segment, move to the next x-coordinate
                x_current += 0.25
                continue
            
            # Calculate the y-offset based on the current slope
            # y_offset = current_slope * (x_current - min_x) + (intercept/num_lanes)
            y_offset = current_slope * (x_current - min_x)

            # Define grid bounds taking into account the slope offset for y
            grid_x_low_bound = (x_current - line_of_sight)
            grid_x_up_bound = x_current + line_of_sight
            grid_y_low_bound = -max_lane_width*1.25  + y_offset
            grid_y_up_bound = max_lane_width*1.25 + y_offset

            # Store the grid bounds
            grid_dict[(y_offset, x_current)] = [grid_x_up_bound, grid_x_low_bound, grid_y_up_bound, grid_y_low_bound]

            # Move to the next segment
            x_current += 0.25
            
        # plot the points and on top of that plot the grid bounds
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], c='b', label='Point Cloud')
        for grid_bounds in grid_dict.values():
            x_up, x_low, y_up, y_low = grid_bounds
            ax.plot([x_low, x_up], [y_up, y_up], c='r')
            ax.plot([x_up, x_up], [y_up, y_low], c='r')
            ax.plot([x_up, x_low], [y_low, y_low], c='r')
            ax.plot([x_low, x_low], [y_low, y_up], c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Grid Bounds')
        plt.show()
        
        # close the plot
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
    


