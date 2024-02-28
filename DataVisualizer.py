# data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data preprocessing
import open3d as o3d


class DataVisualizer:
    def __init__(self) -> None:
        pass
    
    def visualize_lane_detection(self, original_pcd, filtered_pcd):
        """
        Visualizes the original point cloud and the filtered point cloud representing detected lanes.
        
        Parameters:
        - original_pcd: The original Open3D point cloud object.
        - filtered_pcd: The filtered Open3D point cloud object representing detected lanes.
        """
        # Set the color of the original point cloud to gray
        colors = np.asarray(original_pcd.colors)
        gray_color = np.array([[0.5, 0.5, 0.5]])  # Gray color
        if len(colors) == 0:  # If original point cloud has no colors
            colors = np.tile(gray_color, (np.asarray(original_pcd.points).shape[0], 1))
        else:
            colors *= 0.5  # Darken the original colors to distinguish from filtered points
        original_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Set the color of the filtered point cloud to yellow for high visibility
        lane_color = np.array([[1, 0.75, 0.8]])  # Pink color
        num_filtered_points = np.asarray(filtered_pcd.points).shape[0]
        filtered_colors = np.tile(lane_color, (num_filtered_points, 1))
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # Combine the original and filtered point clouds for visualization
        combined_pcd = original_pcd + filtered_pcd
        
        # Visualize the combined point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Before- Visualization")
        vis.add_geometry(combined_pcd)
        vis.run()
        vis.destroy_window()
    
        # Visualize only the filtered point cloud for better visibility
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="After- Visualization")
        vis.add_geometry(combined_pcd)
        vis.add_geometry(filtered_pcd)
        vis.run()
        vis.remove_geometry(combined_pcd, False)
        vis.poll_events()
        vis.update_renderer()
        vis.close()
        vis.destroy_window()
    
    def visualize_lane_groups_by_lane_number(self, pointcloud, lane_groups):
        """
        Visualizes the lane groups by coloring the points in each lane with a consistent color across all X bins.
        Points not belonging to any lane are colored white.
        
        Parameters:
        - pointcloud: Open3D pointcloud object.
        - lane_groups: A dictionary where keys are tuples (bin_index, lane_index) and values are numpy arrays of points assigned to a lane.
        """
        # Initialize a color array for all points in the point cloud to white
        colors = np.ones((np.asarray(pointcloud.points).shape[0], 3))  # Set all points to white initially

        # Generate a color map with a unique color for each lane
        num_lanes = max(lane_index for _, lane_index in lane_groups.keys()) + 1
        unique_colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_lanes))

        # Flatten the point cloud for easy indexing
        flattened_points = np.asarray(pointcloud.points).reshape((-1, 3))

        # Assign colors based on lane number
        for (bin_index, lane_index), group_points in lane_groups.items():
            color = unique_colors[lane_index][:3]  # Color based on lane_index only
            for point in group_points:
                idx = np.where((flattened_points == point).all(axis=1))[0]
                if len(idx) > 0:
                    colors[idx[0]] = color  # Assign the specific color to matching points

        # Update point cloud colors
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud with colored lane groups
        o3d.visualization.draw_geometries([pointcloud], point_show_normal=False, window_name="Lane Groups by Lane Number Visualization - Modified")

