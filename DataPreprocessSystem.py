# data loading
import numpy as np
import os
import open3d as o3d
from sklearn.preprocessing import StandardScaler

class DataPreprocessSystem:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.pointclouds_with_attributes = []

    def load_pointclouds_with_attributes(self, folder_path):
        """
        Loads a pointcloud from a binary file and parses it into a numpy array for further processing.
        Given that open3D offers efficient pointcloud processing, we will use it to load the pointclouds.
        Open3D to load and process pointcloud data, the primary focus is on spatial information (i.e., the x, y, z coordinates of each point). 
        By default, Open3D's PointCloud object does not directly handle non-spatial attributes.
        Hence, relavant non-spatial attributes critical to ego lane detection can be managed alongside Open3D pointcloud objects.
        Here, we returns a list of tuples containing Open3D pointcloud objects and corresponding attributes.
        
        PointCloud point and Attributes
        ----------
        1. scene: scene id - to be added to the point cloud
        2. x: x coordinate of the point
        3. y: y coordinate of the point
        4. z: z coordinate of the point
        5. intensity: intensity of the point
        6. lidar_beam: lidar beam id
        ----------
        """
        # get a list of all point cloud files in the directory
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.bin')]
        # load each point cloud from the folder and assign a scene id to it using zip or enumerate
        for scene_id, file_path in enumerate(file_paths):
            # Load the binary pointcloud data as a NumPy array
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
            
            # Create an Open3D PointCloud object for spatial data (X, Y, Z)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            # add the scene id to the point cloud
            attributes = np.c_[np.ones([points.shape[0], 1], dtype=np.int32) * scene_id, points[:, 3:]]
            
            # Add the tuple (Open3D pointcloud, attributes array) to the list
            self.pointclouds_with_attributes.append((pcd, attributes))
        
        
        return self.pointclouds_with_attributes


    def preprocess_pointcloud(self, pcd, voxel_size=0.05):
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def compute_normals(self, downpcd, radius=0.1, max_nn=30):
        """
        Computes the normals of a downsampled point cloud.
        
        Parameters:
        - downpcd: The downsampled Open3D point cloud object.
        - radius: Search radius for normal computation.
        - max_nn: Maximum number of nearest neighbors to use.
        
        Returns:
        - The Open3D point cloud object with normals computed.
        """
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        return downpcd

    def z_filter(self, pointcloud):
        """
        Filters a point cloud based on z-axis values to remove points that are significantly higher than the ground.
        
        Parameters:
        - pointcloud: The Open3D point cloud object.
        
        Returns:
        - Indices of points within the desired z-axis range.
        """
        z_values = np.asarray(pointcloud.points)[:, 2]
        z_threshold = np.mean(z_values) + 2 * np.std(z_values)
        selected_indices = np.where(z_values < z_threshold)[0]
        return selected_indices

    def scaler_transform(self, pointcloud):
        '''
        Transform the data by calculating the z-score of each value in the sample,
        This is done by taking the feature x,y,z, subtract the mean of the feature and then divide by the standard deviation of the feature.
        This process can be inflence by outliers, hence noise should be removed before scaling.
        - pointcloud: Open3D pointcloud objects
        
        Returns:
        - pointcloud_T: A new Open3D pointcloud objects with transformed coordinates that has the same shape as the original pointcloud.
        '''
        scaler = StandardScaler()
        # Extract the coordinates from the point cloud 
        coordinates = np.asarray(pointcloud.points)
        # fit the scaler to the data, transform the data
        xyz_T = scaler.fit_transform(coordinates)
        # create a point cloud with the transformed coordinates
        pointcloud_T = o3d.geometry.PointCloud()
        pointcloud_T.points = o3d.utility.Vector3dVector(xyz_T)
        return pointcloud_T

def scaler_inverse_transform(self, pointcloud_T, scaler):
    '''
    Transform the data by calculating the z-score of each value in the sample,
    This is done by taking the feature x,y,z, subtract the mean of the feature and then divide by the standard deviation of the feature.
    This process can be inflence by outliers, hence noise should be removed before scaling.
    - pointcloud: Open3D pointcloud objects
    - scaler: The scaler object used to scale the data
    
    Returns:
    - pointcloud: A new Open3D pointcloud objects with transformed coordinates that has the same shape as the original pointcloud.
    '''
    # Extract the coordinates from the point cloud 
    coordinates_T = np.asarray(pointcloud_T.points)
    # fit the scaler to the data, transform the data
    xyz = scaler.inverse_transform(coordinates_T)
    # create a point cloud with the transformed coordinates
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(xyz)
    return pointcloud