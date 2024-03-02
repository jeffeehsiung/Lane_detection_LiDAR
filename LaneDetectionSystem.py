# data loading
import pandas as pd
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks


class LaneDetectionSystem:
    def __init__(self):
        pass
            
    def cluster_with_dbscan(self, pcd, eps=0.08, min_samples=10):
        """
        Applies DBSCAN clustering to a point cloud to identify clusters based on spatial proximity.
        
        Parameters:
        - pcd: The Open3D point cloud object to cluster.
        - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        
        Returns:
        - labels: An array of cluster labels for each point in the point cloud.
        """
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
        return labels

    def evaluate_clustering(self, example_pcd, eps, min_samples):
        labels = np.array(example_pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
        max_label = labels.max()
        num_clusters = max_label + 1
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        return num_clusters, noise_ratio

    #  clustering scoring function
    def score_clustering(self, num_clusters, noise_ratio, optimal_clusters=32, noise_penalty=100):
        """
        Scores the clustering outcome based on the number of clusters, noise ratio, and predefined targets.
        
        Parameters:
        - num_clusters: The number of clusters detected.
        - noise_ratio: The ratio of points classified as noise.
        - optimal_clusters: The target number of clusters for optimal scoring. (ground, lanes, crosswalks, trees, curb, vehicles, signs, barriers, trees, housing, beings.)
        - noise_penalty: The weight of the noise ratio in the score calculation.
        
        Returns:
        - score: A calculated score of the clustering outcome.
        """
        # Reward configurations that are close to the optimal number of clusters
        cluster_score = -abs(num_clusters - optimal_clusters)
        
        # Penalize configurations with a high noise ratio
        noise_score = -noise_ratio * noise_penalty
        
        # Calculate total score
        score = cluster_score + noise_score
        return score

    def segment_ground_plane(self, pcd, attributes, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """
        Segments the ground plane from an input point cloud using the RANSAC algorithm.

        Parameters:
        - pcd: Open3D point cloud object from which the ground plane is to be segmented.
        - distance_threshold: Maximum distance a point can be from the plane model to be considered as an inlier.
        - ransac_n: Number of points to sample for generating the plane model in each iteration.
        - num_iterations: Number of iterations RANSAC will run to maximize inliers.

        Returns:
        - ground_pcd: The segmented ground plane as an Open3D point cloud.
        - non_ground_pcd: All points that are not part of the ground plane as an Open3D point cloud.
        """
        # Perform plane segmentation to separate ground
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
        [a, b, c, d] = plane_model

        # Extract inliers and outliers
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # create a list with zeros with length equal to the number of points in the point cloud
        inniers_bool = np.zeros(len(pcd.points), dtype=bool)
        inniers_bool[inliers] = True
        
        # Extract the inline points from the attributes and outliers from the attributes
        inlier_attributes = attributes[inliers]
        outlier_attributes = attributes[~inniers_bool]
        
        # Optional: Assign a unique color to ground and non-ground points for visualization
        inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green for ground
        outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red for non-ground

        return inlier_cloud, outlier_cloud, inlier_attributes, outlier_attributes

    def learn_intensity_threshold(self, attributes, top_percentage=0.10):
        """
        Learns an intensity threshold based on the analysis of clusters.
        
        Parameters:
        - attributes: A NumPy array containing DBSCAN labels and intensities.
        
        Returns:
        - An intensity threshold learned from the cluster analysis.
        """
        # Assuming DBSCAN labels are the last column, and intensity is the second column
        cluster_labels = attributes[:, -1].astype(int)
        intensities = attributes[:, 1]
        
        # Initialize variables to store sum of intensities and count for each cluster
        cluster_intensity_sum = {}
        cluster_point_count = {}
        
        # Calculate sum of intensities and point count for each cluster
        for label, intensity in zip(cluster_labels, intensities):
            if label not in cluster_intensity_sum:
                cluster_intensity_sum[label] = 0
                cluster_point_count[label] = 0
            cluster_intensity_sum[label] += intensity
            cluster_point_count[label] += 1
        
        # Calculate average intensity and std for each cluster
        mean_intensity = {label: cluster_intensity_sum[label] / cluster_point_count[label]
                            for label in cluster_intensity_sum}
        
        # # Determine threshold as the average of top N highest average intensities
        N = max(1, int(round(len(mean_intensity) * top_percentage)))  # Use top 75% of clusters and cast to int
        top_average_intensities = sorted(mean_intensity.values(), reverse=True)[:N]
        threshold = sum(top_average_intensities) / N
        
        return threshold



    def intensity_filter(self, attributes, intensity_threshold):
        """
        Filters clusters based on intensity and normals analysis.
        Parameters:
        - pcd: An Open3D pointcloud object with normals computed.
        - attributes: A NumPy array with DBSCAN labels and intensities.
        - intensity_threshold: Threshold for filtering based on intensity.
        Returns:
        - A list of indices for points considered as part of lanes.
        """
        
        cluster_labels = attributes[:, -1].astype(int)
        intensities = attributes[:, 1]
        
        selected_indices = []
        for i, (label, intensity) in enumerate(zip(cluster_labels, intensities)):
            if label >= 0 and intensity > intensity_threshold:
                selected_indices.append(i)
        
        return selected_indices
    
    
    def optimize_k_means(self, pointcloud_T, min_n_cluster = 3, max_n_clusters = 12, visualize=False):
        """
        Calculate silhouette score for a cluster of points to find the optimal number of clusters.
        - pointcloud_T: scaler transformed Open3D pointcloud objects
        - range_n_clusters: maximum integers representing the max range of clusters to consider.
        - visualize: A boolean indicating whether to visualize the the cluster result or not.
        
        Returns:
        - num_cluster: The optimal number of clusters
        """
        # extract the x and y coordinates
        xy_T = np.asarray(pointcloud_T.points)[:, :2]
        
        silhouette_score_buffer = []
        # distortions = []
        
        if visualize:
            df = pd.DataFrame(xy_T, columns=['x', 'y'])

        min_n_cluster = min_n_cluster
        for k in range(min_n_cluster, max_n_clusters):
            
            # Initialize the clusterer with n_clusters value and a random generator
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto', max_iter=300)
            cluster_labels = kmeans.fit_predict(xy_T)
            
            #  check also dbscan 
            # db = DBSCAN(eps=(k*(0.01)), min_samples= 10).fit(xy_T)
            # labels = db.labels_
            
            # new columns for each kmeans cluster label in cluster_labels
            if visualize:
                df[f'KMeans_{k}'] = cluster_labels
                
            # calculate the silhouette score
            silhouette_avg = silhouette_score(xy_T, cluster_labels)
            
            silhouette_score_buffer.append(silhouette_avg)
            
            # calculatee the elbow method to find the optimal number of clusters
            # elbow method to find the optimal number of clusters
            # distortions.append(kmeans.inertia_)
            
        if visualize: 
            # plot the clusters
            fig, axs = plt.subplots(1, len(df.columns)-2, figsize=(40, 5))
            for i, ax in enumerate(fig.axes, start=2):
                ax.scatter(df['x'], df['y'], c=df.iloc[:, i], cmap='viridis')
                # set titile same as the column name
                ax.set_title(df.columns[i])
            plt.show()
        
        # # Use the 'kneed' library to identify the elbow point automatically
        # kn = KneeLocator(range(min_n_cluster, max_n_clusters), distortions, curve='convex', direction='decreasing')
        # num_cluster = kn.knee
            
        # choose the number of clusters with the highest silhouette score
        
        # if silhouette_score_buffer is empty, then return the minimum number of clusters
        if len(silhouette_score_buffer) == 0:
            return min_n_cluster
        else:
            return np.argmax(silhouette_score_buffer) + min_n_cluster

        

    def find_number_of_lanes(self, pointcloud, attributes, percentile=90, min_num_peaks=2, max_lane_width = 3.9, visualize=False):
        """
        Find the number of lanes based on the number of intensity peaks.
        - pointcloud: Open3D pointcloud object
        - attributes: The attributes that contains intensities values of the point cloud at index 1
        - min_num_peaks: The number of peaks to detect.
        - percentage: The percentage of the intensity to consider as the threshold.
        Returns:
        - num_lanes: The number of lanes detected based on intensity peaks.
        """
        # select only a section based on x coordinates
        selected_indices = (np.asarray(pointcloud.points)[:, 0] < -20) | (np.asarray(pointcloud.points)[:, 0] > 20)
        # extract the y coordinates and intensities
        y = np.asarray(pointcloud.points)[selected_indices, 1]
        intensities = np.asarray(attributes[selected_indices, 1])
        
        min_y, max_y = np.min(y), np.max(y)
        
        # pick the y bins that has the maximum intensity
        max_num_lanes = 6
        max_two_way_width = max_lane_width * max_num_lanes * 2
        
        # define number of y bins based on the maximum two way width
        y_bins = np.linspace(min_y, max_y, int(max_two_way_width))
        intensity_histogram, _ = np.histogram(y, bins=y_bins, weights=intensities)
        # percentile of the intensity
        threshold = np.percentile(intensity_histogram, percentile)
        
        # find the peaks in the intensity histogram that are larger than the threshold
        peaks_bin, _ = find_peaks(intensity_histogram, height=threshold, distance=max_lane_width/2)
        num_lanes = max(min_num_peaks, len(peaks_bin))
        # convert back the y with peak intensity to the original scale
        y_peak_coordinates = y_bins[peaks_bin]

        if visualize:
            # plot the intensity histogram with the peaks
            plt.plot(y_bins[1:], intensity_histogram)
            plt.plot(y_bins[peaks_bin], intensity_histogram[peaks_bin], "x")
            plt.xlabel('y-coordinate')
            plt.ylabel('intensity')
            plt.show()
        
        return num_lanes, y_peak_coordinates
    
    # define functions to calculate slope of each slope cluster
    def calculate_slope(self, pointcloud, attributes, num_slopes):
        """
        Calculate the slope of each cluster of points in the point cloud.
        - pointcloud: Open3D pointcloud object
        - attributes: The attributes that contains intensities values of the point cloud at index 1
        - num_slopes: The number of clusters to consider.
        Returns:
        - slopes: A dictionary where keys are cluster labels and values are the slope of the cluster.
        """
        # Extract the x and y coordinates
        xy = np.asarray(pointcloud.points)[:, :2]
        # Initialize a dictionary to store slopes
        slopes = {}
        # Iterate over each cluster
        for cluster_label in range(num_slopes):
            # Select points in the current cluster
            cluster_mask = attributes[:, -1] == cluster_label
            cluster_points = xy[cluster_mask]
            # Fit a linear regression line to the cluster
            x, y = cluster_points[:, 0], cluster_points[:, 1]
            coefs = np.polyfit(x, y, 1)
            slope = coefs[0]
            intercept = coefs[1]
            # the slope represents the slope of this cluster line and the intercept represents the y-intercept where the fitted line crosses the y-axis
            slopes[cluster_label] = (slope, intercept)
        return slopes

    def delete_orthogonal_slope(self, pcd, attributes, slopes_dict):
        """
        Deletes points and corresponding attributes where the slope is almost orthogonal to the x-axis.

        Parameters:
        - pcd: The Open3D point cloud object.
        - attributes: Attributes corresponding to each point in the point cloud.
        - slopes: A dictionary where keys are cluster labels and values are the slope and intercept of the cluster.
        - orthogonal_threshold: The threshold angle (in degrees) to consider a slope as nearly orthogonal.

        Returns:
        - filtered_pcd: The filtered Open3D point cloud object.
        - filtered_attributes: Filtered attributes corresponding to the filtered point cloud.
        """
        # calculate the slope of each cluster given the slope dictionary with the slope and intercept
        slopes = {label: slope for label, (slope, _) in slopes_dict.items()}
        # calculate the mean and standard deviation of the slopes
        slope_mean = np.mean(list(slopes.values()))
        slope_std = np.std(list(slopes.values()))
        # remove the slope that is outside of 90% of the probability distribution and the slope where its relative yaw angle is greater than 45 degress: tan(45) = 1
        slopes = {label: slope for label, slope in slopes.items() if abs(slope - slope_mean) < 1.645 * slope_std and abs(slope) < 1}
        
        print(f"mean slope: {slope_mean}, std slope: {slope_std}, delta number of slopes: {len(slopes_dict) - len(slopes)}")
        
        # Identify points to keep
        keep_indices = []
        for i, label in enumerate(attributes[:, -1]):  # Assuming last column is the cluster label
            if label in slopes:
                keep_indices.append(i)
        # Create a new Open3D point cloud for filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[keep_indices])
        filtered_attributes = attributes[keep_indices]
        
        return filtered_pcd, filtered_attributes


    def sort_point_cloud(self, pcd, attributes):
        """
        Sorts the point cloud based on the x-coordinate and assigns a new index to the points.

        Parameters:
        - pcd: The Open3D point cloud object to sort.
        - attributes: Attributes corresponding to each point in the point cloud.

        Returns:
        - sorted_pcd: The sorted Open3D point cloud object.
        - sorted_attributes: Sorted attributes corresponding to the sorted point cloud.
        """
        # Convert Open3D point cloud to NumPy array
        points = np.asarray(pcd.points)
        # Sort the points based on the x-coordinate
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        sorted_attributes = attributes[sorted_indices]

        # Create a new Open3D point cloud for sorted points
        sorted_pcd = o3d.geometry.PointCloud()
        sorted_pcd.points = o3d.utility.Vector3dVector(sorted_points)

        return sorted_pcd, sorted_attributes
                
    def points_to_img(self, pcd_with_attributes, img_size=(1000, 1000), point_size=1):
        '''
        Transform points to a binary image
        '''
        points_2d = np.asarray(pcd_with_attributes[0].points)[:, :2]
        intensities = pcd_with_attributes[1][:, 1]
                    
        img = np.zeros(img_size, dtype=np.uint8)
        # Normalize the points to fit in the image size
        norm_points = cv2.normalize(points_2d, None, alpha=0, beta=img_size[0]-1, norm_type=cv2.NORM_MINMAX).astype(np.int32)
        # Draw the points on the image
        for i,(x,y) in enumerate(norm_points):
            intensity = int(intensities[i])
            cv2.circle(img, (x, y), point_size, intensity, -1)  # 255 is the color (white)
        return img

    # Function to apply the Hough Transform and find lines
    def hough_transform(self, img, threshold=50, minLineLength=50, maxLineGap=10):
        '''
        Apply the Hough Transform to detect lines in a binary image.
        '''
        # Detect lines using the Hough Transform
        lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold, minLineLength, maxLineGap)
        return lines

     