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

    def learn_intensity_threshold(self, attributes, percentage=0.10):
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
        N = max(1, int(round(len(mean_intensity) * percentage)))  # Use top 75% of clusters and cast to int
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
            # seed of 10 for reproducibility.
            # take the nearest 2D points of max distance of 0.05 meters
            kmeans = KMeans(n_clusters=k, random_state=10, n_init=10, max_iter=300)
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
            
        # choose the number of clusters with the highest silhouette score
        num_cluster = np.argmax(silhouette_score_buffer) + min_n_cluster
        
        # # Use the 'kneed' library to identify the elbow point automatically
        # kn = KneeLocator(range(min_n_cluster, max_n_clusters), distortions, curve='convex', direction='decreasing')
        # num_cluster = kn.knee
        
        return num_cluster
    

    def find_number_of_lanes(self, pointcloud, attributes, percentile=90, min_num_peaks=2):
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
        lane_width = 3.75
        max_num_lanes = 6
        max_two_way_width = lane_width * max_num_lanes * 2
        
        # define number of y bins based on the maximum two way width
        y_bins = np.linspace(min_y, max_y, int(max_two_way_width))
        intensity_histogram, _ = np.histogram(y, bins=y_bins, weights=intensities)
        # percentile of the intensity
        threshold = np.percentile(intensity_histogram, percentile)
        # find the peaks in the intensity histogram that are larger than the threshold
        peaks_bin, _ = find_peaks(intensity_histogram, height=threshold, distance=lane_width/3)
        num_lanes = max(min_num_peaks, len(peaks_bin))
        # convert back the y with peak intensity to the original scale
        y_peak_coordinates = y_bins[peaks_bin]
        print(f"Number of lanes detected: {num_lanes} at y-coordinates: {y_peak_coordinates}")
        # # plot the intensity histogram with the peaks
        # plt.plot(y_bins[1:], intensity_histogram)
        # plt.plot(y_bins[peaks_bin], intensity_histogram[peaks_bin], "x")
        # plt.xlabel('y-coordinate')
        # plt.ylabel('intensity')
        # plt.show()
        
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
            slopes[cluster_label] = slope
        return slopes

    def delete_orthogonal_slope(self, pcd, attributes, slopes):
        """
        Deletes points and corresponding attributes where the slope is almost orthogonal to the x-axis.

        Parameters:
        - pcd: The Open3D point cloud object.
        - attributes: Attributes corresponding to each point in the point cloud.
        - slopes: A dictionary where keys are cluster labels and values are the slope of the cluster.
        - orthogonal_threshold: The threshold angle (in degrees) to consider a slope as nearly orthogonal.

        Returns:
        - filtered_pcd: The filtered Open3D point cloud object.
        - filtered_attributes: Filtered attributes corresponding to the filtered point cloud.
        """
        # mean slope
        mean_slope = np.mean(list(slopes.values()))
        # standard deviation of the slope
        std_slope = np.std(list(slopes.values()))
        # remove the slope that is too far from the mean
        slopes = {label: slope for label, slope in slopes.items() if abs(slope - mean_slope) < 1.1 * std_slope}
        print(f"mean slope: {mean_slope}, std slope: {std_slope}, slopes Labels: {slopes} ")
        
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

    def segregate_points_based_on_lanes(self, filtered_points, filtered_attributes, y_peak_coordinates, threshold_intensity, num_bins=100):
        """
        Improved segregation of points based on lanes by analyzing each X-coordinate bin for lane detection
        based on Y-coordinates and intensity.

        Parameters:
        - filtered_points: open3D pointcloud object
        - filtered_attributes: attributes corresponding to filtered_points, including intensity.
        - y_peak_coordinates: Y-coordinates representing the peak locations of lanes.
        - threshold_intensity: Intensity threshold to consider a point as part of a lane.
        - num_bins: Number of bins to divide the X-coordinate range into.

        Returns:
        - lane_groups: A dictionary where keys are tuples (bin_index, lane_index) and values are numpy arrays of points assigned to a lane.
        """
        # Initialize the result dictionary
        lane_groups = {}
        # to np.array
        filtered_points = np.asarray(filtered_points.points)
        filtered_attributes = np.asarray(filtered_attributes)
        # Extract X, Y coordinates and intensity from filtered points and attributes
        X, Y = filtered_points[:, 0], filtered_points[:, 1]
        intensity = filtered_attributes[:, 1]  # Assuming intensity is at index 1

        # Define bins for X-coordinate
        x_min, x_max = np.min(X), np.max(X)
        bins = np.linspace(x_min, x_max, num_bins + 1)

        # Bin points based on X-coordinate
        bin_indices = np.digitize(X, bins) - 1  # Adjusting indices to be 0-based

        # Analyze points in each bin
        for bin_index in range(num_bins):
            # Select points and attributes in current bin
            in_bin_mask = bin_indices == bin_index
            points_in_bin = filtered_points[in_bin_mask]
            intensity_in_bin = intensity[in_bin_mask]
            
            lane_width = 3.9  # Assuming a constant lane width
            # Skip empty bins
            if points_in_bin.size == 0:
                continue
            
            # in the y_peak_coordinates, if two peaks are at proximity of each other, then they are the same lane, combine them by calculating the average
            y_peak_coordinates = np.sort(np.array(y_peak_coordinates))
            
            # merge_close_peaks
            i = 0
            # Iterate over the peaks; since the list might change size, use a while loop
            while i < len(y_peak_coordinates) - 1:
                # Check if the next peak is within lane_width/3 of the current peak
                if y_peak_coordinates[i + 1] - y_peak_coordinates[i] <= lane_width / 3:
                    # Average the current peak and the next one
                    avg_peak = np.mean([y_peak_coordinates[i], y_peak_coordinates[i + 1]])
                    # Replace the two peaks with their average
                    y_peak_coordinates = np.delete(y_peak_coordinates, [i, i + 1])
                    y_peak_coordinates = np.insert(y_peak_coordinates, i, avg_peak)
                    # No need to increment i, as we want to check the next set of peaks against the newly formed average
                else:
                    # Only increment if no merge was done, to move to the next peak
                    i += 1
            
            # Assign points to lanes based on Y-coordinates and intensity
            for lane_index, y_peak in enumerate(y_peak_coordinates):
                # Define lane boundaries based on peak Y-coordinate
                y_min, y_max = y_peak - lane_width/2, y_peak + lane_width/2

                # Identify points within the lane boundaries and above intensity threshold
                lane_mask = (points_in_bin[:, 1] >= y_min) & (points_in_bin[:, 1] <= y_max) & (intensity_in_bin > threshold_intensity)
                points_in_lane = points_in_bin[lane_mask]
                    
                # Update lane groups if any points are identified
                if points_in_lane.size > 0:
                    lane_groups[(bin_index, lane_index)] = points_in_lane

        return lane_groups
                
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

     