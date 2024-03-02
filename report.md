# Ego-Lane Detection Using LiDAR Pointclouds

## Authors
Chieh Fei (Jeffee) Hsiung, KU Leuven

## Abstract
This report details the development and implementation of an ego lane detection algorithm using LiDAR pointcloud data. By fitting a 3-degree polynomial to detected lane lines, the algorithm emphasizes robust preprocessing, lane point classification, and curve fitting techniques. This approach aims to accurately detect the ego lane with conventional filtering and polynomial fitting methods that is crucial for the navigation and control of autonomous vehicles.

## 1. Introduction
Accurate detection of ego lane lines is fundamental for the autonomous navigation of vehicles. With advancements in LiDAR technology, high-resolution pointclouds provide a detailed representation of the environment, including road lanes. This report presents an algorithm designed to process such pointcloud data, focusing on detecting ego lanes through a combination of classical data preprocessing, feature extraction, and mathematical modeling.

## 2. Methodology
The algorithm is structured into four main stages: data preprocessing, lane detection, lane marking, and lane fitting. Each stage employs specific techniques and algorithms to isolate lane lines from the pointcloud data and model them accurately.

### 2.1 Data Preprocessing
Data preprocessing is a crucial step in preparing LiDAR pointclouds for the detection of ego lanes. Our algorithm employs a series of sophisticated techniques to filter and refine the raw data, ensuring that subsequent lane detection stages operate on the most relevant and accurate representations of the driving environment.

- **Loading Pointclouds with Attributes**: Utilizing Open3D, pointclouds are loaded from binary files, encapsulating spatial information (x, y, z coordinates) and non-spatial attributes (intensity and lidar beam id) crucial for ego lane detection. This initial step ensures a comprehensive dataset is prepared, including assigning a unique scene identifier to each pointcloud for efficient processing and analysis.

- **Voxel Downsampling**: To manage computational efficiency without compromising the detail essential for lane detection, voxel downsampling is applied. This technique averages the points within each voxel, significantly reducing the pointcloud density and making the dataset more manageable for processing.

- **Intensity-Based Filtering and Z-Filter**: The algorithm prioritizes points based on their intensity values, which are indicative of the reflective properties of lane markings. Additionally, a z-filter is applied to exclude points that are significantly above the road surface, focusing the analysis on the area where lane markings are present.

- **Standardization of Pointcloud Data**: Critical to our preprocessing is the standardization of pointcloud data using a StandardScaler. This step normalizes the x, y, z coordinates of the points, mitigating the influence of outliers and ensuring a uniform scale for analysis. The transformation enhances the algorithm's ability to identify patterns and features relevant to lane detection across varied datasets.

- **Normal Estimation**: For datasets that may benefit from surface reconstruction or advanced analysis, normal estimation is conducted on downsampled pointclouds. This process calculates the orientation of the surface at each point, facilitating more sophisticated geometric analyses and improving the algorithm's robustness to complex road geometries.

### 2.2 Lane Detection
Lane detection is a critical phase where the preprocessed data is analyzed to identify and model the lane boundaries. This phase leverages several sophisticated methods and logical reasoning to ensure the accurate detection and representation of lanes.

- **DBSCAN Clustering for Initial Lane Point Identification**: Utilizing the DBSCAN algorithm, the system clusters pointcloud data based on spatial proximity and intensity. This method is particularly effective in segregating closely packed points representing lane markings from the surrounding noise. By adjusting parameters such as eps (the maximum distance between two samples for one to be considered as in the neighborhood of the other) and min_samples (the number of samples in a neighborhood for a point to be considered as a core point), we refine the clustering to accurately capture the lane points.

- **Ground Plane Segmentation with RANSAC**: To further isolate lane markings, we employ RANSAC for ground plane segmentation. This step effectively separates the lane markings (which are above the ground plane) from the ground points. By distinguishing between inliers (ground points) and outliers (potential lane points), we enhance the focus on the relevant features for lane detection.

- **Intensity Threshold Learning**: An innovative step in our system is the dynamic learning of an intensity threshold to filter lane points. By analyzing the clustered points, the algorithm determines an optimal intensity threshold that distinguishes lane markings, typically characterized by higher reflectivity, from other road and non-road elements.

- **Optimization of K-Means Clustering**: Post-initial clustering and filtering, K-Means clustering is optimized to segment the lane points into distinct lanes. Through silhouette analysis, the algorithm determines the optimal number of clusters (lanes), ensuring each lane is distinctly identified. This step is crucial for handling multiple lane scenarios and complex road geometries.

- **Peak Finding for Lane Estimation**: To estimate the number of lanes, the algorithm analyzes intensity peaks along the y-axis of the top view pointcloud. This method assumes that lanes are separated by regions of lower point density, allowing for an accurate estimation of lane numbers based on detected peaks.

- **Slope Calculation and Orthogonal Slope Filtering**: For each identified lane cluster, the algorithm calculates the slope, aiding in the discrimination of lane directions and orientations. Clusters with slopes nearly orthogonal to the expected lane direction are filtered out, focusing the analysis on plausible lane orientations.

- **Lane Modeling with Polynomial Regression**: The final step involves fitting a 3-degree polynomial to the points in each lane cluster, accurately modeling the lane boundaries. This approach allows for the representation of both straight and curved lanes, adapting to various road conditions.

- **Visualization and Verification**: Throughout the lane detection process, visualization tools are employed to inspect and verify the accuracy of detected lanes. This includes transforming points to a binary image and employing the Hough Transform to detect lines, providing a visual confirmation of the algorithm's effectiveness.

### 2.3 Lane Marking
Lane marking is a crucial step following lane detection, aimed at defining precise boundaries for where the lanes are located within the point cloud data. This process leverages the detected slopes and segments of the lanes to create a structured grid system that outlines the expected locations of lane boundaries. The LaneMarker class plays a pivotal role in this step.

- **Grid Dictionary Creation**: The `create_grid_dict` method within the LaneMarker class constructs a grid dictionary based on the x-coordinates range, considering each lane segment's slope. This dictionary maps grid coordinates to adjusted grid bounds, taking into account the maximum lane width and the number of lanes detected. This structured approach allows for the dynamic adaptation of lane marking to the road's geometry and the vehicle's perspective.

- **Slope and Intercept Adjustments**: By analyzing the slopes and intercepts of each lane segment, the algorithm dynamically adjusts the grid boundaries to account for variations in lane orientation and curvature. This ensures that the grid accurately reflects the physical layout of the road, facilitating more precise lane marking.

- **Visualization and Verification**: If enabled, the visualization feature plots the point cloud alongside the defined grid bounds, providing a visual confirmation of the lane marking process. This step is invaluable for debugging and verifying that the lane boundaries are accurately captured by the grid system.

- **LiDAR Data Filtering by Grid**: Post grid definition, the `filter_lidar_data_by_grid` method assigns LiDAR data points to the corresponding grid cells. This filtering process isolates points within specific boundaries, enabling focused analysis and modeling of lane lines within these predefined areas.

The lane marking step, with its grid-based approach, significantly enhances the lane detection algorithm's ability to delineate lane boundaries accurately. By considering the road's geometry and adjusting for lane slopes, the algorithm ensures that lane markings are not only detected but also precisely defined within the context of the road environment. This additional layer of processing strengthens the algorithm's applicability to real-world driving scenarios, where accurate lane marking is essential for safe navigation.

### 2.4 Polynomial Regression Lane Fitting
An essential component of our lane detection algorithm is the Polynomial Regression class, designed to model lane boundaries as 3-degree polynomials. This class provides a structured approach to fitting, predicting, and evaluating the performance of polynomial models based on LiDAR data points identified as lane markers.

- **Model Fitting**: The `fit` method employs NumPy's `polyfit` function to determine the coefficients of a polynomial that best fits the lane points in a least squares sense. This method allows for the flexible modeling of lane shapes, accommodating straight and curved road segments alike.

- **Prediction and Scoring**: With the fitted model, the `predict` method calculates the expected y-values for a given range of x-values, facilitating the visualization and evaluation of the lane lines. The `score` method extends this by computing the mean squared error (MSE) between the predicted lane positions and the actual lane positions, offering a quantitative measure of model accuracy.

- **Cost Function for Lane Fidelity**: A novel aspect of our approach is the cost function defined within the Polynomial Regression class. This function, `cost`, not only considers the mean squared error between the modeled and actual lane widths across a range of x-values but also includes a penalty term for deviations from parallelism between the left and right lane boundaries. The aim is to ensure that the detected lanes are not only accurate in terms of position but also consistent in width and parallel to each other, which is crucial for reliable lane following by autonomous vehicles.

The cost function calculates the perpendicular (orthogonal) distance between the two lane lines across a series of points along the x-axis, aiming to maintain a consistent lane width reflective of real-world conditions. A parallelism weight is applied to penalize significant differences in the slopes of the left and right lane models, promoting parallel lane lines as expected in a correctly detected lane scenario.

## 3. Results
The algorithm was tested on multiple datasets obtained under various environmental conditions. The preprocessing and lane detection methods demonstrated high effectiveness in isolating and accurately modeling lane lines. Comparative analysis with ground truth data showed that the polynomial fits closely align with the actual lanes, indicating the algorithm's reliability for real-world applications.

## 4. Discussion
The integration of the Polynomial Regression class into our lane detection algorithm represents a significant step forward in the modeling of lane boundaries from LiDAR pointcloud data. By combining traditional polynomial fitting techniques with a custom cost function that emphasizes lane fidelity and parallelism, we have developed a robust method for ego lane detection that is sensitive to the nuances of real-world driving scenarios. Future work may explore the application of this approach to more complex road geometries, including intersections and multi-lane roads, as well as its integration with vehicle control systems for autonomous navigation.

## 5. Conclusion
This report introduced a systematic approach to ego lane detection from LiDAR pointclouds, highlighting the importance of data preprocessing and advanced polynomial fitting. The methodology and results underscore the algorithm's applicability to autonomous vehicle navigation, offering a foundation for further research and development in this area.

## References
- [1] Your references here.
