import numpy as np
import pandas as pd

# Assuming each point has 5 attributes (x, y, z, intensity, ring)
num_point_attributes = 5

# Replace with the path to your .bin file
lidar_path = "./pointclouds/1553670684146333606.bin"

points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_point_attributes)

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity', 'beamNumber'])

# # Display the first few rows of the DataFrame
# print(df.head())

# save the DataFrame to a csv file
df.to_csv("pointclouds.csv", index=False)