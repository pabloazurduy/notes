# This script generates random points for users and stores, converts the points from degrees to radians,
# measures the time taken by BallTree and cKDTree to find all stores within a certain radius of each eater,
# and compares the results of both methods.

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import time

# Generate random points for users (7 million points)
users = np.random.uniform(low=-90, high=90, size=(7_000_000, 2))  # latitudes are between -90 and 90
users[:, 1] = np.random.uniform(low=-180, high=180, size=7_000_000)  # longitudes are between -180 and 180

# Generate random points for list B (10,000 points)
stores = np.random.uniform(low=-90, high=90, size=(10_000, 2))  # latitudes are between -90 and 90
stores[:, 1] = np.random.uniform(low=-180, high=180, size=10_000)  # longitudes are between -180 and 180

# Convert the points from degrees to radians as both BallTree and cKDTree expect the data in radians
users_rad = np.deg2rad(users)
stores_rad = np.deg2rad(stores)

# Define the radius X in kilometers
X = 1000
# Convert the radius to radians
X_rad = X / 6371.  # where 6371 is the Earth's radius in km

# Measure time taken by BallTree
start_time = time.time()
tree = BallTree(stores_rad, leaf_size=15, metric='haversine')
indices_balltree = tree.query_radius(users_rad, r=X_rad, return_distance=False, )
end_time = time.time()
balltree_time = end_time - start_time
print(f"BallTree took {balltree_time} seconds")

# Measure time taken by cKDTree
start_time = time.time()
tree = cKDTree(stores_rad)
indices_ckdtree = tree.query_ball_point(users_rad, r=X_rad, workers=-1)
end_time = time.time()
ckdtree_time = end_time - start_time
print(f"cKDTree took {ckdtree_time} seconds")

# Compare the results
print(f"Time difference: {abs(balltree_time - ckdtree_time)} seconds")
