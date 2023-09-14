import open3d as o3d
import pptk
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import pptk
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, Point


class Preprocessor:
    def __init__(self, data):
        self.data = data

    def remove_outlier(data, nb_neighbors, std_ratio):
        data = np.array(data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        inliers_df = pd.DataFrame(inlier_cloud.points, columns=["x", "y", "z"])
        return inliers_df

preprocessor = Preprocessor(data)
nb_neighbors = 20
std_ratio = 2.0
inliers_df = preprocessor.remove_outlier(nb_neighbors, std_ratio)