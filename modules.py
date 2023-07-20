import numpy as np
import open3d as o3d
import pptk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import convex_hull_plot_2d
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from modules import data_open

def data_open():
    data = np.loadtxt('filtered_portland.txt')
    data_3d = data[:, :3]
    return data_3d