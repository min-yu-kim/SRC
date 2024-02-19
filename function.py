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

def refining(data, refining_height):
    data = np.array(data)
    points = data[1:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    filtered_points = points[points[:, 2] > refining_height]
    filtered_points = pd.DataFrame(filtered_points)
    pptk.viewer(filtered_points)
    return filtered_points

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

def kmeans(data, cluster, algorithm, city):
        cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow',
                'chartreuse', 'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue',
                'steelblue', 'blue', 'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose',
                'lightskyblue', 'aquamarine', 'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray',
                'lightgoldenrodyellow', 'dodgerblue', 'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen',
                'sandybrown', 'tomato', 'deepskyblue', 'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue',
                'goldenrod', 'lightslategray', 'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise',
                'springgreen', 'mediumseagreen', 'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred',
                'olive', 'olivedrab', 'darkkhaki', 'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray',
                'mediumblue', 'navy', 'mediumslateblue', 'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid',
                'darkmagenta', 'saddlebrown']
        data = np.array(data)
        data_2d = data[:, :2]
        start = time.time()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(data_2d)
        label = kmeans.labels_
        print("time: ", time.time() - start)
        label = pd.DataFrame(label)
        label = label.to_numpy()
        data_label = np.append(data, label, axis=1)
        df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
        df_np = df_pd.to_numpy()
        # np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
        for n in range(int(max(df_pd['label']))):
            plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                        s=20, alpha=0.5)
        plt.axis('equal')
        plt.title(f'{algorithm}_{city}')
        # plt.savefig(f'plot_{algorithm}_{city}.png')
        plt.show()


def DBSCAN(data, eps, min_samples, algorithm, city):
    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    # cmap = ['navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy' ,'navy', 'navy', 'navy', 'navy']
    data = np.array(data)
    data_2d = data[:, :2]
    start = time.time()
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data_2d)
    label = dbscan.labels_
    print("time: ", time.time() - start)
    label = pd.DataFrame(label)
    label = label.to_numpy()
    data_label = np.append(data, label, axis=1)
    df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
    df_np = df_pd.to_numpy()
    np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
    for n in range(int(max(df_pd['label']))):
        plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                    s=2, alpha=0.5)
    plt.axis('equal')
    plt.title(f'{algorithm}_{city}')
    # plt.savefig(f'plot_{algorithm}_{city}.png')
    # plt.show()
    return label


def MiniBatchKMeans(data, cluster, algorithm, city):
    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    data = np.array(data)
    data_2d = data[:, :2]
    start = time.time()
    from sklearn.cluster import MiniBatchKMeans
    minibatchkmeans = MiniBatchKMeans(n_clusters=cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
    minibatchkmeans.labels_ = minibatchkmeans.fit_predict(data_2d)
    label = minibatchkmeans.labels_
    print("time: ", time.time() - start)
    label = pd.DataFrame(label)
    label = label.to_numpy()
    data_label = np.append(data, label, axis=1)
    df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
    df_np = df_pd.to_numpy()
    # np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
    for n in range(int(max(df_pd['label']))):
        plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                    s=20, alpha=0.5)
    plt.axis('equal')
    plt.title(f'{algorithm}_{city}')
    # plt.savefig(f'plot_{algorithm}_{city}.png')
    plt.show()


def GaussianMixture(data, cluster, algorithm, city):
    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    data = np.array(data)
    data_2d = data[:, :2]
    start = time.time()
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=cluster, covariance_type='full', random_state=100)
    gmm.fit(data_2d)
    gmm.labels_ = gmm.predict(data_2d)
    label = gmm.labels_
    print("time: ", time.time() - start)
    label = pd.DataFrame(label)
    label = label.to_numpy()
    data_label = np.append(data, label, axis=1)
    df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
    df_np = df_pd.to_numpy()
    # np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
    for n in range(int(max(df_pd['label']))):
        plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                    s=20, alpha=0.5)
    plt.axis('equal')
    plt.title(f'{algorithm}_{city}')
    # plt.savefig(f'plot_{algorithm}_{city}.png')
    plt.show()


def OPTICS(OPTICS_sample, data, algorithm, city):
    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown', 'red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown', 'red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown', 'red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown', 'red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown', 'red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    data = np.array(data)
    data_2d = data[:, :2]
    start = time.time()
    from sklearn.cluster import OPTICS
    optics_clustering = OPTICS(min_samples=3).fit(data_2d)
    label = optics_clustering.labels_
    print("time: ", time.time() - start)
    label = pd.DataFrame(label)
    label = label.to_numpy()
    data_label = np.append(data, label, axis=1)
    df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
    df_np = df_pd.to_numpy()
    # np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
    for n in range(int(max(df_pd['label']))):
        plt.scatter(df_pd.loc[df_pd['label'] == n, 'x' ], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                    s=20, alpha=0.5)
    plt.axis('equal')
    plt.title(f'{algorithm}_{city}')
    # plt.savefig(f'plot_{algorithm}_{city}.png')
    plt.show()


def MeanShift(data, algorithm, city):
    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    data = np.array(data)
    data_2d = data[:, :2]
    start = time.time()
    from sklearn.cluster import MeanShift
    meanshift = MeanShift()
    meanshift.fit(data_2d)
    meanshift_labels = meanshift.fit_predict(data_2d)
    label = meanshift_labels
    print("time: ", time.time() - start)
    label = pd.DataFrame(label)
    label = label.to_numpy()
    data_label = np.append(data, label, axis=1)
    df_pd = pd.DataFrame(data_label, columns=['x', 'y', 'z', 'label'])
    df_np = df_pd.to_numpy()
    # np.savetxt(f'filtered_{algorithm}_{city}.csv', df_np, delimiter=',')
    for n in range(int(max(df_pd['label']))):
        plt.scatter(df_pd.loc[df_pd['label'] == n, 'x'], df_pd.loc[df_pd['label'] == n, 'y'], color=cmap[n],
                    s=20, alpha=0.5)
    plt.axis('equal')
    plt.title(f'{algorithm}_{city}')
    # plt.savefig(f'plot_{algorithm}_{city}.png')
    plt.show()

def plot_3d(n, fig, data_3d, rotated_points):

    cmap = ['red', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'khaki', 'greenyellow', 'chartreuse',
            'limegreen', 'lime', 'turquoise', 'cyan', 'dodgerblue', 'deepskyblue', 'skyblue', 'steelblue', 'blue',
            'purple', 'fuchsia', 'orchid', 'pink', 'hotpink', 'magenta', 'mistyrose', 'lightskyblue', 'aquamarine',
            'paleturquoise', 'lavenderblush', 'cyan', 'coral', 'lightgray', 'lightgoldenrodyellow', 'dodgerblue',
            'mediumpurple', 'salmon', 'wheat', 'powderblue', 'palegreen', 'sandybrown', 'tomato', 'deepskyblue',
            'lemonchiffon', 'firebrick', 'turquoise', 'royalblue', 'skyblue', 'goldenrod', 'lightslategray',
            'lightgray', 'limegreen', 'cornflowerblue', 'darkturquoise', 'springgreen', 'mediumseagreen',
            'mediumaquamarine', 'palevioletred', 'indianred', 'mediumvioletred', 'olive', 'olivedrab', 'darkkhaki',
            'khaki', 'peachpuff', 'rosybrown', 'slategray', 'darkslategray', 'mediumblue', 'navy', 'mediumslateblue',
            'rebeccapurple', 'indigo', 'mediumorchid', 'darkorchid', 'darkmagenta', 'saddlebrown']
    max_heights = []
    for i, grid in enumerate(rotated_points):
        max_height = -np.inf
        for vertex in grid:
            dist = np.linalg.norm(data_3d[:, :2] - vertex, axis=1)
            closest_idx = np.argmin(dist)
            height = data_3d[closest_idx, 2]
            if height > max_height:
                max_height = height
        max_heights.append(max_height)
        # print(max_heights)
        # print(max(max_heights))
        # max_value = max(max_heights)
        # print(max_value)
        # print(grid)
        grid_matrix = np.zeros((len(grid), 3))


        for j, vertex in enumerate(grid):
            # print(grid_matrix)
            # print(vertex)
            grid_matrix[j, :] = [vertex[0], vertex[1], max_heights[i]]
            for k, l in zip(grid_matrix[:, 0], grid_matrix[:, 1]):
                if k != 0 or l != 0:
                    plt.scatter(k, l, c='k', s=2)
        # plt.scatter(x, y, c=label, s=20, alpha=0.5, cmap='rainbow')

        new_arr = []
        for i in range(0, len(grid_matrix), 2):
            if i + 1 < len(grid_matrix):
                new_arr.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
            else:
                new_arr.append(np.array([grid_matrix[i], grid_matrix[0]]))

        new_arr2 = []
        for i in range(1, len(grid_matrix), 2):
            if i + 1 < len(grid_matrix):
                new_arr2.append(np.array([grid_matrix[i], grid_matrix[i + 1]]))
            else:
                new_arr2.append(np.array([grid_matrix[i], grid_matrix[0]]))

        surface_points = []
        for arr in new_arr:
            vertices1 = [(x, y, 0) for x, y, _ in arr]
            vertices2 = [(x, y, z) for x, y, z in arr]
            vertices = np.vstack((vertices1, vertices2))
            surface_points += vertices.tolist()
            x, y, z = zip(*vertices)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n], showscale=False))

        for arr in new_arr2:
            vertices1 = [(x, y, 0) for x, y, _ in arr]
            vertices2 = [(x, y, z) for x, y, z in arr]
            vertices = np.vstack((vertices1, vertices2))
            surface_points += vertices.tolist()
            x, y, z = zip(*vertices)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3], color=cmap[n], showscale=False))

        vertices1 = [(x, y, 0) for x, y, _ in grid_matrix]
        vertices2 = [(x, y, z) for x, y, z in grid_matrix]
        vertices = np.vstack((vertices1, vertices2))
        surface_points += vertices.tolist()
        x, y, z = zip(*vertices1)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
        x, y, z = zip(*vertices2)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=cmap[n], showscale=False))
    return surface_points

def convex_hull_plot(coords):
    points = np.array(coords)
    hull = ConvexHull(points)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

def pca(n_xy):
    n_xy = n_xy.to_numpy()
    # plt.scatter(n_x, n_y)
    cov = np.cov(n_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    slope = largest_eigenvector[1] / largest_eigenvector[0]
    theta = np.arctan(slope)
    return theta

def matrix_rotate(theta, n_xy):
    n_xy = n_xy.to_numpy()
    n_x = n_xy[:, 0]
    n_y = n_xy[:, 1]
    if theta > np.pi / 2:
        theta = -theta
    n_x_new = n_x * math.cos(theta) + n_y * math.sin(theta)
    n_y_new = -n_x * math.sin(theta) + n_y * math.cos(theta)
    n_x_new_max = max(n_x_new)
    n_y_new_max = max(n_y_new)
    n_x_new_min = min(n_x_new)
    n_y_new_min = min(n_y_new)

    n_x_ver1 = n_x_new_max * math.cos(theta) - n_y_new_max * math.sin(theta)
    n_y_ver1 = n_x_new_max * math.sin(theta) + n_y_new_max * math.cos(theta)
    n_x_ver2 = n_x_new_max * math.cos(theta) - n_y_new_min * math.sin(theta)
    n_y_ver2 = n_x_new_max * math.sin(theta) + n_y_new_min * math.cos(theta)
    n_x_ver3 = n_x_new_min * math.cos(theta) - n_y_new_max * math.sin(theta)
    n_y_ver3 = n_x_new_min * math.sin(theta) + n_y_new_max * math.cos(theta)
    n_x_ver4 = n_x_new_min * math.cos(theta) - n_y_new_min * math.sin(theta)
    n_y_ver4 = n_x_new_min * math.sin(theta) + n_y_new_min * math.cos(theta)
    coords = [[n_x_ver1, n_y_ver1], [n_x_ver2, n_y_ver2], [n_x_ver4, n_y_ver4], [n_x_ver3, n_y_ver3]]
    return coords

def grid_rotate(coords, theta):
    points = np.array(coords)
    hull = ConvexHull(points)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    #for simplex in hull.simplices:
     #   plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    x, y = zip(*coords)
    center_x = (max(x) + min(x)) / 2  # x 중심점
    center_y = (max(y) + min(y)) / 2  # y 중심점
    pivot = (center_x, center_y)
    # origin = np.array(coords[0])
    coords = np.array(coords) - pivot

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotated_coords = np.dot(coords, rotation_matrix)
    rotated_coords += pivot

    # print(rotated_coords)
    # rotated_points = np.array(rotated_coords)
    # hull2 = ConvexHull(rotated_points)
    # # plt.plot(points[:, 0], points[:, 1], 'o')
    # for simplex in hull2.simplices:
    #     plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], 'r-')

    # plt.axis('equal')
    # plt.show()
    return rotated_coords

def grid_generate(rotated_coords, grid_size):
    # rotated_coords = rotated_coords.transpose()
    x = [coord[0] for coord in rotated_coords]
    y = [coord[1] for coord in rotated_coords]

    start_x = min(x)
    start_y = min(y)

    if max(x) - min(x) < grid_size or max(y) - min(y) < grid_size:
        rotated_coords = rotated_coords.tolist()
        return [rotated_coords]
    dx = grid_size  # x 좌표 간격
    dy = grid_size  # y 좌표 간격
    grid_coords = []
    # 그리드 생성
    for i in range(int((max(x) - start_x) / dx)+1):
        for j in range(int((max(y) - start_y) / dy)+1):
            # 그리드 좌표 계산
            x1 = start_x + i * dx
            x2 = start_x + (i + 1) * dx
            y1 = start_y + j * dy
            y2 = start_y + (j + 1) * dy

            # 그리드 좌표 저장
            grid_coords.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    grid_coords = np.array(grid_coords)
    # for grid in grid_coords:
    #     hull3 = ConvexHull(grid)
    #     for simplex in hull3.simplices:
    #         plt.plot(grid[simplex, 0], grid[simplex, 1], 'r-')

    # plt.axis('equal')
    # plt.show()
    return grid_coords

def grid_rotate_to_origin(rotated_coords, grid_coords, theta):
    # grid_points = np.array(grid_coords)
    # hull = ConvexHull(points)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    x, y = zip(*rotated_coords)
    center_x = (max(x) + min(x)) / 2  # x 중심점
    center_y = (max(y) + min(y)) / 2  # y 중심점
    pivot = (center_x, center_y)
    # grid_coords = np.array(grid_coords)
    rotated_points = []  # 회전된 좌표를 저장할 리스트
    for grid in grid_coords:
        grid_points = np.array(grid) - pivot

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rotated_coords = np.dot(grid_points, rotation_matrix)
        rotated_coords += pivot
        rotated_coords = np.array(rotated_coords)

        hull3 = ConvexHull(rotated_coords)
        for simplex in hull3.simplices:
            plt.plot(rotated_coords[simplex, 0], rotated_coords[simplex, 1], 'b-')


        # rotated_points.append(rotated)
        rotated_points.append(rotated_coords.tolist())

    # rotated_coords.append(rotated_coords)
    # plt.axis('equal')
    # plt.show()
    return rotated_points

def select_intersecting_grids(rotated_points, x, y):
    selected_grids = []
    for grid in rotated_points:
        grid_points = np.array(grid)
        # x, y 좌표를 이용하여 그리드와 겹치는 영역이 있는지 확인합니다.
        inside = np.logical_and(x > grid_points[:, 0].min(), x < grid_points[:, 0].max())
        inside = np.logical_and(inside, y > grid_points[:, 1].min())
        inside = np.logical_and(inside, y < grid_points[:, 1].max())
        # 겹치는 영역이 있으면 그리드를 선택합니다.
        if np.sum(inside) > 0:
            selected_grids.append(grid)
    return selected_grids

def plot_corridor(p1, p2, r, fig):
     t = np.linspace(0, 10, 100)
     x = p1[0] + t * (p2[0] - p1[0])
     pos_list = []  # 이동하는 위치를 저장할 리스트

     for i in range(len(x)):
         pos = p1 + (p2 - p1) * (i + 1) / (len(x))
         pos_list.append(pos.tolist())  # 현재 위치를 리스트에 추가
         theta = np.linspace(0, 2 * np.pi, 100)
         phi = np.linspace(0, np.pi, 100)
         x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
         y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
         z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
         colors = np.zeros_like(z_sphere)
         color_val = "gray"
         fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
                                      colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))

def max_find(data_3d, coords):
    max_heights = []
    max_height = -np.inf
    for vertex in coords:
        dist = np.linalg.norm(data_3d[:, :2] - vertex, axis=1)
        closest_idx = np.argmin(dist)
        height = data_3d[closest_idx, 2]
        if height > max_height:
            max_height = height
    max_heights.append(max_height)
    # print(max_heights)
    # print(max(max_heights))
    max_value = max(max_heights)
    # print(max_value)
    return max_value

def plot_corridors(p1, p2, r, fig, new_coords):
    t = np.linspace(0, 1, 100)
    x = p1[0] + t * (p2[0] - p1[0])
    faces = [(0, 1, 2, 3),  # bottom
             (4, 5, 6, 7),  # top
             (0, 1, 5, 4),  # front
             (2, 3, 7, 6),  # back
             (1, 2, 6, 5),  # right
             (0, 3, 7, 4)]  # left
    for face in faces:
        p1 = np.array(new_coords[face[0]])
        p2 = np.array(new_coords[face[1]])
        p3 = np.array(new_coords[face[2]])
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        n = np.cross(v1, v2)
        for i in range(len(x)):
            pos = p1 + (p2 - p1) * (i + 1) / (len(x))
            dist = np.dot(n, np.array(pos) - p1) / np.linalg.norm(n)
            theta = np.linspace(0, 2 * np.pi, 100)
            phi = np.linspace(0, np.pi, 100)
            x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
            y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
            z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
            colors = np.zeros_like(z_sphere)
            if abs(dist) <= r:
                color_val = "red"
            else:
                color_val = "gray"
            fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
                                     colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))

def make_pos(point1, point2):
    t = np.linspace(0, 1, 100)
    x = point1[0] + t * (point2[0] - point1[0])
    pos_list = []
    for i in range(len(x)):
        pos = point1 + (point2 - point1) * (i + 1) / (len(x))
        pos_list.append(pos)
    return pos_list

def is_collision(new_coords, r, pos_list):
    xy_points = [(p[0], p[1]) for p in new_coords]
    hull_xy = ConvexHull(np.array(xy_points))
    xy_polygon = Polygon(np.array(xy_points)[hull_xy.vertices])
    yz_points = [(p[1], p[2]) for p in new_coords]
    hull_yz = ConvexHull(np.array(yz_points))
    yz_polygon = Polygon(np.array(yz_points)[hull_yz.vertices])
    xz_points = [(p[0], p[2]) for p in new_coords]
    hull_xz = ConvexHull(np.array(xz_points))
    xz_polygon = Polygon(np.array(xz_points)[hull_xz.vertices])

    crush_pos = []
    normal_pos = []
    for pos in pos_list:
        #print(pos)
        point = Point(pos[0], pos[1], pos[2])
        xy_circle = Point(point.x, point.y).buffer(r)
        yz_circle = Point(point.y, point.z).buffer(r)
        xz_circle = Point(point.x, point.z).buffer(r)
        #if xy_circle.intersects(xy_polygon) and yz_circle.intersects(yz_polygon) and xz_circle.intersects(xz_polygon):
        if xy_polygon.intersects(xy_circle) and yz_polygon.intersects(yz_circle) and xz_polygon.intersects(xz_circle):
            collision = True
            crush_pos.append(pos.tolist())
        else:
            collision = False
            if pos.tolist() not in normal_pos:
                normal_pos.append(pos.tolist())
    return [collision, crush_pos, normal_pos]

def plot_corridor_crush(r, fig, crush_pos):
    for pos in crush_pos:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
        colors = np.zeros_like(z_sphere)
        color_val = "red"
        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
                                      colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))

def plot_corridor_normal(r, fig, normal_pos):
    for pos in normal_pos:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
        colors = np.zeros_like(z_sphere)
        color_val = "gray"
        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
                                      colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))

def plot_corridors(p1, p2, r, fig):
    t = np.linspace(0, 1, 100)
    x = p1[0] + t * (p2[0] - p1[0])

    for i in range(len(x)):
        pos = p1 + (p2 - p1) * (i + 1) / (len(x))
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        x_sphere = pos[0] + r * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = pos[1] + r * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = pos[2] + r * np.outer(np.ones(100), np.cos(phi))
        colors = np.zeros_like(z_sphere)
        color_val = "gray"
        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=colors,
                                     colorscale=[[0, color_val], [1, color_val]], showscale=False, opacity=0.3))

def sample_obstacle(obstacle, sampling_distance=1):
    sampled_points = []
    for i in range(len(obstacle) - 1):
        x1, y1 = obstacle[i]
        x2, y2 = obstacle[i + 1]
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        num_samples = int(distance / sampling_distance)
        if num_samples > 0:
            x_step = (x2 - x1) / num_samples
            y_step = (y2 - y1) / num_samples
            for j in range(num_samples):
                x = x1 + j * x_step
                y = y1 + j * y_step
                sampled_points.append((x, y))
    return sampled_points