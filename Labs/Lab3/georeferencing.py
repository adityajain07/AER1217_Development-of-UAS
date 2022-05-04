import numpy as np
from math import atan2
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import json
import math
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# add data labels in final plot
def add_text(x, y):
    for k in range(len(x)):
        pos_label = '{' + str(round(x[k], 2)) + ',' + str(round(y[k], 2)) + '}'
        plt.text(x[k], y[k], pos_label, horizontalalignment='center', verticalalignment='bottom')


# camera's field of view angle and aspect ratio
d_fov = 64 * np.pi / 180
AR_x = 16
AR_y = 9

# opening JSON file
detected = open('detection_info.json')

# load JSON object as a dictionary
data = json.load(detected)

# extracting frames and calculating circle center coordinates in world frame
circles_x = []
circles_y = []

for frame in data:
    # extracting pose and circle information
    pose = data[frame]['pose']
    circle = data[frame]['circles']['0']

    # body-to-camera transformation; camera parameters
    body_to_camera = np.array([[0.0, -1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    d_coeff = np.array([0.159121, -0.549986, 0.003377, 0.001831, 0.0])
    K = np.array([[674.84, 0.0, 298.61],
                  [0.0, 674.01, 189.32],
                  [0.0, 0.0, 1.0]])

    # body frame to camera frame
    pos_b = np.array([0, 0, 0, 1]).reshape(4, 1)
    pos_c = np.matmul(body_to_camera, pos_b)

    # normalized coordinates
    pos_n = np.array([0, 0, 1]).reshape(3, 1)

    # adding distortion
    r = math.sqrt(pos_n[0] ** 2 + pos_n[1] ** 2)
    dist_radial = 1 + d_coeff[0] * pow(r, 2) + d_coeff[1] * pow(r, 4) + d_coeff[4] * pow(r, 6)
    dist_tan_x = 2 * d_coeff[2] * pos_n[0] * pos_n[1] + d_coeff[3] * (pow(r, 2) + 2 * (pos_n[0] ** 2))
    dist_tan_y = d_coeff[2] * (pow(r, 2) + 2 * (pos_n[1] ** 2)) + 2 * d_coeff[3] * pos_n[0] * pos_n[1]

    x_d = dist_radial * pos_n[0] + dist_tan_x
    y_d = dist_radial * pos_n[1] + dist_tan_y
    pos_d = np.array([x_d, y_d, 1]).reshape(3, 1)

    # vehicle origin in image frame from bottom-left exis
    pos_i = np.dot(K, pos_d)
    x_v = int(pos_i[0])
    y_v = int(pos_i[1])

    # distance and angle between vehicle and circle centers in image frame
    circle_x = circle[0]
    circle_y = circle[1]
    dist_pixel = math.sqrt((x_v - circle_x) ** 2 + (y_v - circle_y) ** 2)
    angle = atan2(circle_y - y_v, circle_x - x_v)

    # circle center coordinates in world frame
    if dist_pixel <= 125:
        height = pose['z']
        diagonal = 2 * np.tan(d_fov / 2) * height
        ratio_const = np.sqrt((diagonal ** 2) / (AR_x ** 2 + AR_y ** 2))
        width = AR_x * ratio_const
        m_per_pixel = width / 640
        dist_m = dist_pixel * m_per_pixel

        # circle centres in world frame
        c_x_w = pose['x'] + dist_m * np.cos(angle + pose['yaw'])
        c_y_w = pose['y'] + dist_m * np.sin(angle + pose['yaw'])

        circles_x.append(c_x_w)
        circles_y.append(c_y_w)

# DBSCAN - density-based clustering
# 2D features
features = [circles_x, circles_y]
features = np.asarray(features)
features = np.transpose(features)

# scaling data for clustering, removes mean and scales to unit variance
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# segregate the data into clusters
dbscan = DBSCAN(eps=0.1, min_samples=25)
clusters = dbscan.fit_predict(scaled)

# plotting the circles detected
plt.rcParams.update({'font.size': 20})
plt.figure(1)
plt.scatter(circles_x, circles_y, s=2, alpha=0.5)
plt.title("Circles detected in the inertial frame")
plt.xlabel("x-coordinates in metres")
plt.ylabel("y-coordinates in metres")
plt.grid()

# Plot the clusters formed
colormap = ListedColormap(["lightgrey", "blue", "red", "green", "darkorange", "cyan", "magenta"])
plt.figure(2)
plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap=colormap, marker='o', edgecolors='k')
plt.title("6 clusters of circles' positions, outliers in grey")
plt.xlabel("x-coordinates in metres")
plt.ylabel("y-coordinates in metres")
plt.grid()

# calculating mean and standard deviation of each circle
centres_x = []
centres_y = []
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
for i in range(n_clusters):
    x_coord = [circles_x[x] for x in range(len(clusters)) if clusters[x] == i]
    y_coord = [circles_y[y] for y in range(len(clusters)) if clusters[y] == i]
    centres_x.append(np.mean(x_coord))
    centres_y.append(np.mean(y_coord))

# plotting the calculated circle positions
plt.figure(3)
plt.plot(centres_x, centres_y, 'og', markersize=15)
add_text(centres_x, centres_y)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("Position of the circle markers in the inertial frame")
plt.xlabel("x-coordinates in metres")
plt.ylabel("y-coordinates in metres")
plt.grid()
plt.show()

# closing JSON file
detected.close()
