import bagpy
import numpy as np
import math
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt

# LINEAR
b = bagreader('silinear.bag')

# extracting as dataframes
commanded = b.message_by_topic('/trajectory_generator')
current   = b.message_by_topic('/vicon/ARDroneCarre/ARDroneCarre')
df_commanded = pd.read_csv(commanded)
df_current   = pd.read_csv(current)

# extracting time column
time_com = df_commanded.iloc[:, 0]
time_cur = df_current.iloc[:, 0]

# calculating yaw from quaternions
x  = df_current.iloc[:, 9]
y  = df_current.iloc[:, 10]
z  = df_current.iloc[:, 11]
w  = df_current.iloc[:, 12]
t1 = +2.0 * (w * z + x * y)
t2 = +1.0 - 2.0 * (y * y + z * z)
yaw = np.zeros(np.shape(time_cur))
for i in range(len(t1)):
    yaw[i] = math.atan2(t1[i], t2[i])

# errors
time_fac  = len(time_cur)/len(time_com)
error_x   = np.zeros(np.shape(time_com))
error_y   = np.zeros(np.shape(time_com))
error_z   = np.zeros(np.shape(time_com))
error_yaw = np.zeros(np.shape(time_com))
for t in range(len(time_com)):
    error_x[t]   = df_commanded.iloc[t, 1] - df_current.iloc[int(t*time_fac), 6]
    error_y[t]   = df_commanded.iloc[t, 2] - df_current.iloc[int(t*time_fac), 7]
    error_z[t]   = df_commanded.iloc[t, 3] - df_current.iloc[int(t*time_fac), 8]
    error_yaw[t] = df_commanded.iloc[t, 4] - yaw[int(t*time_fac)]

# plotting
plt.rcParams.update({'font.size': 15})

fig1, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.x', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.x', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Linear motion along X-axis", fontsize=20)
plt.xlabel("time", fontsize=15)
plt.ylabel("time", fontsize=15)

fig2, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.y', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.y', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Linear motion along Y-axis", fontsize=20)

fig3, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.z', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.z', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Linear motion along Z-axis", fontsize=20)

fig4, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='angular.z', data=df_commanded)
ax[0].scatter(time_cur, yaw)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Plot for yaw in linear trajectory", fontsize=20)

plt.figure()
plt.plot(time_com, error_x,   linewidth=2)
plt.plot(time_com, error_y,   linewidth=2)
plt.plot(time_com, error_z,   linewidth=2)
plt.plot(time_com, error_yaw, linewidth=2)
plt.legend(['position in x', 'position in y', 'position in z', 'yaw angle'], fontsize=20)
plt.title("Errors in linear trajectory", fontsize=20)
plt.grid()
plt.show()


# CIRCULAR
b = bagreader('simulate.bag')

# extracting as dataframes
commanded = b.message_by_topic('/trajectory_generator')
current = b.message_by_topic('/vicon/ARDroneCarre/ARDroneCarre')
df_commanded = pd.read_csv(commanded)
df_current   = pd.read_csv(current)

# extracting time column
time_com = df_commanded.iloc[:, 0]
time_cur = df_current.iloc[:, 0]

# calculating yaw from quaternions
x  = df_current.iloc[:, 9]
y  = df_current.iloc[:, 10]
z  = df_current.iloc[:, 11]
w  = df_current.iloc[:, 12]
t1 = +2.0 * (w * z + x * y)
t2 = +1.0 - 2.0 * (y * y + z * z)

yaw = np.zeros(np.shape(time_cur))
for i in range(len(t1)):
    yaw[i] = math.atan2(t1[i], t2[i])

df_commanded.iloc[:, 6] = df_commanded.iloc[:, 6] - 2*np.pi

# errors
time_fac  = len(time_cur)/len(time_com)
error_x   = np.zeros(np.shape(time_com))
error_y   = np.zeros(np.shape(time_com))
error_z   = np.zeros(np.shape(time_com))
error_yaw = np.zeros(np.shape(time_com))
for t in range(len(time_com)):
    error_x[t]   = df_commanded.iloc[t, 1] - df_current.iloc[int(t*time_fac), 6]
    error_y[t]   = df_commanded.iloc[t, 2] - df_current.iloc[int(t*time_fac), 7]
    error_z[t]   = df_commanded.iloc[t, 3] - df_current.iloc[int(t*time_fac), 8]
    error_yaw[t] = df_commanded.iloc[t, 6] - yaw[int(t*time_fac)]

# plotting
plt.rcParams.update({'font.size': 15})

fig6, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.x', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.x', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Circular motion along X-axis", fontsize=20)

fig7, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.y', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.y', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Circular motion along Y-axis", fontsize=20)

fig8, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='linear.z', data=df_commanded)
ax[0].scatter(x='Time', y='transform.translation.z', data=df_current)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Circular motion along Z-axis", fontsize=20)

fig9, ax = bagpy.create_fig()
ax[0].scatter(x='Time', y='angular.z', data=df_commanded)
ax[0].scatter(time_cur, yaw)
plt.legend(['commanded', 'actual'], fontsize=20)
plt.title("Plot for yaw in circular trajectory", fontsize=20)

plt.figure()
plt.plot(time_com, error_x, linewidth=2)
plt.plot(time_com, error_y, linewidth=2)
plt.plot(time_com, error_z, linewidth=2)
plt.plot(time_com, error_yaw, linewidth=2)
plt.legend(['position in x', 'position in y', 'position in z', 'yaw angle'], fontsize=20)
plt.title("Errors in circular trajectory", fontsize=20)
plt.grid()
plt.show()
