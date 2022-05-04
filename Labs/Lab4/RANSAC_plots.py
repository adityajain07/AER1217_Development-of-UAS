import matplotlib.pyplot as plt

k_iter = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
error_k = [1.23, 1.04, 0.83, 1.03, 0.91, 0.98, 0.99, 1.02, 1.07, 1.14, 1.13]
inlier_k = [65.86, 67.53, 68.33, 69.50, 69.67, 70.20, 70.61, 70.70, 70.76, 71.10, 71.17]

err_thres = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
error_e = [1.58, 1.21, 1.02, 0.70, 1.73, 1.15, 1.62, 1.36]
inlier_e = [41.38, 53.42, 60.86, 66.29, 70.14, 72.45, 74.47, 76.70]
            
plt.rcParams.update({'font.size': 20})

plt.figure(1)
plt.plot(k_iter[2], error_k[2], 'o', markersize=11)
plt.plot(k_iter, error_k, 'o-')
plt.xlabel("RANSAC iterations")
plt.ylabel("mean absolute error from ground truth")
plt.legend(["chosen k with least error"])
plt.ylim([0.0, 2.0])
plt.grid()

plt.figure(2)
plt.plot(k_iter, inlier_k, 'o-')
plt.xlabel("RANSAC iterations")
plt.ylabel("mean inliers' percentage")
plt.ylim([0.0, 100.0])
plt.grid()

plt.figure(3)
plt.plot(err_thres[3], error_e[3], 'o', markersize=11)
plt.plot(err_thres, error_e, 'o-')
plt.xlabel("error threshold value")
plt.ylabel("mean absolute error from ground truth")
plt.legend(["chosen threshold with least error"])
plt.ylim([0.0, 3.0])
plt.grid()

plt.figure(4)
plt.plot(err_thres, inlier_e, 'o-')
plt.xlabel("error threshold value")
plt.ylabel("mean inliers' percentage")
plt.ylim([0.0, 100.0])
plt.grid()

plt.show()

