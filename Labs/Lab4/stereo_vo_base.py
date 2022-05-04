"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
from turtle import position
import numpy as np
import numpy.linalg as lg
import cv2 as cv
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)


    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)

        return frame

    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera

        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt
                temp_feature[:, 4:6] = kp_l[pt_idx].pt
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) &
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) &
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor

    # new function for pose estimation calculations
    def estimated_pose(self, n_matched, curr_pos, prev_pos, N):

        # taking random sample or subset
        indices = np.random.random_integers(0, n_matched-1, N)
        p_prev = prev_pos[indices]
        p_curr = curr_pos[indices]

        # calculating centroids
        p_a = np.mean(p_prev, axis=0)
        p_b = np.mean(p_curr, axis=0)

        # defining data matrix with weights = 1
        sum_ba = 0
        for j in range(N):
            sum_ba += np.matmul((p_curr[j, :] - p_b), (p_prev[j, :] - p_a).T)
        W_ba = sum_ba/N

        # Singlular Value Decomposition
        V, _, U_transpose = lg.svd(W_ba)

        # rotation and translation matrices
        detV = lg.det(V)
        detU = lg.det(U_transpose.T)
        C_ba = np.matmul(np.matmul(V, [[1, 0,         0],
                                       [0, 1,         0],
                                       [0, 0, detU*detV]]), U_transpose)
        r_ba_a = -np.matmul(C_ba.T, p_b) + p_a

        return C_ba, r_ba_a

    def calculate_3d_points(self, b, f_u, f_v, c_u, c_v, u_l, v_l, u_r, v_r, n_matched):
        """calculate 3D points using inverse stereomodel"""

        positions_3d = []
        for j in range(n_matched):
            scalar = b / (u_l[j] - u_r[j])
            matrix = [[            0.5 * (u_l[j] + u_r[j]) - c_u],
                      [(f_u/f_v) * 0.5 * (v_l[j] + v_r[j]) - c_v],
                      [                                      f_u]]
            p = np.dot(scalar, matrix)
            positions_3d.append(p)

        return np.array(positions_3d)


    def pose_estimation(self, features_coor):
        # dummy C and r
        C = np.eye(3)
        r = np.array([0, 0, 0])
        # feature in right img (without filtering)
        # f_r_prev, f_r_cur = features_coor[:,2:4], features_coor[:,6:8]
        # ------------- start your code here -------------- #

        # initialize random generator

        # number of matched features
        n_matched = len(features_coor)

        # defining camera parameters locally
        b   = self.cam.baseline
        f_u = self.cam.fx
        f_v = self.cam.fy
        c_u = self.cam.cu
        c_v = self.cam.cv

        # extract pixel positions from previous frame
        prev_u_l = features_coor[:, 0]
        prev_v_l = features_coor[:, 1]
        prev_u_r = features_coor[:, 2]
        prev_v_r = features_coor[:, 3]

        # calculate 3D points using inverse stereo camera model for previous time step
        prev_pos = self.calculate_3d_points(b, f_u, f_v, c_u, c_v,
                            prev_u_l, prev_v_l, prev_u_r, prev_v_r,
                            n_matched)


        # extract pixel positions from current frame
        curr_u_l = features_coor[:, 4]
        curr_v_l = features_coor[:, 5]
        curr_u_r = features_coor[:, 6]
        curr_v_r = features_coor[:, 7]

        # calculate 3D points using inverse stereo camera model for current time step
        curr_pos = self.calculate_3d_points(b, f_u, f_v, c_u, c_v,
                            curr_u_l, curr_v_l, curr_u_r, curr_v_r,
                            n_matched)

        # RANSAC iterations
        prob_p = 0.9
        prob_w = 0.7
        k_iter = int(np.ceil(np.log(1 - prob_p) / np.log(1 - prob_w**3)))
        error_thresh = 0.1
        # k_iter       = 10

        # final inlier variables
        inlier_prev = []
        inlier_curr = []
        f_r_prev    = []
        f_r_cur     = []
        n_inliers   = 0

        for k in range(k_iter):
            C_ba, r_ba_a = self.estimated_pose(n_matched, curr_pos, prev_pos, 3)

            # temp inlier variables
            temp_n_inliers   = 0
            temp_inlier_prev = []
            temp_inlier_curr = []
            temp_f_r_prev    = []
            temp_f_r_cur     = []

            for j in range(n_matched):
                error = 0.5*np.matmul((curr_pos[j, :] - np.matmul(C_ba, (prev_pos[j, :] - r_ba_a))).T,
                                     (curr_pos[j, :] - np.matmul(C_ba, (prev_pos[j, :] - r_ba_a))))

                if error < error_thresh:
                    temp_n_inliers += 1
                    temp_inlier_prev.append(prev_pos[j, :])
                    temp_inlier_curr.append(curr_pos[j, :])
                    temp_f_r_prev.append(features_coor[j, 2:4])
                    temp_f_r_cur.append(features_coor[j, 6:8])

            # print('Inlier percentage for iteration ' + str(k) + ' is: ' + str(temp_n_inliers/n_matched*100))
            # new inlier set is bigger than previous
            if temp_n_inliers > n_inliers:
                n_inliers   = temp_n_inliers
                inlier_prev = temp_inlier_prev
                inlier_curr = temp_inlier_curr
                f_r_prev    = temp_f_r_prev
                f_r_cur     = temp_f_r_cur

        print('Final inlier percentage: ' + str(n_inliers/n_matched*100))
        # pose estimation
        inlier_curr = np.array(inlier_curr)
        inlier_prev = np.array(inlier_prev)
        f_r_cur = np.array(f_r_cur)
        f_r_prev = np.array(f_r_prev)
        C_ba, r_ba_a = self.estimated_pose(n_inliers, inlier_curr, inlier_prev, n_inliers)
        C = C_ba
        r = - np.matmul(C_ba, r_ba_a)

        # replace (1) the dummy C and r to the estimated C and r.
        #         (2) the original features to the filtered features
        return C, r, f_r_prev, f_r_cur

    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right

    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME

        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking

    def update(self, img_left, img_right, frame_id):

        self.new_frame_left = img_left
        self.new_frame_right = img_right

        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)

        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)

        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)

        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right

        return frame_left, frame_right

