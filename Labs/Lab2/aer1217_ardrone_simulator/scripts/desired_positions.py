#!/usr/bin/env python2

"""ROS Node for publishing desired positions."""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np
from geometry_msgs.msg import Twist

# Import class that computes the desired positions
# from aer1217_ardrone_simulator import PositionGenerator


class ROSDesiredPositionGenerator(object):
    """ROS interface for publishing desired positions."""

    def __init__(self):
        # Publisher
        self.pub_traj = rospy.Publisher('/trajectory_generator', Twist, queue_size=500)

        # defining index for array of trajectory points
        self.idx = 0

        # index to reset the trajectory
        self.reset_idx = 0

        self.total_points  = 0

        # defining message type for the publisher
        self.pub_traj_msg = Twist()

        # initialize waypoints with start position and define trajectory - LINEAR
        self.target_pos1   = [-1.5, -1.5, 1.0]
        self.target_pos2   = [1.5, 1.5, 2.0]
        self.target_pos    = self.linear_trajectory()

        # initialize waypoints with start position and define trajectory - CIRCULAR
        self.start_pos    = [1.5, 0.0, 0.5]
        self.del_z        = 1
        self.radius       = 1.5
        # self.target_pos   = self.circle_trajectory()

        # Publish the control command at the below frequency
        self.desired_pos_frequency = 10
        rospy.Timer(rospy.Duration(1 / self.desired_pos_frequency), self.publish_trajectory)

    def publish_trajectory(self, event=None):
        self.pub_traj_msg.linear.x = self.target_pos[0, self.idx]
        self.pub_traj_msg.linear.y = self.target_pos[1, self.idx]
        self.pub_traj_msg.linear.z = self.target_pos[2, self.idx]
        self.pub_traj_msg.angular.z = self.target_pos[3, self.idx]

        self.idx += 1

        if self.idx == self.total_points:
            self.idx = self.reset_idx

        self.pub_traj.publish(self.pub_traj_msg)

    def linear_trajectory(self):
        # convert to an array, taking each position
        traj_points = np.asarray([[0, 0, 0], self.target_pos1, self.target_pos2, self.target_pos1])

        points_x = np.asarray([])
        points_y = np.asarray([])
        points_z = np.asarray([])
        angles_z = np.asarray([])
        for k in range(np.shape(traj_points)[0] - 1):

            # calculate total waypoints
            distance  = np.abs(traj_points[k, :] - traj_points[k+1, :])
            max_dist  = np.max(distance)
            total_pts = int(max_dist/0.02)

            # to repeat the trajectory
            if k == 1:
                self.reset_idx = self.total_points
            self.total_points += total_pts + 100

            # define trajectory
            x = np.linspace(traj_points[k, 0], traj_points[k+1, 0], total_pts)
            y = np.linspace(traj_points[k, 1], traj_points[k+1, 1], total_pts)
            z = np.linspace(traj_points[k, 2], traj_points[k+1, 2], total_pts)

            # concatenate horizontally
            points_x = np.hstack((points_x, x, [traj_points[k+1, 0] for a in range(100)]))
            points_y = np.hstack((points_y, y, [traj_points[k+1, 1] for b in range(100)]))
            points_z = np.hstack((points_z, z, [traj_points[k+1, 2] for c in range(100)]))
            angles_z = np.hstack((angles_z, 0.0 * z,           [0.0 for c in range(100)]))

        # update trajectory and return
        self.target_pos = np.asarray([points_x, points_y, points_z, angles_z])
        return self.target_pos

    def circle_trajectory(self):
        points_x = np.asarray([])
        points_y = np.asarray([])
        points_z = np.asarray([])
        angles_z = np.asarray([])

        # calculate total trajectory to the initial position
        distance = np.abs(self.start_pos)
        max_dist = np.max(distance)
        total_pts = int(max_dist/0.02)

        # define trajectory
        pos_x = np.linspace(0, self.start_pos[0], total_pts)
        pos_y = np.linspace(0, self.start_pos[1], total_pts)
        pos_z = np.linspace(0, self.start_pos[2], total_pts)
        yaw   = np.linspace(0,             np.pi, total_pts)

        # concatenate horizontally
        points_x = np.hstack((points_x, pos_x, [self.start_pos[0] for a in range(100)]))
        points_y = np.hstack((points_y, pos_y, [self.start_pos[1] for b in range(100)]))
        points_z = np.hstack((points_z, pos_z, [self.start_pos[2] for c in range(100)]))
        angles_z = np.hstack((angles_z,   yaw, [            np.pi for c in range(100)]))

        # to repeat the trajectory
        self.reset_idx    = total_pts + 100
        self.total_points = total_pts + 100

        # the circular part
        z = 0
        z_initial     = self.start_pos[2]
        z_inc         = self.del_z/36
        for angle in range(0, 360, 5):
            if angle == 180:
                z_inc     = -1 * z_inc
                z_initial += self.del_z
                z = 0
            rad = angle * np.pi / 180
            waypoints = [self.radius * np.cos(rad), self.radius * np.sin(rad), z_initial + z * z_inc]
            points_x  = np.hstack((points_x, [   waypoints[0] for a in range(10)]))
            points_y  = np.hstack((points_y, [   waypoints[1] for b in range(10)]))
            points_z  = np.hstack((points_z, [   waypoints[2] for c in range(10)]))
            angles_z  = np.hstack((angles_z, [(np.pi+rad-0.1) for c in range(10)]))
            self.total_points += 10
            z += 1

        # update trajectory and return
        self.target_pos = np.asarray([points_x, points_y, points_z, angles_z])
        return self.target_pos


if __name__ == '__main__':
    rospy.init_node('desired_position')
    ROSDesiredPositionGenerator()
    rospy.spin()
