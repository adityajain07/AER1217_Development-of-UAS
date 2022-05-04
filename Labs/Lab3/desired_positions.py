#!/usr/bin/env python2

"""ROS Node for publishing desired positions."""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import math
import matplotlib.pyplot as plt

import roslib
import rospy
import numpy as np
import os
from geometry_msgs.msg import TransformStamped, Twist, Vector3

# Import class that computes the desired positions
# from aer1217_ardrone_simulator import PositionGenerator


class ROSDesiredPositionGenerator(object):
    """ROS interface for publishing desired positions."""

    def __init__(self):
        # Publisher
        self.pub_traj = rospy.Publisher('/trajectory_generator', Twist, queue_size=500)

        # Subscriber
        self.sub_vicon_data = rospy.Subscriber('/vicon/ARDroneCarre/ARDroneCarre',
                                               TransformStamped,
                                               self.current_position)

        # Drone's current position
        self.x_cur = 0.0
        self.y_cur = 0.0
        self.z_cur = 0.0

        # defining index for array of trajectory points
        self.idx = 0

        # initializations
        self.height = 1.75
        self.overlap = 0.5
        self.d_fov = 64 * np.pi / 180
        self.error_range = 0.05
        self.image_path = os.path.dirname(os.path.abspath(__file__)) + '/'

        # field of view rectangle calculations
        self.diagonal = 2 * np.tan(self.d_fov/2) * self.height
        ratio_const = np.sqrt((self.diagonal**2)/(16**2 + 9**2))
        self.l_fov = 9 * ratio_const
        self.w_fov = 16 * ratio_const

        # re-calculating with overlap
        self.l_fov = self.overlap * self.l_fov
        self.w_fov = self.overlap * self.w_fov

        # defining message type for the publisher
        self.pub_traj_msg = Twist()

        # define corners and calculate trajectory
        self.x_min = -2
        self.x_max = 2
        self.y_min = -2
        self.y_max = 2
        self.target_pos   = self.traj_calculation()

        # Publish the control command at the below frequency
        self.desired_pos_frequency = 10
        rospy.Timer(rospy.Duration(1 / self.desired_pos_frequency), self.publish_trajectory)

    def current_position(self, vicon_data):
        self.x_cur = vicon_data.transform.translation.x
        self.y_cur = vicon_data.transform.translation.y
        self.z_cur = vicon_data.transform.translation.z

    def publish_trajectory(self, event=None):
        x_diff = self.x_cur - self.target_pos[0, self.idx]
        y_diff = self.y_cur - self.target_pos[1, self.idx]
        z_diff = self.z_cur - self.target_pos[2, self.idx]

        error = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        if error < self.error_range:
            self.idx += 1

        self.pub_traj_msg.linear.x = self.target_pos[0, self.idx]
        self.pub_traj_msg.linear.y = self.target_pos[1, self.idx]
        self.pub_traj_msg.linear.z = self.target_pos[2, self.idx]
        self.pub_traj_msg.angular.z = self.target_pos[3, self.idx]

        self.pub_traj.publish(self.pub_traj_msg)

    def traj_calculation(self):
        points_x = np.asarray([])
        points_y = np.asarray([])
        points_z = np.asarray([])
        angles_z = np.asarray([])

        # initial position
        init_x = self.x_min + self.l_fov/2
        init_y = self.y_min + self.w_fov/2

        # steps for land-mover path
        len_x = self.x_max - self.x_min
        len_y = self.y_max - self.y_min

        # land-mover follow
        temp = 1
        pos_x = init_x
        pos_y = init_y
        while self.y_max > pos_y > self.y_min:
            while self.x_max > pos_x > self.x_min:
                points_x = np.hstack((points_x, pos_x))
                points_y = np.hstack((points_y, pos_y))
                points_z = np.hstack((points_z, self.height))
                angles_z = np.hstack((angles_z, 0.0))
                pos_x = pos_x + temp * self.l_fov
            pos_y = pos_y + self.w_fov
            pos_x = pos_x - temp * self.l_fov
            temp = temp * -1

        # back to origin
        points_x = np.hstack((points_x, 0.0))
        points_y = np.hstack((points_y, 0.0))
        points_z = np.hstack((points_z, 0.1))
        angles_z = np.hstack((angles_z, 0.0))

        # update trajectory and return
        self.target_pos = np.asarray([points_x, points_y, points_z, angles_z])

        # saving the plot
        plt.plot([self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
                 [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min])
        plt.plot(points_x, points_y, 'o-')
        plt.savefig(self.image_path + 'lawn_mover_traj.png', bbox_inches='tight')

        return self.target_pos


if __name__ == '__main__':
    rospy.init_node('desired_position')
    ROSDesiredPositionGenerator()
    rospy.spin()