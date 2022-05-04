#!/usr/bin/env python2

"""ROS Node for saving quadrotor's actual and desired pose"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import rospy
from geometry_msgs.msg import TransformStamped, Twist
from tf.transformations import euler_from_quaternion
import numpy as np
import pandas as pd

class SavePoseTrajectory(object):
    """ROS interface for saving quadrotor's current pose and desired trajectory"""
    
    def __init__(self):
        # Subscribers
        self.sub_vicon_data = rospy.Subscriber('/vicon/ARDroneCarre/ARDroneCarre',
                                               TransformStamped,
                                               self.get_cur_pose)

        self.sub_pos_generator = rospy.Subscriber('/trajectory_generator',
                                                  Twist,
                                                  self.get_des_pose)                                        

        # position variables
        self.x_cur   = 0.0;         self.x_des    = 0.0
        self.y_cur   = 0.0;         self.y_des    = 0.0
        self.z_cur   = 0.0;         self.z_des    = 0.0
        self.yaw_cur = 0.0;         self.yaw_des  = 0.0

        # variable storing the combined data
        self.comb_data = []

        # write the data to disk when node is shutdown
        rospy.on_shutdown(self.write_data())

        # saving the data at below frequency
        self.save_loop_frequency = 20
        rospy.Timer(rospy.Duration(1/self.save_loop_frequency), self.save_data)

    def get_cur_pose(self, data):
        """fetches the current pose of the quadrotor"""

        self.x_cur = data.transform.translation.x
        self.y_cur = data.transform.translation.y
        self.z_cur = data.transform.translation.z

        quaternion = np.array([data.transform.rotation.x,
                               data.transform.rotation.y,
                               data.transform.rotation.z,
                               data.transform.rotation.w])
        
        euler        = euler_from_quaternion(quaternion)
        self.yaw_cur = euler[2]


    def get_des_pose(self, data):
        """fetches the desired pose of the quadrotor"""

        self.x_des   = data.linear.x
        self.y_des   = data.linear.y
        self.z_des   = data.linear.z
        self.yaw_des = data.angular.z

    
    def save_data(self, event=None):
        """saves the combined pose and trajectory data in one variable"""
        
        self.comb_data.append([self.x_cur, self.y_cur, self.z_cur, self.yaw_cur, \
            self.x_des, self.y_des, self.z_des, self.yaw_des])


    def write_data(self):
        """writes the combined data to disk"""

        df = pd.DataFrame (self.comb_data, columns = ['x_cur, y_cur, z_cur, yaw_cur', \
                                                        'x_des', 'y_des', 'z_des', 'yaw_des'])
        df.to_csv('simulation_data.csv', index=False)


if __name__ == '__main__':
    rospy.init_node('save_data')
    pose_traj_data = SavePoseTrajectory()
    rospy.spin()
    