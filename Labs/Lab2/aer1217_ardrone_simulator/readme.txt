# AER1217 Winter 2022, Lab 2
# Team E: Aditya Jain, Apurv Mishra, Praharsha Abbireddy

=============================
Runnning the code 
=============================
Run 'roslaunch aer1217_ardrone_simulator ardrone_simulator.launch' in terminal. This will launch all the required nodes.


=============================
Files Information 
=============================
1. position_controller.py - calculates the commanded control using the vicon feedback, quadrotor dynamics and desired trajectory
                        
2. ros_interface.py - subscribes to vicon feedback and desired trajectory topic, calls the position_controller and publishes the required control to /cmd_vel_RHC

3. indoor_robotics_lab_interface.py - subscribes to /cmd_vel_RHC, calculates required motor speeds and publishes it

4. desired_positions.py - generates the trajectory waypoints and publishes it to /trajectory_generator

5. record_pose_trajectory.py - saves the current and desired pose of the quadrotor during the simulation for plotting

6. bag_to_plot.py - reads relevant topics from the bag file and plots the data

6. ardrone_simulator.launch - launches all the required nodes for the simulation to occur