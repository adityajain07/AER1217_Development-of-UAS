# AER1217 Winter 2022, Lab 3
# Team E: Aditya Jain, Apurv Mishra, Praharsha Abbireddy

=============================
Runnning the code 
=============================
Run 'roslaunch aer1217_ardrone_vicon ardrone_vicon.launch' in terminal. This will launch all the required nodes.


=============================
Files Information 
=============================
1. position_controller.py - calculates the commanded control using the vicon feedback, quadrotor dynamics and desired trajectory
                        
2. ros_interface.py - subscribes to vicon feedback and desired trajectory topic, calls the position_controller and publishes the required control to /cmd_vel_RHC

3. desired_positions.py - generates the trajectory waypoints for the lawn-mover pattern and publishes it to /trajectory_generator

4. bag_to_data.py - reads relevant topics from the bag file, and saves the image and corresponding pose data to a JSON file

5. detect_circles.py - uses Hough transform to detect circles and saves the data to the JSON file, corresponding to each frame

6. georeferencing.py - reads the JSON file with all the image and pose data, uses this data and provided camera matrices to find the centre of circle markers in the inertial frame 

7. ardrone_vicon.launch - launches all the required nodes for the simulation to occur