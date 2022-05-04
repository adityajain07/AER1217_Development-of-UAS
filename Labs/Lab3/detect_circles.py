# %% [markdown]
# About          : AER1217 - Circle Detection <br>
# Date Started   : March 17, 2022 <br>
# 

# %%
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import json

# %% [markdown]
# Variable definitions

# %%
data_dir   = '/Users/adityajain/Downloads/lab3data_images_pose/'
save_dir   = '/Users/adityajain/Downloads/lab3data_detected_circles/'
pose_file  = data_dir + 'pose_info.json'
data_pose  = json.load(open(pose_file))
image_list = os.listdir(data_dir)

# storing pose and detection information
data_detection = {}

# %% [markdown]
# Hough circle transform for finding camera centres

# %%
for img_name in image_list:
    if img_name.endswith('.jpg'):
        img       = cv2.imread(data_dir + img_name)
        img_green = img[:, :, 1]        # extracting the green channel    
        img_green = cv2.medianBlur(img_green,5)

        circles = cv2.HoughCircles(img_green,cv2.HOUGH_GRADIENT,1,minDist=100,
                            param1=50,param2=30,minRadius=25,maxRadius=40)

        # getting pose data for this frame
        img_name_json, _ = os.path.splitext(img_name)
        pose_data        = data_pose[img_name_json]

        if type(circles) is np.ndarray:            
            # combined pose and detection data
            data_detection[img_name_json]                  = {}
            data_detection[img_name_json]['pose']          = {}
            data_detection[img_name_json]['pose']['x']     = pose_data['x']
            data_detection[img_name_json]['pose']['y']     = pose_data['y']
            data_detection[img_name_json]['pose']['z']     = pose_data['z']
            data_detection[img_name_json]['pose']['roll']  = pose_data['roll']
            data_detection[img_name_json]['pose']['pitch'] = pose_data['pitch']
            data_detection[img_name_json]['pose']['yaw']   = pose_data['yaw']
            data_detection[img_name_json]['circles']       = {}
            
            circles = np.uint16(np.around(circles))
            count = 0
            for i in circles[0,:]:
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),2)
                data_detection[img_name_json]['circles'][count] = [int(i[0]), int(i[1]), int(i[2])]   # (x, y, radius)
                count += 1

            cv2.imwrite(save_dir + img_name, img)

with open(save_dir + 'detection_info.json', 'w') as outfile:
            json.dump(data_detection, outfile)

# %% [markdown]
# Plotting drone positions for detected circles

# %%
data_detection  = json.load(open(save_dir + 'detection_info.json'))

x_pos = []
y_pos = []

for frame in data_detection.keys():
    x_pos.append(data_detection[frame]['pose']['x'])
    y_pos.append(data_detection[frame]['pose']['y'])

plt.scatter(x_pos, y_pos, s=2, alpha=0.5)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Drone positions of circle detection')
plt.savefig('capturing_position.png')
plt.show()



# %%



