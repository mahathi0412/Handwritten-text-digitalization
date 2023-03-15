import cv2
import p
import numpy as np
import argparse
import glob
import os
from matplotlib import pyplot as plt
#d=r'C:\Users\Mahathi Mandelli\Desktop\final project\images_folder'
path = "C:\\Users\\Mahathi Mandelli\\Desktop\\final project\\final_data\\data_subset\\data_subset"
for j in os.listdir(path):
	if j.endswith(".jpg") or j.endswith(".jpeg") or j.endswith(".png"):
		#f=j
	#image = cv2.imread("BS1.jpeg")#d=r'C:\Users\mahes\Desktop\Demo'
		#print(f)
		image=cv2.imread(j)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
		wide = cv2.Canny(blurred, 10, 200)
		tight = cv2.Canny(blurred, 225, 250)
		auto = p.auto_canny(blurred)
		print(auto)
#hss=skimage.feature.hessian_matrix(blurred, sigma=1, mode='constant', cval=0, order='rc')
		hss=p.myHarris(auto)
# show the images

		#os.chdir(d)
		cv2.imwrite(j,np.hstack([wide]))
		#cv2.imshow("Edge",np.hasstack)

	