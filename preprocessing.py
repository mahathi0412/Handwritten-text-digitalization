# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy import signal
import os

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def myHarris(image):
    '''
    Compute Harris operator using hessian matrix of the image 
    input : image 
    output : Harris operator. '''
    # x derivative
    sobelx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    #y derivative
    sobely = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])
    # Get Ixx
    # To get second derivative differentiate twice.
    Ixx = signal.convolve2d(signal.convolve2d(signal.convolve2d(image, sobelx, "same"),sobelx,"same"), sobelx, "same")
    # Iyy  
    Iyy = signal.convolve2d(signal.convolve2d(image, sobely, "same"),sobely,"same")
    # Ixy Image 
    Ixy = signal.convolve2d(signal.convolve2d(image, sobelx, "same"),sobely,"same")
    # Get Determinant and trace
    det = Ixx*Iyy - Ixy**2
    trace = Ixx + Iyy
    # Harris is det(H) - a * trace(H) let a = 0.2
    H = det - 0.2 * trace
    # Lets show them 
    '''
	plt.figure("Original Image")
    plt.imshow(image)
    plt.figure("Ixx")
    plt.imshow(Ixx)
    plt.set_cmap("gray")
    plt.figure("Iyy")
    plt.imshow(Iyy)
    plt.figure("Ixy")
    plt.imshow(Ixy)
    plt.figure("Harris Operator")
    plt.imshow(np.abs(H))
    plt.show()
	'''
    return  H
