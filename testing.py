#Importing the Libraries 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import os 
from keras.models import load_model 
#Loading the Model 
model = load_model('best_model.h5') 
#Creating a dictionary of letters with Index 
dict_word = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'} 
#Reading the images from the Specified Path 
images=[cv.imread(file) for file in glob.glob("C:\\Users\\mahes\\Downloads\\new_1\\Handwritten-Character-Recognition-main\\Images\\*.png")] 
images=[cv.imread(file) for file in glob.glob("D:\\Documents\\*.png")] 
#Plotting the Images read from the file 
fig, axes = plt.subplots(7, 4, figsize = (30,30)) 
axes = axes.flatten() 
for i in range(len(images)): 
axes[i].imshow(images[i]) 
plt.delaxes(ax = axes[26]) 
plt.delaxes(ax = axes[27]) 
#Image Processing, Plotting and Predicting the images 
fig, axes = plt.subplots(7, 4, figsize = (30, 30)) 
axes = axes.flatten() 
f=open(r'D:\\documents\\text.txt','w')
for i in range(len(images)): 
gray = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY) 
gray = cv.medianBlur(gray,5) 
ret,gray = cv.threshold(gray,75,180,cv.THRESH_BINARY) 
element = cv.getStructuringElement(cv.MORPH_RECT,(90,90)) 
gray = cv.morphologyEx(gray,cv.MORPH_GRADIENT,element) 
gray = gray/255. #downsampling 
#gray = 1 - gray 
gray = cv.resize(gray, (28,28)) #resizing 
#reshaping the image 
gray = np.reshape(gray, (28, 28)) 
axes[i].imshow(gray) 
pred = dict_word[np.argmax(model.predict(np.reshape(gray,(1,28,28,1))))] 
axes[i].set_title("Prediction: " + pred, fontsize = 30, fontweight = 'bold', color = 'green') 
plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4) 
f.write(pred) 
f.close()
