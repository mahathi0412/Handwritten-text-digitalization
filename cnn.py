import cv2 as cv 
import numpy as np 
import pandas as pd 
from keras.models import load_model 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
#Reading Dataset and Analyzing it 
data = pd.read_csv('archive/A_Z Handwritten Data.csv')
#Splitting x and y column from dataset 
X = my_data[:,1:] 
y = my_data[:,:1] 
print(X.shape) 
print(y.shape) 
(372450, 784) 
(372450, 1) 
#Split into train and validation set 
#X_train.shape (297960, 784) 
#(X_test.shape 74490, 784) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) 
#reshaping to 28*28 pixels from 784 
X_train = np.reshape(X_train,(X_train.shape[0],28,28)) 
X_test = np.reshape(X_test,(X_test.shape[0],28,28)) 
print(X_train.shape) 
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape) 
#Plotting number of images for each alphabet from Dataset 
count = np.zeros(26, dtype = 'int') #count list containing all zeroes 
for i in y: 
count[i] += 1 
#creating a list of alphabets 
alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
plt.figure(figsize=(15,10)) 
plt.barh(alphabets, count, color = "cyan") 
plt.xlabel("Number of Alphabets",fontsize = 20, fontweight = 'bold',color = 'green') 
plt.ylabel("Alphabets",fontsize = 30, fontweight = 'bold',color = 'green') 
plt.title("No. of images available for each alphabet in the dataset", fontsize = 20, fontweight = 'bold', color = "red") 
plt.grid() 
plt.show()
#Show random images 
img_list = shuffle(X_train[:1000]) 
fig,ax = plt.subplots(3,3,figsize=(15,15)) 
axes = ax.flatten() 
for i in range(9): 
axes[i].imshow(img_list[i]) 
axes[i].grid() 
plt.show() 
#Reshaping train & test images from dataset to put in the CNN Model 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1) 
print("New shape of train and test dataset") 
print(X_train.shape) 
print(X_test.shape) 
#Downsampling the images to make them in the range of (0-1) 
X_train = X_train/255. 
X_test = X_test/255. 
#Convert the int values of labels to categorical values of 26 
categorical_ytrain = to_categorical(y_train, num_classes = 26, dtype = 'int') 
print("New shape of train labels:", categorical_ytrain.shape) 
categorical_ytest = to_categorical(y_test, num_classes = 26, dtype = 'int') 
print("New shape of test labels:", categorical_ytest.shape) 
#CNN Model Architecture 
model = Sequential() 
#First Conv1D layer 
model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',input_shape = (28,28,1))) 
model.add(MaxPooling2D(pool_size = (2,2),strides = 2)) 
#Second Conv1D layer 
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation = 'relu', padding = 'same')) 
model.add(MaxPooling2D(pool_size = (2,2), strides = 2)) 
#Third Conv1D layer 
model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu', padding = 'valid')) 
model.add(MaxPooling2D(pool_size = (2,2), strides = 2)) 
#Flatten layer 
model.add(Flatten()) 
#Dense layer 1 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.2)) 
#Dense layer 2 
model.add(Dense(64,activation = 'relu')) 
#Final layer of 26 nodes 
model.add(Dense(26,activation = 'softmax')) 
#Define the loss function to be categorical cross-entropy since it is a multi-classification problem: 
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) 
#Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and to save the best model after every epoch: 
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.001) 
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
#Training the model and evaluate the performance on the holdout set: 
history = model.fit(x = X_train, y = categorical_ytrain, epochs = 100, callbacks=[es,mc], validation_data = (X_test,categorical_ytest)) 
#evaluating model on test dataset 
model.evaluate(X_test,categorical_ytest) 
model.summary()
#Find accuracy, losses of Model 
print("The validation accuracy is :", history.history['val_accuracy'][-1]) 
print("The training accuracy is :", history.history['accuracy'][-1]) 
print("The validation loss is :", history.history['val_loss'][-1]) 
print("The training loss is :", history.history['loss'][-1]) 
#Plotting the Model loss and Accuracy on the line graph 
plt.figure(figsize = (6,6)) 
plt.plot(history.history['loss'], label='train') 
plt.plot(history.history['val_loss'], label='test') 
plt.legend() 
plt.title("Model Loss") 
plt.show() 
plt.figure(figsize = (6,6)) 
plt.plot(history.history['accuracy'], label='train') 
plt.plot(history.history['val_accuracy'], label='test') 
plt.legend() 
plt.title("Model Accuracy") 
plt.show() 
#Making prediction of test data 
dict_word = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'} 
img_list_test = shuffle(X_test[:1000]) 
fig, axes = plt.subplots(3, 3, figsize = (12, 15)) 
axes = axes.flatten() 
for i in range(9): 
img = np.reshape(X_test[i], (28, 28)) 
axes[i].imshow(img_list_test[i]) 
pred = dict_word[np.argmax(model.predict(np.reshape(img_list_test[i],(1,28,28,1))))] 
axes[i].set_title("Prediction: " + pred, fontsize = 20, fontweight = 'bold', color = 'red') 
axes[i].grid()
