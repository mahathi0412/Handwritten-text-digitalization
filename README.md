Hand-written text recognition is an algorithm that has been designed to detect handwritten characters of any sort quickly when input is supplied by image means, and an image to text converter converts scanned text into digital editable text.
Steps involed:
1) Data collection - Data for training the model is taken from friends of all the alphabets in both upper and lower cases in cursive style. The collected images are taken in “.jpg” format. Also used datasets from IAM and MNIST.
2) Data pre-processing - Augmented the data in different ways like shearing, skewing, rotating etc., to increase the volume of the data and resized the images to a desired size, here 100x100. Few letters are not rotated such as “a”, “c”, “e”, “f”, “h”, “m”, “w”, “p”, “q”.
3) Data Filtering - Got rid of noise such as outlier values and error data from raw data to make the data clean and be proper for further processing. Used Canny Edge Detection for this step.
4) Data segmentation - Used findContours function to retrieve all the contours in the image that it can find.
5) CNN model - Built a model to recognize the letter.

The model takes in the image of handwritten text, breaks the word to individual letters, passes it to the CNN model and then the letter matched gets digitized and is stored in a notepad.
