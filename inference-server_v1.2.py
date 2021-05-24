'''
Inference Server
start:
#python inference-server.py [front_end_server]

! pip install opencv-python
! apt update && apt install -y libsm6 libxext6  libxrender-dev

'''

###########################################
#
# Imports section
#
###########################################
import sys
import PIL
import os
import tensorflow as tf; print(tf.__version__)
import numpy as np
import json
import cv2
import picamera
import requests
import time
import datetime

from io import BytesIO
from IPython.display import display, Image
from keras import backend as K
from keras.models import Model, load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from matplotlib import pyplot as plt

###########################################
#
# Arguments check
# 
# The inference process is called from "start-park.sh" that will pass, via parameters:
#
# 1.- The URL to the server (front_end_server) that is managing the photo upload
# 2.- The path to the file containing the pretrained model. In this case the used model is a pretrained InceptionV3 over ImageNet 
# 	  <<FIXME pretrained how and how the classifier has been defined, just Inception or replacing last FC layers? >>
#
###########################################
if (len(sys.argv)==3):
    print ("inference-server.py - Front-end server IP:" + str(sys.argv[1]))
    FRONT_END = str(sys.argv[1])
    print ("inference-server.py - Pretrained inference model:" + str(sys.argv[2]))
    PRETRAIN_MODEL = str(sys.argv[2])
else: 
    print ("inference-server.py - Missing parameters invoking the inference server: inference-server.py $front-end-server-url $pretrained-model-file-path")
    sys.exit(0)

###########################################
#
# Set the number of clases and the labels map for the classifier; it will use 2 labels:
#
# - "empty", associated to a 1 value, if the parking place is free
# - "occupied", associated to a 0 value, if the parking place is occupied by another car
#
###########################################
nb_classes  = 2
label_map = {'occupied': 0, 'empty': 1}

# Dimensions for the captured camera images 
img_width, img_height = 299, 299

###########################################
#
# Cropping the areas in the parking lot, corresponding to the positions for each one of the 6 available places
# This will be used later to locate each one of the places in the image that will be taken for the camera so to be able
# of making the inference empty/occupied over each one of the places by sending the "sub-image" in the crop area to the inference model.
#
###########################################
crop_area = [[171,44,439,187],
		     [171,220,439,422],
		     [171,449,439,618],
		     [621,44 ,898,187],
		     [621,220,898,422],
		     [621,449,898,618]]

###########################################
#
# Confidence threshold to be applied to the inference result. This threshold will be used 
# to determine if the inference when indicating "occupied" has been successful enough to accept it. 
# In other case, the classification with "empty" will have the priority.
#
###########################################
CAR_THRESHOLD = 84

###########################################
#
# Pretrained model loading. 
# The file containing the pretrained model must be stored in the same directory in which the inference server
# is executing and at the same level.
#
###########################################
if os.path.isfile(PRETRAIN_MODEL):
    model = load_model(PRETRAIN_MODEL)
    print("inference-server.py - Pretrained model loaded")

# TO DELETE!!! -- The inference server does not do any training, this should be done as a part of a modified
# InceptionV3 architecture for the classifier if it is decided to apply transfer learning.
# ------------------------------------------------------------------------------------------------------------
###########################################
#
# Fitting the pretraining model 
# 
# is executing and at the same level.
#
###########################################
#train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        #shear_range=0.2,
#        #zoom_range=0.2,
#        horizontal_flip=True
#)

# The test dataset must be just rescaled to 0-1 to fit the same scale than the images used for training
# Data augmentation is never done over a test dataset.
# test_datagen = ImageDataGenerator(rescale=1./255)

# print ("\nTraining directory:")
# train_generator = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=64,
#    class_mode='categorical'
#)

#{'empty': 0, 'occupied': 1}
# label_map = (train_generator.class_indices)
# ------------------------------------------------------------------------------------------------------------

###########################################
#
# Method inference
#
# Input: PIL Image to classify
# Output: Inference probabilities
#
# Receives an image that corresponds to an individual parking place, sends it to the model to be predicted
# as free or occupied, takes the predictions for both classes and returns a JSON containing the predictions
# with the next format:
#
# { "name":"Predictions",  "results": [ { "class1":"occupied", "value":"93" },{ "class2":"empty", "value":"6" }]  }
#
###########################################
def inference(img):

	# Preprocessing the image
	#
	# The model has been trained using InceptionV3 that uses the preprocess function on all the images when learning the model. 
	# Therefore the images to infer have to run through the same preprocess function before doing anything else to be 
	# aligned with the images that have been used to calculate the parameters of the neural network.
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)	# Dimension expansion with axis=0 equivalent to x[np.newaxis,:] 
    x = preprocess_input(x)
    
	# Make the predictions using the preprocessed image. This will return a unidimensional vector with the inferred probabilities 
	# for each label with a format like [[0.06030592 0.9396941 ]]
    preds = model.predict(x)
    
    # Reversing the elements in the map with the labels, passing from {'empty': 0, 'occupied': 1} to {0: 'empty', 1: 'occupied'}
	# making the boolean value to be the key of the label
    inv_map = {v: k for k, v in label_map.items()}
	
	# Creates an array of pairs with values [label, % inferred probability] where the probability value for the inference 
	# comes from the predictions made. The loop to create the array must be over preds[0] for preds is an array of arrays [[0.06030592 0.9396941 ]] 
	# and it is needed to loop over the probabilities that is the component at position 0
    i=0
    a = []
    for prob in preds[0]:
        a.append([i,int(prob*100)])
        i+=1

	# Sorting the array with the predictions results usgin the axis=0, so direction down.
    sorted = np.argsort(a, axis=0) 
    
	# 
	final = []
    for elem in sorted:
		element = [inv_map[elem[1]],int(preds[0][elem[1]]*100)]
        final.append(element)
		print("inference-server.py - inference - Appending " + element)
		
	# 
    inference = np.flip(final, axis=0)
    
	# 
	partJson = '{ \"name\":\"' + "Predictions" + '\",  \"results\": [ '
    for i in range(0,nb_classes):
        partJson += '{ \"class' + str(i+1) + '\":\"' + str(inference[i][0]) + '\", \"value\":\"' + str(inference[i][1]) + '\" },' 
    json = partJson[:-1]+ ']  }'
    
	return json

###########################################
#
# Method display_img_array
#
# Input: OpenCV image
# Output: PIL Image object
#
# This method takes an image that has been created using OpenCV (and therefore is a ndarray), transforms it
# to a PIL image and, before returning it, sets the correct order of the colors reversing the final array
# for it to pass from ndarray BGR to PIL RGB order being in that way returned wiht its true colors.
#
# OpenCV ndarray (BGR):
#
# 	array([[[136, 121, 118],
#        	[135, 123, 119],
#        	[136, 124, 120],
#        	...]
#
# PIL image (RGB):
#
# 	array([[[118, 121, 136],
#   	    [119, 123, 135],
#       	[120, 124, 136],
#        	...]
#
###########################################
def display_img_array(ima):

	# Coverting the OpenCV ndarray image in a PIL image
    img = PIL.Image.fromarray(ima)
    
	# Check the size of the image and rotates it 90 degrees in case of the width smaller than the height 
	# , that way always ensuring we deal with a horizontal orientated image. The rotated image will be expanded 
	# to hold all the original image by using the flag expand = True
    print("inference-server.py - display_img_array - ( width: " + str(img.size[0])+", height: "+str(img.size[1])+")")
    if (img.size[0] < img.size[1]):
        img = img.rotate(90, expand = True)
    
	# TO DELETE!! -- Not needed for it is not further used
	# --------------------------------------------------------------------------------------------------------------
	# Creates a bytes object and use it to save the image object in a .png formatted image
    # bio = BytesIO()
    # img.save(bio, format='png')    
    # display(Image(bio.getvalue(), format='png', retina=True))
	# --------------------------------------------------------------------------------------------------------------	
    
	# Creates a copy of the image array by converting it to a numPy array type, then making a reverse copy
	# by using the notation for extended slicing:
	#
	# - [:,:,::-1] -> Reverse the array [::-1] to which it is applied in all the array dimensions [:,:] (all rows & all columns)
	#
	# Reversing the array is necessary because the ndarray image that is coming as a parameter to this method has been created using OpenCV
	# OpenCV assumes the order for the colors is BGR but PIL assumes the order is RGB, so when converting from the OpenCV ndarray image to the
	# PIL image by using the function "fromarray()", the PIL image is created with the wrong order in the colors so it will be displayed weirdly. Therefore
	# the image needs to be transformed to have the correct RGB order that is expected for PIL and for that, it is needed to reverse the array elements 
	# making them pass from BGR to RGB and therefore, setting them in the correct order for the PIL Image to be properly displayed.
	#
	# A copy is needed to avoid that changes made in the original array to affect the returned one for copies of numpy arrays, 
	# unless copy() method is used, are images of the original array so any changes to it are automatically applied
	# to their images.
	#
    imcv = np.asarray(img)[:,:,::-1].copy()
	
    return imcv
	
###########################################
#
# Method scan
#
# Input: PIL Image to classify
# Output: A vector with 6 binary elements 0-1, one for each parking place, indicating 0 - free / 1 - occupied
#
# This method will receive an image of the parking that has been captured from the raspberry camera and will infer,
# ,for each parking place within the picture, if the parking place is free or occupied, by returning finally a vector
# filled with binary values in which a 0 value will indicate the parking place is free and a 1 value will indicate
# the parking place is occupied
#
########################################### 
def scan(img):

	# Init local variables
	occupied = [0,0,0,0,0,0]
    i=0
	
	# For each parking place in the globally defined parking places
    for box in crop_area:
    
		# Croping the image portion that fits within the dimensions of the pre-defined parking place
		# This will create an object with array interface that will allow it to be converted to an image object
        crop_img = img[box[1]:box[3],box[0]:box[2]]
        #plt.imshow(crop_img)
        
		# Convert the cropped object to an image array
		crop_img = display_img_array(crop_img)
        
		# Inference the result
        result = inference(crop_img)
		#print("inference-server.py - scan - " + result)		
        
		# Gets into the status variable the class that has the highest probability. The JSON has been created to get 
		# in its first element the class that has the highest probability so it is enough to get the first element.
		jsonResult = json.loads(result)
        status = jsonResult['results'][0]['class1']
        #print(jsonResult['results'][0]['value'])
        
		# If either the place is empty or the confidence in the prediction is under the defined threshold, 
		# a 0 value will be set in the occupation vector. Otherwise, a 1 value will be set. 
		if status=="empty":
            occupied[i] = 0
            print("inference-server.py - scan - status: empty")
        else:
            if (int(jsonResult['results'][0]['value']) < CAR_THRESHOLD):
                occupied[i] = 0
				print("inference-server.py - scan - status: empty")
            else:
                occupied[i] = 1
				print("inference-server.py - scan - status: occupied")
        
		# Increment the index for the occupation vector and pass to inference the next parking place 
		i=i+1
		
    return (occupied)    

###########################################
#
# Method boxing
#
# Input: PIL Image with a capture from the parking
# Input: 6 positions vector filled with either 0s (place's free) and 1s (place's occupied) as per the inference outcome
# Output: The same PIL image received with 6 rectangles added in green or red marking the places availability
#
# Method that receive an image along with its parking places infered occupation and draws a rectangle around the place
# to mark each parking place either in green, meaning this the parking place is free, or red, meaning this
# the parking place is occupied.
#
########################################### 
def boxing(img,occupied):
    
	# Copy of the image for it is an input argument so read-only
	# and it is wanted to modify it by adding rectangles to it
	img_status = img
    i = 0
	
	# For each rectangle that has been previsouly set in "crop_area" and it's defining each parking place
    for box in crop_area:
    
		# If the place: 
		# 1.- Is occupied, a red rectangle (0,0,255) will be drawn around the place with a thickness of 5 pixels
		# 2.- Is free, a green rectangle (0,255,0) will be drawn around the place with a thickness of 5 pixels
		if (occupied[i]==1):
            img_status = cv2.rectangle(img_status,(box[0]+10,box[1]+10),(box[2]-10,box[3]-10),(0,0,255),5) 
        else:
            img_status = cv2.rectangle(img_status,(box[0]+10,box[1]+10),(box[2]-10,box[3]-10),(0,255,0),5)
        i = i+1
    
	#plt.imshow(img_status)
    return(img_status)

###########################################
#
# Main program
#
########################################### 

# Set the server URL for the server coming from the console parameters that have been passed in the execution command
url = FRONT_END 

# Init configuration for the camera connected to the parking Raspberry Pi
camera = picamera.PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()

# Camera warm-up time of 2 seconds
time.sleep(2)
ts = time.time()

while (True):	# Starting a daemon to capture parking images automatically each 

	# Calculating the start time of the cycle
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
    print("inference-server.py - Cycle starts at: " + st)
	
	# Capture an image of the parking and save it with the name "image-mini_tmp.jpg"
    camera.capture('image-mini_tmp.jpg')
	
	# If the image has been properly captured
    if os.path.isfile('image-mini_tmp.jpg'):
        
		# Loading the parking image; no flag is used as second parameter meaning this the image will be loaded in color
		# The loaded image is stored as an array of 2 dimension corresponding the dimensions on the image in rows and columns
		img = cv2.imread('image-mini_tmp.jpg') 
		
		# Infering the parking places status (free/occupied) from the loaded image,
		# then marking the positions by adding to the image a green (free) or red (occupied) rectangle
        occupation = scan(img) # Vector of 6 positions with 0 (free) or 1 (occupied) as values
        img_status = boxing(img,occupation) # PIL Image from the parking with 6 rectangles drawn indicating the place availability.		
		
		# Writing the image modified with the 6 rectangles to the local path
		cv2.imwrite('image-mini.jpg',img_status,[int(cv2.IMWRITE_JPEG_QUALITY), 40])
        
		# Adding the information about the image and its related free places to the global json status file
		# indicating the name of the image and how many free places were there when the picture was taken
		jfile = open('status-mini.json',"w")
        s='{"name":"image-mini.jpg", "free":"' + str(6 - np.sum(occupation)) + '" }'
        print("inference-server.py - " + s)
        jfile.write(s) 
        jfile.close()

        # Send the image to be stored in the front-end server
        file = open('image-mini.jpg','rb')
        files = {'file': file}        
		try:
            r = requests.post(url,files=files)
            print("inference-server.py - " + r.text)
        except:
            print("inference-server.py - Error in connection to frontend server")
            print(sys.exc_info()[0]) 
        finally:
            file.close()
       
	    # Send the json file with the image information to be stored in the front-end server
        filej = open('status-mini.json','rb')
        filesj = {'file': filej}        
		try:
            r = requests.post(url,files=filesj)
            print("inference-server.py - " + r.text)
        except:
            print("inference-server.py - Error in connection to frontend server")
            print(sys.exc_info()[0])
        finally:
            file.close()

	else:
	
		# The image has not been taken or taken but not stored locally
		print("inference-server.py - Image file not found in path")
		print(sys.exc_info()[0])
				
	# Calculating the end time of the cycle			
	ts = time.time() 
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
	print("inference-server.py - Cycle stops at: " + st)
	print("------------------------------------")
        
	# Pause 1 second before starting next camera captura
    time.sleep(1)


