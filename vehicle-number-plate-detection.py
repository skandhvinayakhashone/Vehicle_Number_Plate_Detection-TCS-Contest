#importing Packages
import json
import ast
import requests
import os
from PIL import Image
import pytesseract
import cv2
import numpy as np

#getting file name of the json file
filename="vehicle-number-plate-detection+Datasets/Indian_Number_plates.json"

#reading the lines of the json file
with open(filename, 'r') as f:
        lines = f.readlines()

#obtainig the images from the urls in the json file and cropping them with the co-ordinates in the json file and storing an array of images
images=[]
i=0
for line in lines:
    dict_type=json.loads(line)
    image_url=dict_type['content']
    img_data = requests.get(image_url).content
    file_name="Hum_TCS_images/"+str(i)+".jpg"
    with open(file_name, 'wb') as handler:
        handler.write(img_data)
    img = Image.open(file_name)
    x1=dict_type['annotation'][0]['points'][0]['x']*dict_type['annotation'][0]['imageWidth']
    y1=dict_type['annotation'][0]['points'][0]['y']*dict_type['annotation'][0]['imageHeight']
    x2=dict_type['annotation'][0]['points'][1]['x']*dict_type['annotation'][0]['imageWidth']
    y2=dict_type['annotation'][0]['points'][1]['y']*dict_type['annotation'][0]['imageHeight']
    images.append(img.crop((x1,y1,x2,y2)))
    os.remove(file_name)
    i=i+1


#import the tesseract location and configure the type of operation it needs to perform
pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.0.0/bin/tesseract'
config = ('-l eng --oem 1 --psm 3')

final_text=[]
for cropped_image in images:
    opencvImage = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)#convert the image format from Pillow to OpenCV
    blur = cv2.bilateralFilter(opencvImage,15,75,75)#applying bilateral filter to remove blur
    grayImage = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)#converting the blur image to grayscale
    th3 = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,12)#calculationg the adaptive threshold for the gray image
    ret3,th3 = cv2.threshold(th3,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)#changing the scale fo the threshold to 0 to 255
    imagem = cv2.bitwise_not(th3)#inverting the bits from 0 to 255 and vice versa
    kernel = np.ones((1,1),np.uint8)
    eroded_img = cv2.dilate(imagem,kernel,iterations = 1)#dilating the image to get sharper edges
    text = pytesseract.image_to_string(cropped_image, config=config)
    final_text.append(text)

 for i in range(len(final_text)):
 	print("Image "+str(i+1), end="  ")
 	print(text[i])