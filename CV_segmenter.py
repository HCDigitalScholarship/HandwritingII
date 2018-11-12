import cv2
import numpy as np
#import image
image = cv2.imread('input.jpg')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,5), np.uint8) #change to kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)

cv2.imshow('marked areas',image)
cv2.waitKey(0)

#####################################
#add this to get individual characters from words
# Resize the contour containing the word using the code as follows.
im = cv2.resize(image,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
#Apply same contour detection process as in line segmentation,
#but with a kernel of size (5,5), i.e :
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(im_th, kernel, iterations=1)
