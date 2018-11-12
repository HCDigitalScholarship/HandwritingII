

import numpy as np
import argparse
import cv2 # OpenCV 3.4.2 required
from imutils.object_detection import non_max_suppression #starred repo..as an alternative an use NMSBoxes(CV)

# constructs  argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "path-to-image", type=str) #MUST include path to image
ap.add_argument("-east", "path-to-EAST", type=str) #MUST include path to trained EAST model
ap.add_argument("-c", "--min-confidence", type=float default=0.5)
#ap.add_argument("-w", "--width", type=int, default=320) probs dont need
#ap.add_argument("-e", "--height", type=int, default=320) probs dont need
#Should adjust ROI padding depending on results
ap.add_argument("-p", "--padding", type=float, default=0.0)
args = vars(ap.parse_args())

def predictions(scores, geometry): #detects regions of text produces 2 arrays(probability
#of given area having text present and mapping to bounding box location in image).
# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding  confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidence = []

	# loop over the number of rows
	for y in range(0, numRows):
	# extract the probabilities, followed by the geometrical data used to get potential bounding box
	# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability it will be ignored
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be  downsized to 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction & compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# returns a tuple of the bounding boxes and associated confidences
	return (rects, confidence)

# load the input image and obtai image dimensions
image = cv2.imread(args["image"])
orig = image.copy() # copied so output results can be drawn later
(origH, origW) = image.shape[:2]

#might not need to perform grayscale and binary transformations...
#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)

# set the new width and height and then determine the ratio(used to scale bounding box coordinates)
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

#EAST detector:
#  output layer names for the EAST detector model that we are interested(must keep naming)
# first layer = output probabilities
# second layer =  for deriving the bounding box coordinates
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST detector
print("loading EAST  detector...")
net = cv2.dnn.readNet(args["east"]) #MAKE SURE v = OpenCV 3.4.2

# constructs a blob from the image and does a forward pass of the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames) #scores and geomerty are obtained from first pass

#gives predictions, then  applies non-maxima suppression to suppress weak/ overlapping bounding boxes
(rects, confidences) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)


results = [] # lists of results which will contain bounding boxes and text
# loops over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	#scales bounding boxes, pads the boxes and extracts the padded region of interest(roi)
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])
	# applies padding to each side of the bounding box
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	RegionInterest = orig[startY:endY, startX:endX]
    #possibly remove below
	results.append(((startX, startY, endX, endY), RegionInterest))
print results #confirmation
