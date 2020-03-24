from skimage import feature
import numpy as np
import cv2

import matplotlib.pyplot as plt

""" https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/"""
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist, lbp
def resizeImage(image):
    (h, w) = image.shape[:2]

    width = 12 #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
	#calculate the resize ratio
    ratio = width / float(w)
	#Calculate the dimensions when the height is adjusted accordingly
    dim = (width, int(h * ratio))
	#Rexize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #Return the resized image
    return resized

def extract_LBP(points, image, yard_stick_distance):
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	#Select 8 neighbors and a radius of 1
	desc = LocalBinaryPatterns(8, 1)
	#initiate empty lists
	data = []
	labels = []
	#Create a boundingbox around the landmarks of interest
	x,y,w,h = cv2.boundingRect(points)
	# make the coordinates abs() to prevent errors and add the error margin
	y = int(max([0, y - 0.01 * yard_stick_distance]))
	x = int(max([0, x - 0.01 * yard_stick_distance]))

	w = int(abs(w) + 0.01 * yard_stick_distance)
	h = int(abs(h) + 0.01 * yard_stick_distance)
	print(x, y, w, h)
	# select the part of the image filtered by the bounding box as the region of interest
	ROI = image[y:y+h,x:x+w]
	ROI = resizeImage(ROI)



	#convert ROI to a grayscale image
	gray_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray_img)
	print(hist.shape)
	#save the histogram and convert it to a numpy array
	data.append(hist)
	data = np.array(data)
	data= data.T
	#Return the extracted LBP feature
	return data
