#importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
from PA import open_landmark_txt
from PA import generalized_procrustes_analysis
from PA import apply_procrustes_transform
import os
from SVM import open_landmark_txt, apply_procrustes_to_centroid
import numpy as np
import pandas as pd
from PA import procrustes, open_landmark_txt
from PIL import Image
import matplotlib.cm as cm
# from histogram import hog, visualise_histogram
from skimage import color
from background_subtraction import remove_background
from numpy.linalg import norm
from sklearn.cluster import MiniBatchKMeans

"""https://www.kaggle.com/pierre54/bag-of-words-model-with-sift-descriptors"""
def extract_SIFT(points, image, yard_stick_distance):
    """This function first determines the region of interest and then extracts the SIFT features.
    Input:
    - points: numpy array of type float32 containing the landmarks of interest of a specific region
    - image: the image where you would like to extract the features from
    - yard_stick_distance: The normalization distance of the face
    Output:
    - kp: the number of keypoints detected
    - des: per keypoint a 128 vector describing the keypoint found """
    #get a bounding box around the landmarks of the feature
    x,y,w,h = cv2.boundingRect(points)
    # make the coordinates abs() to prevent errors and add the margin to the bounding box
    y = int(max(0, y - 0.01 * yard_stick_distance))
    x = int(max(0, x - 0.01 * yard_stick_distance))

    w = int(abs(w) + 0.01 * yard_stick_distance)
    h = int(abs(h) + 0.01 * yard_stick_distance)

    # select the part of the image filtered by the bounding box as the region of interest
    ROI = image[y:y+h,x:x+w]

    #fix the width to 64 and adjust the height accordingly
    ratio = 64 / w
    dim = (64, int(h * ratio))
    print(dim)
    print(ROI.shape)
    if int(h * ratio) < 2:
        dim = (64, 64)
    #Resize the image
    resized = cv2.resize(ROI, dim, interpolation=cv2.INTER_AREA)
    #convert ROI to a grayscale image
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    #intiate an empty list
    kp = []
    #Iterate over all the landmarks of interest
    for x_1 in points:
        x_1 = x_1.flatten()
        # Convert the landmark to a keypoint
        keypoint = cv2.KeyPoint(max(0, x_1[0] - x), max(0, x_1[1] - y), 7)
        #Save the keypoint to a list
        kp.append(keypoint)
    #Create SIFT descriptor
    sift = cv2.xfeatures2d.SIFT_create()
    #Compute the SIFT feature based on the landmark keypoints 
    kp1, des1 = sift.compute(gray_img, kp, np.array([]))

    #Return the keypoints
    return kp1, des1
