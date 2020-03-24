import numpy as np
import cv2
from skimage import draw
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import os


def remove_background(img, landmarks, photonumber = None):
    """This function removes the background from a given image, resulting in a background solely containing the horse head. The background is removed by
    using a convex hull. The image will be saved in the cropped directory if a photonumber is specified.
    Input:
    - img: the image from which the background needs to be subtracted
    - landmarks: the landmarks of the given image
    - photonumber: if given, the image with the removed background will be saved. Default is None
    Output:
    - cropped_i: the image with the background removed"""
    #check the number of landmarks to define the head pose and the given contour landmarks
    if len(landmarks) == 44:
        start = landmarks[0:10]
        end = landmarks[24:44]
    elif len(landmarks) == 45:
        start = landmarks[0:6]
        end = landmarks[28:45]
    elif len(landmarks) == 54:
        start = landmarks[0:10]
        end = landmarks[35:54]
    #Concatenate the contour landmarks into a single numpy array
    points = np.concatenate((start, end),axis=0)

    #Create the convex hull given the contour landmarks
    hull = ConvexHull(points)
    #Draw the convex hull
    Y, X = draw.polygon(points[hull.vertices,1], points[hull.vertices,0], img.shape)
    #Create an empty image
    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    #Fill the background using red
    a_2d_index = np.array([1,0,0]).astype('bool')
    a_1d_fill = 255
    cropped_img[:,:,a_2d_index] = a_1d_fill
    #Save the inside of the convex hull on the empty red image
    cropped_img[Y, X] = img[Y, X]
    #if photonumber is given..
    if photonumber != None:
        #check if path already exits, if not make the path
        if os.path.isdir('Final/cropped_images') == False:
            os.makedirs('Final/cropped_images')
        #save the cropped image
        cv2.imwrite('Final/cropped_images/cropped_%s.jpg' % (photonumber), cropped_img)
    #return the images with the background removed
    return cropped_img
