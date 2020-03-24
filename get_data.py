#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#%%============================================================================

import pandas as pd
import os
import math
import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import ExifTags
from PIL import Image

DATASET = os.path.join(os.getcwd(), 'Final/dataset')
lms_folder = os.path.join(DATASET, 'landmarks')

#%%============================================================================
#                          CREATE DATA STRUCTURE
#============================================================================

filename = os.path.join(DATASET, 'pose_annotations.xlsx')
data = pd.read_excel(filename)
data = data.iloc[:, 0:2]
data.columns = ['image', 'pose']
print(data.head(5))

#%% Change image col to image path
#-----------------------------------------------------------------------------
data['image'] = [os.path.join('Final', 'dataset', 'images', str(d) + '.jpg') for d in data.loc[:,'image'].values]



#%% Convert pose from string to number
#-----------------------------------------------------------------------------
"""
left side = [-90 -60]
left tilted = [-60 -30]
front = [-30 0 30]
right tilted = [30 60]
right side = [60 90]
"""

data.pose.loc[data.pose == 'left side'] = -60
data.pose.loc[data.pose == 'left tilted'] = -30
data.pose.loc[data.pose == 'front'] = 0
data.pose.loc[data.pose == 'right tilted'] = 30
data.pose.loc[data.pose == 'right side'] = 60



#%% Save landmarks for each image
#-----------------------------------------------------------------------------
lms_files = [os.path.join(lms_folder, l) for l in  os.listdir(lms_folder)]
lms_files.sort()

lms = [None] * len(data)
for file in lms_files:
    if file.endswith('.txt'):

        index = int(file.split('/')[-1].split('.')[0].split('_')[-1]) - 1 #the image name start in 1 and not 0
        f = open(file, 'r')
        x_y = f.read().replace('\n',' ')
        x_y = np.array(list(map(float, x_y.split()))).reshape(-1,2)
        lms[index] = np.vstack(x_y)
        f.close()

data['landmarks'] = lms

"""
Test if the landmarks are in the right place
=============================================

for i,img_name in enumerate(data['image'].values):
    img = cv.imread(img_name)

    for pt in data.at[i,'landmarks']:
        cv.circle(img, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1) # coordenates - x, y



    cv.imshow('Horse face - %s' % (img_name), img)
    cv.waitKey(0)
    cv.destroyAllWindows()
"""


#%% Convert all numbers from string to float (lms) and int (occlusions)
#-----------------------------------------------------------------------------

occs = [None] * len(data)
for i,d in enumerate(data.loc[:,'occlusions'].values):
    if type(d) == str:
        occs[i] = [int(k) for k in d.split(',')]
    elif math.isnan(d) is False:
        occs[i] = int(d)
    else:
        occs[i] = []

data['occlusions'] = occs

data.to_pickle(os.path.join(DATASET, 'lms_annotations.pkl'))


#%% Draw landmarks

# The mpimg library that was being used by Hilde and Bram (and was also used to
# make the landmarks does not support jpeg images - so, this support is made by
# the Pillow library. However, this library reads the metadata of the image but can't
# alter it, so the orientation (normally register and corrected by the camera) is not
# altered...


"""
img = Image.open(filename)
print(img._getexif().items())
exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
if not exif['Orientation']:
    img=img.rotate(90, expand=True)
img.thumbnail((1000,1000), Image.ANTIALIAS)
img.save(output_fname, "JPEG")

"""
for i, info in enumerate(data.values):
    img_cv = cv.imread(info[0])
    img_mpimg = mpimg.imread(info[0])

    if len(np.shape(img_mpimg)) > 2:
        img_mpimg = cv.cvtColor(img_mpimg, cv.COLOR_RGB2BGR)

    if np.shape(img_mpimg) != np.shape(img_cv) or np.sum(img_mpimg - img_cv) != 0:
        image = Image.open(info[0])
        #print(img._getexif().items())
        if hasattr(image, '_getexif'): # only present in JPEGs
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            e = image._getexif() # returns None if no EXIF data

            if e is not None:
                exif=dict(e.items())
                if orientation in exif: # If the exif file as a orientation parameter (some don't have)
                    orientation = exif[orientation]
                    print('%d - orientation: ' % i, orientation, info[0])
                    if orientation == 3:
                        image = image.transpose(Image.ROTATE_180)
                    if orientation == 6:
                        in_h, in_w = np.shape(image)[:2]
                        image = image.transpose(Image.ROTATE_270)
                        lms = info[-2]
                        h, w = np.shape(image)[:2]

                        #data.at[i,'landmarks'] = [[w - pt[1], pt[0]] for pt in lms]
                        #data.at[i, 'lms_mirror'] = [[pt[1], pt[0]] for pt in lms]


                        for pt in data.at[i,'landmarks']:
                            cv.circle(img_cv, (int(pt[0]), int(pt[1])), 40, (0,0,255), -1) # coordenates - x, y

                        """
                        cv.imshow('orientation 6', img_cv)
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                        """
                    elif orientation == 8:
                        image = image.transpose(Image.ROTATE_90)
                        lms = info[-2]
                        h, w = np.shape(image)[:2]
                        #data.at[i,'landmarks'] = [[pt[1], h - pt[0]] for pt in lms]
                        #data.at[i, 'lms_mirror'] = [[w - pt[1], h - pt[0]] for pt in lms]

                        for pt in data.at[i,'landmarks'] :
                            cv.circle(img_cv, (int(pt[0]), int(pt[1])), 40, (255,0,0), -1) # coordenates - x, y

                        """
                        cv.imshow('orientation 8', img_cv)
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                        """


"""

    if np.shape(img_mpimg) != np.shape(img_cv):
        print(i)
    elif np.sum(img_cv - img_mpimg) != 0:
        print(i)
"""

# data.to_pickle(os.path.join(DATASET, 'lms_annotations.pkl'))
