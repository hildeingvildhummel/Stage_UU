
#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#%============================================================================
import numpy as np
import os
import pickle
import cv2 as cv
import shutil
import random
import csv
import matplotlib.pyplot  as plt
random.seed(10)

DATASET = os.path.join(os.getcwd(), 'Final/dataset')

#create directory to save absolute pose images
ANIMAL = 'horse'
TEST_LIST = os.path.join(DATASET,'test_' + ANIMAL + 's.csv')

if ANIMAL == 'horse':
    ABS_POSE = os.path.join(DATASET,'abs_pose')
elif ANIMAL == 'donkey':
    ABS_POSE = os.path.join(DATASET,'abs_pose_donkeys')

POSE_0 = os.path.join(ABS_POSE, '0_30')
POSE_30 = os.path.join(ABS_POSE, '30_60')
POSE_60 = os.path.join(ABS_POSE, '60_90')

if not os.path.exists(ABS_POSE):
    for dir_ in [ABS_POSE, POSE_0, POSE_30, POSE_60]:
        os.makedirs(dir_)

    for folder in [POSE_0, POSE_30, POSE_60]:
        os.makedirs(os.path.join(folder, 'test'))

data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))
# data = pickle.load(gzip.open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))

#shape_0 = (500, 265)
#shape_30 = (447, 500)
#shape_60 = (363, 500)

shape_0 = (265,500)
shape_30 = (500, 447)
shape_60 = (500, 363)


#%%============================================================================
#                               FUNCTIONS
#%============================================================================

def create_pts(lms, pts_path):
    with open(pts_path, 'w') as pts_file:
        pts_file.write('version: 1\n')
        pts_file.write('n_points:  ' + str(len(lms)) + '\n')
        pts_file.write('{\n')
        for (x, y) in lms:
            pts_file.write(str(int(x)) + ' ' + str(int(y)) + '\n')
        pts_file.write('}')

shapes = []
def crop_image(img, lms, pose, name):

    error = 0.10
    lms = np.vstack(lms)
    lms_x = lms[:,0]
    lms_y = lms[:,1]


    img_h, img_w = img.shape[:2]
    x_min =  max(0,int(min(lms_x) - error * img_w))
    x_max = min(img_w, int(max(lms_x) + error * img_w))

    y_min = max(0, int(min(lms_y) - error * img_h))
    y_max = min(img_h, int(max(lms_y) + error * img_h))

    """
    Test - landmark position (original image)
    =======================================

    img_original = img.copy()
    print('image shape : ', np.shape(img))
    for pt in lms:
        r = 5
        cv.circle(img_original, (int(pt[0]), int(pt[1])), r, (255,0,0), -1)

    cv.imshow('original', img_original)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    img_resize = img[y_min : y_max, x_min : x_max]
    resize_h, resize_w = img_resize.shape[:2]

    if pose == 0:
        img_resize =  cv.resize(img_resize, shape_0)
        new_h = shape_0[0]
        new_w = shape_0[1]
    elif abs(pose) == 30:
        img_resize =  cv.resize(img_resize, shape_30)
        new_h = shape_30[0]
        new_w = shape_30[1]
    elif abs(pose) == 60:
        img_resize =  cv.resize(img_resize, shape_60)
        new_h = shape_60[0]
        new_w = shape_60[1]

    lms_resize = []
    image_copy = img_resize.copy()
    for pt in lms:
        new_pt = (int((pt[0] - x_min) * new_h/resize_w), int((pt[1] - y_min) * new_w/resize_h))
        lms_resize.append(new_pt)
        r = 2
        cv.circle(image_copy, new_pt, r, (255,0,0), -1)

    """

    n = 1
    x = [l[0] for l in lms_resize]
    y = [l[1] for l in lms_resize]


    plt.figure()
    plt.imshow(cv.cvtColor(img_resize, cv.COLOR_BGR2RGB))
    plt.plot(x,y,'ro',markersize=3)
    for xy in zip(x, y):
        plt.annotate('%s' % n, xy = xy, color='white', size=8)
        n += 1

    plt.savefig('%s.png' % name)

    Test - landmark position (resized image)
    =======================================

    cv.imshow(name, image_copy)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """

    return img_resize, lms_resize

def mean_shape_pose(shapes):
    shapes = np.vstack(shapes)
    shapes_0 = shapes[shapes[:,0] == 0]
    shapes_30 = shapes[shapes[:,0] == 30]
    shapes_60 = shapes[shapes[:,0] == 60]

    print('Check dimensions: ', len(shapes), ' vs ', len(shapes_0) + len(shapes_30) + len(shapes_60))

    mean_0 = np.mean(shapes_0[:,1] /shapes_0[:,2] , axis = 0)
    mean_30 = np.mean(shapes_30[:,1] /shapes_30[:,2], axis = 0)
    mean_60 = np.mean(shapes_60[:,1] /shapes_60[:,2], axis = 0)

    median_0 = np.median(shapes_0[:,1] /shapes_0[:,2], axis = 0)
    median_30 = np.median(shapes_30[:,1] /shapes_30[:,2], axis = 0)
    median_60 = np.median(shapes_60[:,1] /shapes_60[:,2], axis = 0)

    std_0 = np.std(shapes_0[:,1] /shapes_0[:,2], axis = 0)
    std_30 = np.std(shapes_30[:,1] /shapes_30[:,2], axis = 0)
    std_60 = np.std(shapes_60[:,1] /shapes_60[:,2], axis = 0)

    print('Mean shapes_0: ', mean_0)
    print('Mean shapes_30: ', mean_30)
    print('Mean shapes_60: ', mean_60)

    print('Median shapes_0: ', median_0)
    print('Median shapes_30: ', median_30)
    print('Median shapes_60: ', median_60)

    print('Std shapes_0: ', std_0)
    print('Std shapes_30: ', std_30)
    print('Std shapes_60: ', std_60)


    return mean_0, mean_30, mean_60

def save_resized_img_and_pts():
    for img_info in data.values:
        #filter the images that don't have landmark annotations!
        if img_info[-2] is not None and img_info[1] == ANIMAL:
            pose = img_info[2]
            img_name = img_info[0].split('/')[-1]

            img = cv.imread(os.path.join(os.getcwd(), img_info[0]))
            if pose < 0:
                img = cv.flip(img, 1)
                h, w = np.shape(img)[:2]
                mirror_lms = []
                for pt in img_info[-1]:
                    new_pt = [w - pt[0], pt[1]]
                    mirror_lms.append(new_pt)

                lms = np.vstack(mirror_lms)
                pose = -pose
            else:
                lms = img_info[-1]

            img, lms = crop_image(img, lms, pose, img_name)

            if pose == 0:
                img_path = os.path.join(POSE_0, img_name)
            elif pose == 30:
                img_path = os.path.join(POSE_30, img_name)
            elif pose == 60:
                img_path = os.path.join(POSE_60, img_name)

            pts_path = img_path.replace('jpg', 'pts')
            create_pts(lms, pts_path)

            cv.imwrite(img_path, img)

def get_jpgs(directory):
    jpgs = []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            jpgs.append(os.path.join(directory, file))
    return jpgs

def extract_test_set():
    images_0 = get_jpgs(POSE_0)
    images_30 = get_jpgs(POSE_30)
    images_60 = get_jpgs(POSE_60)

    print(len(images_60))
    test_set = []
    with open(TEST_LIST) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            test_set.append(int(row[0]))

    a = 0
    b = 0
    c = 0

    for test_image in test_set:

        for i_0 in images_0:
            if '/' + str(test_image) + '.jpg' in i_0:
                a += 1
                shutil.move(i_0, os.path.join(POSE_0, 'test', i_0.split('/')[-1]))
                shutil.move(i_0.replace('jpg', 'pts'), os.path.join(POSE_0, 'test', i_0.split('/')[-1].replace('jpg', 'pts')))

        for i_30 in images_30:
            if '/' + str(test_image) + '.jpg' in i_30:
                b += 1
                shutil.move(i_30, os.path.join(POSE_30, 'test', i_30.split('/')[-1]))
                shutil.move(i_30.replace('jpg', 'pts'), os.path.join(POSE_30, 'test', i_30.split('/')[-1].replace('jpg', 'pts')))

        for i_60 in images_60:
            if '/' + str(test_image) + '.jpg' in i_60:
                c += 1
                shutil.move(i_60, os.path.join(POSE_60, 'test', i_60.split('/')[-1]))
                shutil.move(i_60.replace('jpg', 'pts'), os.path.join(POSE_60, 'test', i_60.split('/')[-1].replace('jpg', 'pts')))

    #print('a: ', a)
    #print('b: ', b)
    #print('c: ', c)
#%%============================================================================
#                               MAIN
#%%============================================================================

#mean_0, mean_30, mean_60 = mean_shape_pose(shapes)
def main():
    save_resized_img_and_pts()
    extract_test_set()

#main()
