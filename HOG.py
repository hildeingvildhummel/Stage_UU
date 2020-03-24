from skimage.transform import resize
from skimage.feature import hog
import cv2
from PA import open_landmark_txt
import os
from os import path
import numpy as np
from background_subtraction import remove_background
from SIFT import extract_SIFT
from CNN_feature_extraction import extract_VGG16
from LBP import extract_LBP
from SVM import apply_procrustes_to_centroid, apply_procrustes_to_centroid_prototype
import pickle
import math

def extract_ROI_points(landmarks, str_head_pose, feature):
    """This function extracts the points of interest given the head pose and the feature.
    Input:
    - landmarks: The landmarks of the image as a 2D numpy array.
    - str_head_pose: The head pose of the animal in the images given in string format
    - feature: Pain feature to be extracted. Could either be back_ear, front_ear, nose, second_nose, eye, second_eye, or mouth (Depends on the head pose)
    Output:
    - points: A numpy array containing the landmarks of the region of interest in the right format. """
    # select which head pose is used and then extract the landmarks describing the feature and save this as an array
    #if str_head_pose is given as tilted ...
    if str_head_pose == 'tilted':

        #save for every given feature the points of interest as a numpy array and convert the values to float32
        if feature == 'back_ear':
            points = np.array([[landmarks[0]], [landmarks[1]], [landmarks[2]], [landmarks[3]], [landmarks[4]]], np.float32)
        elif feature == 'front_ear':
            points = np.array([[landmarks[5]], [landmarks[6]], [landmarks[7]], [landmarks[8]], [landmarks[9]]], np.float32)
        elif feature == 'nose':
            points = np.array([[landmarks[16]], [landmarks[17]], [landmarks[18]], [landmarks[19]], [landmarks[20]], [landmarks[21]]], np.float32)
        elif feature == 'eye':
            points = np.array([[landmarks[10]], [landmarks[11]], [landmarks[12]], [landmarks[13]], [landmarks[14]], [landmarks[15]]], np.float32)
        elif feature == 'mouth':
            points = np.array([[landmarks[29]], [landmarks[30]], [landmarks[31]]], np.float32)
        eye_center = [(landmarks[13][0] + landmarks[10][0])/2, (landmarks[12][1] + landmarks[15][1])/2]
        #select the nose center
        nose_center = [(landmarks[18][0] + landmarks[16][0])/2, (landmarks[20][1] + landmarks[17][1])/2]
        #calculate the distance
        yard_stick = list(np.array(eye_center) - np.array(nose_center))
        #make all the distance absolute
        yard_stick = [abs(i) for i in yard_stick]
        yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
    #Perform the exact same approach for the side head pose. The order of the landmarks are altered.
    elif str_head_pose == 'side':

        if feature == 'front_ear':
            points = np.array([[landmarks[0]], [landmarks[1]], [landmarks[2]], [landmarks[3]], [landmarks[4]], [landmarks[5]]], np.float32)
        elif feature == 'nose':
            points = np.array([[landmarks[16]], [landmarks[17]], [landmarks[18]], [landmarks[19]], [landmarks[20]], [landmarks[21]]], np.float32)
        elif feature =='eye':
            points = np.array([[landmarks[22]], [landmarks[23]], [landmarks[24]], [landmarks[25]], [landmarks[26]], [landmarks[27]]], np.float32)
        elif feature == 'mouth':
            points = np.array([[landmarks[12]], [landmarks[13]], [landmarks[14]], [landmarks[15]]], np.float32)
        eye_center = [(landmarks[22][0] + landmarks[25][0])/2, (landmarks[27][1] + landmarks[24][1])/2]
        nose_center = [(landmarks[17][0] + landmarks[20][0])/2, (landmarks[16][1] + landmarks[18][1])/2]
        yard_stick = list(np.array(eye_center) - np.array(nose_center))
        yard_stick = [abs(i) for i in yard_stick]
        yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
    #Perform the exact same approach for the frontal head pose. The order of the landmarks are altered.
    elif str_head_pose == 'front':


        if feature =='back_ear':
            points = np.array([[landmarks[0]], [landmarks[1]], [landmarks[2]], [landmarks[3]], [landmarks[4]]], np.float32)
        elif feature == 'front_ear':
            points = np.array([[landmarks[5]], [landmarks[6]], [landmarks[7]], [landmarks[8]], [landmarks[9]]], np.float32)
        elif feature == 'nose':
            points = np.array([[landmarks[16]], [landmarks[17]], [landmarks[18]], [landmarks[19]], [landmarks[20]], [landmarks[21]]], np.float32)
        elif feature == 'second_nose':
            points = np.array([[landmarks[22]], [landmarks[23]], [landmarks[24]], [landmarks[25]], [landmarks[26]], [landmarks[27]]], np.float32)
        elif feature == 'eye':
            points = np.array([[landmarks[10]], [landmarks[11]], [landmarks[12]], [landmarks[13]], [landmarks[14]], [landmarks[15]]], np.float32)
        elif feature == 'second_eye':
            points = np.array([[landmarks[28]], [landmarks[29]], [landmarks[30]], [landmarks[31]], [landmarks[32]], [landmarks[33]]], np.float32)
        #Both eye centers and nostril centers are calculated and both distances are calculated. The mean of both of these distances is used as the normalization distance
        eye_center = [(landmarks[13][0] + landmarks[10][0])/2, (landmarks[12][1] + landmarks[15][1])/2]
        nose_center = [(landmarks[16][0] + landmarks[18][0])/2, (landmarks[20][1] + landmarks[17][1])/2]
        distance_1 = list(np.array(eye_center) - np.array(nose_center))
        distance_1 = [abs(i) for i in distance_1]
        eye_2_center = [(landmarks[30][0] + landmarks[33][0])/2, (landmarks[28][1] + landmarks[31][1])/2]
        nose_2_center = [(landmarks[23][0] + landmarks[26][0])/2, (landmarks[24][1] + landmarks[22][1])/2]
        distance_2 = list(np.array(eye_2_center) - np.array(nose_2_center))
        distance_2 = [abs(i) for i in distance_2]
        yard_stick = list(np.average(np.array([distance_1, distance_2]), axis=0))
        yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
    #Return the points
    return points, yard_stick_distance

def extract_single_model_points(landmarks, feature):
    """This function extracts the ROI landmark points of the single model.
    It extracts the ear, the eye and the nose of the aligned shape.
    Input:
    - landmarks: a 2D numpy array containing the aligned shape
    - feature: The pain feature of interest, in string format. Could either be ear, eye or nose
    Output:
    - points: The points of interest in the correct format for opencv
    - yard_stick_distance: The normalization yardstick distance """
    #Select the correct landmarks per pain feature and convert it to a numpy float array.
    if feature == 'ear':
        points = np.array([[landmarks[0]], [landmarks[1]], [landmarks[2]], [landmarks[3]], [landmarks[4]]], np.float32)
    elif feature == 'eye':
        points = np.array([[landmarks[5]], [landmarks[6]], [landmarks[7]], [landmarks[8]], [landmarks[9]], [landmarks[10]]], np.float32)
    elif feature == 'nose':
        points = np.array([[landmarks[11]], [landmarks[12]], [landmarks[13]], [landmarks[14]], [landmarks[15]], [landmarks[16]]], np.float32)
    #Select the eye center
    eye_center = [(landmarks[8][0] + landmarks[5][0])/2, (landmarks[7][1] + landmarks[10][1])/2]
    #Select the center of the nostrils
    nose_center = [(landmarks[11][0] + landmarks[13][0])/2, (landmarks[15][1] + landmarks[12][1])/2]
    #Calculate the yardstick distance
    yard_stick = list(np.array(eye_center) - np.array(nose_center))
    yard_stick = [abs(i) for i in yard_stick]
    yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
    #Return the points of interest in the right format and the yardstick distance
    return points, yard_stick_distance


def extract_HOG(points, image, yard_stick_distance):
    """https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/"""

    """This function extracts the HOG features of a region of interest and saves the histogram to a npy file. First it selects the region of interest of the given image and then it extracts the HOG features.
    Input:
    - points: The points of the landmarks of interest in the right format (numpy array of float32)
    - image: The image of the features to be extracted
    - yard_stick_distance: The normalization distance of the face
    Output:
    - fd: The histogram of the HOG features extracted"""

    #get a bounding box around the landmarks of the feature
    x,y,w,h = cv2.boundingRect(points)
    # make the coordinates abs() to prevent errors
    y = int(max(0, y - 0.01 * yard_stick_distance))
    x = int(max(0, x - 0.01 * yard_stick_distance))

    w = int(abs(w) + 0.01 * yard_stick_distance)
    h = int(abs(h) + 0.01 * yard_stick_distance)

    # select the part of the image filtered by the bounding box as the region of interest
    ROI = image[y:y+h,x:x+w]

    #Resize the image to 64x128
    resized_img = resize(ROI, (128,64))

    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    #return the features
    return fd




def get_extraction_features(head_pose, str_head_pose, amount_of_clusters, extraction_method, single_model = False, animal = None, split = 'train', already_augmentated_percentage = None, color = None):
    """This function is the 'main' function for feature extraction. It extracts all the features of a given head pose after aligning the image to its mean shape. It saves the extracted features in a text file
    Input:
    - head_pose: the Excel file containing the missing landmarks per image of the given head pose. If single_model is set to True, this variable must be None
    - feature: The pain feature to be extracted. Could either be back_ear, front_ear, mouth, eye, second_eye, nose or second_nose
    - str_head_pose: The head pose of the given images in string format
    - amount_of_clusters: The amount of clusters used for creating the mean shape. In this case, this variable is always 1 (variances not tested...)
    - single_model: If set to True, no split was made between the head poses. Default is False. If set to True, str_head_pose MUST be None.
    - animal: The animal in the images, could either horse or donkey.
    - split: Define if the data is augmented or not. If set to 'train' the data is augmented (default), if set to 'test' the dataset is not augmented
    - already_augmentated_percentage: given as a float. If given, use the noisy test set with induced noise
    Output:
    - The saved feature numpy files. This function does not return anything. """
    #select the photonumbers
    photonumbers = head_pose.iloc[:,  0]
    #If split is set to train..
    if split == 'train':
        #Repeat all the photoumbers 3 times due to augmentation
        indices = np.repeat(photonumbers, 3)
    #if no augmentation is given..
    else:
        #Keep the original photoumbers
        indices = photonumbers
    #Select pain features based on head pose
    if str_head_pose == 'front':
        features = ['back_ear', 'front_ear', 'eye', 'nose', 'second_eye', 'second_nose']
    elif str_head_pose == 'tilted':
        features = ['back_ear', 'front_ear', 'eye', 'nose', 'mouth']

    elif str_head_pose == 'side':
        features = ['front_ear', 'eye', 'nose', 'mouth']


    #If single model is set to True ..
    if single_model == True:
        #Align the shapes to the single model mean shape
        transformed, M_list = apply_procrustes_to_centroid(head_pose, str_head_pose, amount_of_clusters, single_model = True, split = split)
    #if animal is given and single model set to False..
    elif animal != None and single_model == False:
        if already_augmentated_percentage != None:
            transformed, M_list = apply_procrustes_to_centroid(head_pose, str_head_pose, amount_of_clusters, animal = animal, split = split, already_augmentated_percentage = already_augmentated_percentage)
        else:
            #Align to the right mean shape
            transformed, M_list = apply_procrustes_to_centroid(head_pose, str_head_pose, amount_of_clusters, animal = animal, split = split)
    else:

        #apply the PA to align the face to the mean shape given the K-mean centroid
        transformed, M_list = apply_procrustes_to_centroid(head_pose, str_head_pose, amount_of_clusters, split = split)
    #save the number of ladmarks
    length = int(len(transformed))

    #print the headpose and the feature selected
    print(str_head_pose)
    print(len(indices) - length)
    print(length)

    #Initialize a counter
    counter = 0
    #Set old photonumber
    old_photonumber = 0
    #iterate over all the images
    for i in range(0, length):
        extracted_features = []
        #select the first landmark set
        landmarks = transformed[i]
        #save the corresponding photonumber
        photonumber = indices.iloc[i]
        #print photonumber
        print(photonumber)
        #get the transformation matrix generated from the transformation
        M = M_list[i]
        #If this image does not exist yet, open the original image
        img = cv2.imread('all_images_together/%s.jpg' % (photonumber))
        #save the original landmarks
        original_landmarks = open_landmark_txt('all_landmarks_together/landmarks_%s.txt' % (photonumber))
        #remove the background
        cropped_img = remove_background(img, original_landmarks, photonumber, color = color)

        # #Affine transformation to the image

        image = cv2.warpAffine(cropped_img,M[0:2,:],(2800,4000), borderValue=(255,0,0))

        #If single model is set to True
        if single_model == True:
            print(len(landmarks))
            #Open the original landmarks
            original_landmarks = open_landmark_txt('all_landmarks_together/landmarks_%s.txt' % (photonumber))
            #Check the lenght of the original landmarks to find out which head pose is given in the image
            if len(original_landmarks) == 44:
                str_head_pose = 'tilted'
            elif len(original_landmarks) == 45:
                str_head_pose = 'side'
            elif len(original_landmarks) == 54:
                str_head_pose = 'front'
        #Select the pain features based on the head pose of the given image
        if str_head_pose == 'front':
            features = ['back_ear', 'front_ear', 'eye', 'nose', 'second_eye', 'second_nose']
        elif str_head_pose == 'tilted':
            features = ['back_ear', 'front_ear', 'eye', 'nose', 'mouth']

        elif str_head_pose == 'side':
            features = ['front_ear', 'eye', 'nose', 'mouth']
        #Initialize empty list
        extracted_list = []
        #Iterate over the features
        for feature in features:
            #If single model is set to True..
            if single_model == True:
                #Try to extract the ROI points and the yard_stick distance
                try:

                    #get the points of interest
                    points, yard_stick_distance = extract_ROI_points(landmarks, str_head_pose, feature)
                    missing = False
                #Else, set missing to True
                except:
                    missing = True
            else:
                #get the points of interest

                points, yard_stick_distance = extract_ROI_points(landmarks, str_head_pose, feature)

                missing = False

        #if HOG is given as extractor ..
            if extraction_method == 'HOG':
                #If missing is true, assing the features as an empty vector
                if missing == True:
                    fd = np.zeros(3780)
                #Else, extract the HOG features
                else:

                    fd = extract_HOG(points, image, yard_stick_distance)
                #Append the features to the list
                extracted_features.append(fd)

            #if SIFT is given as extractor..
            elif extraction_method == 'SIFT':
                #If missing is True, asssign features as an empty factor
                if missing == True:
                    fd = np.zeros(128)
                    kp = 1
                else:
                    try:
                        #extract the SIFT features (as histogram)
                        kp, fd = extract_SIFT(points, image, yard_stick_distance)
                    except:
                        fd = np.zeros((1, 128))
                        kp = 1
                #Append every landmark vector seperately to the list
                for i in fd:
                    extracted_features.append(i)

            #if CNN is given as extractor...
            elif extraction_method == 'CNN':
                #If missing is True, assign an empty vector as feature
                if missing == True:
                    fd = np.zeros(4096)
                #Else, extract the CNN features
                else:
                    try:

                        fd = extract_InceptionV3(points, image, yard_stick_distance)
                        kp = 0
                    except:
                        fd = np.zeros(4096)
                        kp = 0
                #Save feature to the list
                extracted_features.append(fd)

            #if LBP is given as extractor...
            elif extraction_method == 'LBP':
                #If missing is set to True, assign an empty vector as feature
                if missing == True:
                    fd = np.zeros((10, 1))
                #Else, extract the LBP features
                else:
                    try:

                        #extract the LBP features
                        fd = extract_LBP(points, image, yard_stick_distance)
                        print(fd.shape)
                    except:
                        fd = np.zeros((10, 1))
                #Save LBP features to list
                extracted_features.append(fd)
        #Convert list to array
        extracted_array = np.array(extracted_features)
        #check if a new photonumber is given. If so, add 1 to the counter. Else, the counter will go back to 0
        if old_photonumber == photonumber:
            counter += 1
        else:
            counter = 0

        if single_model == True:
            #check if path already exits, if not make the path
            if os.path.isdir('Final/%s_features/single_model' % (extraction_method)) == False:
                os.makedirs('Final/%s_features/single_model' % (extraction_method))

            print(extracted_array.shape)
            np.save('Final/%s_features/single_model/%s_features_%s_%s_%s' %(extraction_method, extraction_method, str(counter),str(photonumber), animal), extracted_array)
            old_photonumber = photonumber

        elif already_augmentated_percentage != None:
            #check if path already exits, if not make the path
            if os.path.isdir('Final/%s_features' % (extraction_method)) == False:
                os.makedirs('Final/%s_features' % (extraction_method))

            print(extracted_array.shape)
            #Save the features to numpy file
            np.save('Final/%s_features/%s_features_%s_%s_%s_%s_%s' %(extraction_method, extraction_method, str_head_pose, str(counter),str(photonumber), animal, str(already_augmentated_percentage)), extracted_array)
            old_photonumber = photonumber
        else:
            #check if path already exits, if not make the path
            if os.path.isdir('Final/%s_features' % (extraction_method)) == False:
                os.makedirs('Final/%s_features' % (extraction_method))

            print(extracted_array.shape)
            #Save features to numpy file
            np.save('Final/%s_features/%s_features_%s_%s_%s_%s' %(extraction_method, extraction_method, str_head_pose, str(counter),str(photonumber), animal), extracted_array)
            old_photonumber = photonumber

def extract_features_prototype(image, landmarks, str_head_pose):
    """This function extracts the HOG, LBP, CNN and SIFT features of the ROIs in the image given the head pose and the landmarks.
    First, it aligns the image to the mean shape and then crops the image in a for loop to extract the ROIs and extract the features.
    Input:
    - image: The image from where the features need to be extracted
    - landmarks: The landmarks of the image given as a 2D numpy array
    - str_head_pose; The head pose of the horse in the image in string format. Could either be tilted, side or front
    Output:
    - 4 2D numpy arrays if the features"""
    #Create a list of all the ROI
    feature_list = ['front_ear', 'back_ear', 'mouth', 'eye', 'nose', 'second_eye', 'second_nose']
    #Select features per head pose ADJUST THIS PART TO MATCH WITH THE HEAD POSE DETECTION MODEL
    if str_head_pose == 'tilted':
        features = [0, 1, 2, 3, 4]

    elif str_head_pose == 'side':
        features = [0, 2, 3, 4]

    elif str_head_pose == 'front':
        features = [0, 1, 3, 4, 5, 6]

    #align the shape to the mean shape, given the landmarks and the head pose in string format
    transformed, M = apply_procrustes_to_centroid_prototype(str_head_pose, landmarks)
    #Remove the background of the image
    cropped_img = remove_background(image, landmarks)
    #Transform the image
    img = cv2.warpAffine(cropped_img,M[0:2,:],(2800,4000), borderValue=(255,0,0))
    #Create empty lists to save the extracted features to
    HOG_list = []
    SIFT_list = []
    CNN_list = []
    LBP_list = []
    #Iterate over the pain features in the image:
    for feature_number in features:
        #Select the pain feature
        feature = feature_list[feature_number]
        #get the points of interest
        points, yard_stick_distance = extract_ROI_points(transformed, str_head_pose, feature)
        #Extract the HOG features
        HOG_feature = extract_HOG(points, img, yard_stick_distance)
        print('HOG')
        HOG_list.append(HOG_feature)

        #Extract SIFT
        sift_points, SIFT = extract_SIFT(points, img, yard_stick_distance)
        print('SIFT', SIFT.shape)
        for i in SIFT:
            SIFT_list.append(i)

        #Extract CNN
        CNN_feature = extract_InceptionV3(points, img, yard_stick_distance)
        print('CNN')
        CNN_list.append(CNN_feature)
    #Convert all feature lists to array
    HOG = np.array(HOG_list)
    SIFT = np.array(SIFT_list)
    CNN = np.array(CNN_list)
    print(HOG.shape, SIFT.shape, CNN.shape)
    #Return the extracted features
    return HOG, SIFT, CNN
