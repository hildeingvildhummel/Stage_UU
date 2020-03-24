import numpy as np
import pandas as pd
from PA import apply_procrustes_transform
import math
from k_means import scaling
from PA import open_landmark_txt
import itertools
import pickle

def procrustes_single_model(photonumber, mean_shape, filtered_matrix, tr_Y_pts):
    """This function aligns the input shape to the mean shape and predicts the places of the missing landmarks by applying the transformation to the original landmarks.
    Input:
    - photonumber: The photonumber of the given image
    - mean_shape: The consensus shape of the single model
    - filtered_matrix: The filtered landmarks containing a single eye, ear and nostril
    - tr_Y_pts: The points which are already aligned to the mean shape
    Output:
    - tr_Y_pts: the total shape containg all the landmarks which are either aligned or guessed"""
    #Open the orignal landmark file as a 2D array
    original_landmarks = open_landmark_txt('all_landmarks_together/landmarks_%s.txt' % (str(photonumber)))
    #Define which landmarks are missing
    if len(original_landmarks) == 44:
        missing_landmarks = [0, 1, 2, 3, 4, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    elif len(original_landmarks) == 54:
        missing_landmarks = [0, 1, 2, 4, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    elif len(original_landmarks) == 45:
        missing_landmarks = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

    #return the transformation components of the alignment of the filtered matrix to the filtered mean shape
    d, z, Tform = procrustes(filtered_matrix, mean_shape)
    # Build and apply transform matrix...
    # Note: for affine need 2x3 (a,b,c,d,e,f) form
    R = np.eye(3)
    R[0:2,0:2] = Tform['rotation']

    S = np.eye(3) * Tform['scale']
    S[2,2] = 1
    t = np.eye(3)
    t[0:2,2] = Tform['translation']
    M = np.dot(np.dot(R,S),t.T).T


    aY_pts = np.hstack((original_landmarks[missing_landmarks],np.array(([[1]*len(original_landmarks[missing_landmarks])])).T))
    #guess the location of the missing landmarks
    guessed_tr_Y_pts = np.dot(M,aY_pts.T).T
    guessed_tr_Y_pts = np.delete(guessed_tr_Y_pts, -1, axis=1)
    counter = -1
    #iterate over the indices of the missing landmarks
    for missing_index in missing_landmarks:
        counter += 1
        #try to insert the guessed landmark into the aligned filtered dataframe
        try:
            tr_Y_pts = np.insert(tr_Y_pts, missing_index, guessed_tr_Y_pts[counter], 0)
        #else, add the guessed landmarks to the end of the aligned filtered dataframe
        except:
            tr_Y_pts = np.vstack((tr_Y_pts, guessed_tr_Y_pts[counter]))
    #Return the complete lanmark array
    return tr_Y_pts

def apply_procrustes_to_centroid(head_pose, str_head_pose, amount_of_clusters = 1, single_model = False, animal = None, split = 'train', already_augmentated_percentage = None):
    """This function applies the PA transformation of all the shapes, given the head pose, to the mean shape.
    Input:
    - head_pose: Excel file containing the photonumbers, head pose and missing landmarks
    - str_head_pose: The head pose of the given images in string format. Could either be tilted, side or front
    - amount_of_clusters: The number of clusters used for creating the centroid. Default is 1
    - single_model: If given, use a single mean shape independent of the given head pose. Default is False
    - animal: Which animal is given in the image, could either be horse or donkey.
    - split: If the shapes are from the test set or from the training set, given in string format. Could either be train or test. Default is train
    - already_augmentated_percentage: If noise is induced. Default is None. For this project, 0.04 and 0.08 are used
    Output:
    - transformed_shapes: One array of shapes aligned to its mean shape
    - M: A list of transformation matrices which is applied """

    #create one big array containing all the landmarks together
    if single_model == True:

        X, indices = scaling(head_pose, str_head_pose, single_model = True, save = True, split = split, animal = 'horse')

    elif animal != None and single_model == False:
        if already_augmentated_percentage != None:
            X, indices = scaling(head_pose, str_head_pose, animal = animal, split = split, already_augmentated_percentage = already_augmentated_percentage)
        else:
            X, indices = scaling(head_pose, str_head_pose, animal = animal, split = split)
        print(len(X))

    else:

        X, indices = scaling(head_pose, str_head_pose, split = split)
    #Open the labels assigned by the KMeans
    if single_model == True:
        label_file = open('Final/labels_horses/labels_single_model_%s_clusters.txt' % (amount_of_clusters), 'r')
    elif animal != None:
        if animal == 'donkey':
            label_file = open('Final/labels_horses/labels_noisy_%s_clusters_%s_%s_all.txt' % (amount_of_clusters, str_head_pose, animal), 'r')
        else:
            label_file = open('Final/labels_horses/labels_noisy_%s_clusters_%s_%s_all.txt' % (amount_of_clusters, str_head_pose, animal), 'r')

    else:
        label_file = open('Final/labels_horses/labels_noisy_%s_clusters_%s.txt' % (amount_of_clusters, str_head_pose), 'r')
    #Read the labels file
    label_file = label_file.read().splitlines()

    #create empty list to append the trandformed shapes
    transformed_shapes = []
    #create empty list to append rotation matrix
    M_list = []
    #iterate over all the landmarks
    for index in range(0, len(X)):

        #get the label of the landmark shape
        cluster_number = label_file[index]
        #remove the enters from the clusternumber
        cluster_number = cluster_number.replace('\n','')
        #open the mean shape
        if single_model == True:
            mean_shape = open('Final/labels_horses/center_single_model_%s_%s_clusters.txt' % (cluster_number, amount_of_clusters))
        elif animal != None and single_model == False:
            mean_shape = open('Final/labels_horses/center_noisy_%s_%s_clusters_%s_%s_all.txt' % (cluster_number, amount_of_clusters, str_head_pose, animal))
        else:

            #open and read the mean shape
            mean_shape = open('Final/labels_horses/center_noisy_%s_%s_clusters_%s.txt' % (cluster_number, amount_of_clusters, str_head_pose))
        mean_shape = mean_shape.read()
        #replace the enters and tabs in the txt file
        mean_shape = mean_shape.replace('\n',' ')
        #create an array out of the txt file coordinates
        mean_shape = np.array(list(map(float, mean_shape.split()))).reshape(-1, 2)
        #filter the list of missing landmarks from the excel file
        photonumber = indices[index]
        print('apply procrutes: ', photonumber)
        missing = head_pose.loc[head_pose.iloc[:, 0] == int(photonumber)]
        missing = missing.dropna()
        #check if there are missing landmarks
        if len(missing) > 0:
            #make a list out of the missing landmarks

            missing = missing.values
            flat_list = [item for sublist in missing for item in sublist]
            missing = flat_list[2:]
            missing_landmarks = [int(i) for i in missing]


            #remove the missing landmarks from the array
            filtered_matrix = np.delete(X[index], missing_landmarks, axis=0)
            # remove the same landmarks from the mean shape
            filtered_mean_shape = np.delete(mean_shape, missing_landmarks, axis=0)
            #apply procrustes transform to align the filtered matrix to the filtered mean shape
            tr_Y_pts, M, generated_mean_shape, d = apply_procrustes_transform(filtered_matrix, filtered_mean_shape, len(filtered_matrix))

            #return the transformation components of the alignment of the filtered matrix to the filtered mean shape
            d, z, Tform = procrustes(filtered_matrix, filtered_mean_shape)
            # Build and apply transform matrix...
            R = np.eye(3)
            R[0:2,0:2] = Tform['rotation']

            S = np.eye(3) * Tform['scale']
            S[2,2] = 1
            t = np.eye(3)
            t[0:2,2] = Tform['translation']

            M = np.dot(np.dot(R,S),t.T).T
            #append the rotation matrix to the list
            M_list.append(M)
            # Confirm points...
            try:

                aY_pts = np.hstack((mean_shape[missing_landmarks],np.array(([[1]*len(mean_shape[missing_landmarks])])).T))
                #guess the location of the missing landmarks
                guessed_tr_Y_pts = np.dot(M,aY_pts.T).T
                guessed_tr_Y_pts = np.delete(guessed_tr_Y_pts, -1, axis=1)
                counter = -1
                #iterate over the indices of the missing landmarks
                for missing_index in missing_landmarks:
                    counter += 1
                    #try to insert the guessed landmark into the aligned filtered dataframe
                    try:
                        tr_Y_pts = np.insert(tr_Y_pts, missing_index, guessed_tr_Y_pts[counter], 0)
                    #else, add the guessed landmarks to the end of the aligned filtered dataframe
                    except:
                        tr_Y_pts = np.vstack((tr_Y_pts, guessed_tr_Y_pts[counter]))
                #append the aligned shape, with the guessed landmarks, to the list
                transformed_shapes.append(tr_Y_pts)
            except:
                continue
        #if there are no missing landmarks
        else:
            #align the shape to the mean shape
            tr_Y_pts, M, generated_mean_shape, d = apply_procrustes_transform(X[index], mean_shape, len(X[index]))
            if single_model == True:
                tr_Y_pts = procrustes_single_model(photonumber, mean_shape, X[index], tr_Y_pts)
                print(len(tr_Y_pts))
            #append the aligned shape to the list
            transformed_shapes.append(tr_Y_pts)
            #append the rotation matrix to the list
            M_list.append(M)

    #convert the list containing all the  shapes into an array
    transformed_shapes = np.array(transformed_shapes)
    #return the transformed shapes and the rotation matrices
    return transformed_shapes, M_list

def apply_procrustes_to_centroid_prototype(str_head_pose, landmarks):
    """This function alignes a single new shape to the mean shape given the landmarks and the head pose
    Input:
    - str_head_pose: The head pose of the horse, in string format. Could either be front, tilted or side
    - landmarks: The landmarks of the image as a 2D array
    Output:
    - tr_Y_pts: The aligned landmarks to the mean shape
    - M: the transformation matrix"""
    #Open the mean shape as a 2D array
    mean_shape = open('Final/labels_horses/center_noisy_0_1_clusters_%s_horse_all.txt' % (str_head_pose))
    mean_shape = mean_shape.read()
    #replace the enters and tabs in the txt file
    mean_shape = mean_shape.replace('\n',' ')
    #create an array out of the txt file coordinates
    mean_shape = np.array(list(map(float, mean_shape.split()))).reshape(-1, 2)
    #align the shape to the mean shape
    tr_Y_pts, M, generated_mean_shape, d = apply_procrustes_transform(landmarks, mean_shape, len(landmarks))
    #Return the aligned landmarks an the transformation matrix
    return tr_Y_pts, M


def create_HOG_dataframe_with_induced_noise(photonumbers, str_head_pose, features, extraction_method = 'HOG', induced_noise = 0, single_model = False, animal = 'horse', split = 'train', automatically_SIFT = False, color = None):
    """This function creates one big list containing all the pain features together per image. It iterates over all the images and then over all the pain features. Choose between the training set and the test set
    Input:
    - photoumbers: a list containing the photonumbers of all the images
    - str_head_pose: the head pose of the images in string format. If single_model is set to True, set this to None
    - features: the names of the ROI extracted, could either be front_ear, back_ear, nose, eye, mouth, second_eye or second_nose. In string format.
    - extraction_method: the extracted features to be used to make the big array. Could either be: SIFT, HOG, LBP or CNN. Default is HOG
    - induced_noise: if given, use the features with induces noise. This if the augmentation, alignment and extraction is performed again with the noisy landmarks! Default is 0, meaning no noise.
    - single_model: if set to True, the features extracted from the single model (no split in head pose) is used. The str_head_pose MUST be set to None, if this variable is set to True. Default is False.
    - animal: The animal in the image, could either be horse or donkey. Default is donkey.
    - split: Determine if the data is augmentated or not. Choose between train (augmenated) or test (not augmentated). Default is train.
    - automatically_SIFT: If set to true, the keypoints will be used which are automatically computed. Default if False
    Output:
    - X_data: The big dataframe containing all the pain features of all the images in the training set or test set
    -sample_numbers: If SIFT is given, it returns a list of the number of keypoints in per image. Otherwise, it returns a list of zeros.
    """



    #Intialize an old photonumber
    old_photonumber = 0
    #make an empay list to append the whole dataset to
    X_data = []
    #Initialize a counter
    counter = 0
    #Create an empty dataframe to save the keypoint numbers to
    sample_numbers = []
    # first_photo = 0
    feature_list = []
    #iterate over all the photonumbers
    for photonumber in photonumbers:
        #print the progress
        print('create HOG dataframe induced: ', photonumber)
        # print('old photonumber: ', old_photonumber)
        #check if train is given, so if the data is augmentated
        if induced_noise == 0 and split == 'train':
            #if photoumber is the same
            if photonumber == old_photonumber:
                #add 1 to the counter
                counter += 1
            #if the counter is equal to 3
            elif counter == 3:
                #set counter to 0 again
                counter = 0
            #Otherwise, set counter to 0
            else:
                counter = 0

        #If induced noise is given..
        elif induced_noise != 0:
            #if the photoumber did not change..
            if photonumber == old_photonumber:
                #add 1 to the counter
                counter += 1
            #otherwise set counter to 0
            else:
                counter = 0
        #if split is set to test, counter is alwoys 0
        if split == 'test':
            counter = 0

        #make a string format of the counter
        percentage = str(counter)

        #Intialize an empty list
        all_HOG = []

        #Initialize a variable to add the total number of keypoints to per image
        scoring_number = 0

        #If extraction method is set as HOG..
        if extraction_method == 'HOG':
            #Open the corresponding extracted features and save it as a 1D array to a list
            if single_model == True:
                try:
                    HOG = np.load('Final/HOG_features/single_model/HOG_features_%s_%s_%s.npy' % (percentage, photonumber, animal), allow_pickle = False)
                    if features == None:
                        #If no features are specified, use all the ROIs
                        all_HOG.append(HOG.flatten())
                    else:
                        #Otherwise, only save the corresponding extracted features
                        all_HOG.append(HOG[features].flatten())
                    print(HOG[features].flatten().shape)
                except:
                    print('except')
                    all_HOG.append(np.zeros(len(features) * 3780))

            #If no induced noise is given..
            elif induced_noise == 0:
                #Open the corresponding extracted features and save it as a 1D array to a list
                HOG = np.load('Final/HOG_features/HOG_features_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())
                    print(features)
                    print(HOG[features].flatten().shape)

            else:
                #Open the corresponding extracted features and save it as a 1D array to a list
                HOG = np.load('Final/HOG_features/HOG_features_%s_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal, str(induced_noise)), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())

        #If extraction method is given as LBP..
        elif extraction_method == 'LBP':
            if single_model == True:
                try:
                    #Open the LBP features given the feature and the photonumber
                    HOG = np.load('Final/LBP_features/single_model/LBP_features_%s_%s_%s.npy' % (percentage, photonumber, animal), allow_pickle = False)
                    if features == None:
                        #If no features are specified, use all the ROIs
                        all_HOG.append(HOG[:, :, 0].flatten())
                    else:
                        #Otherwise, only save the corresponding extracted features
                        all_HOG.append(HOG[features, :, 0].flatten())
                        print(HOG[features, :, 0].flatten().shape)
                except:
                    all_HOG.append(np.zeros((len(features), 10)).flatten())
                    print('except')
                    print(np.zeros(len(features) * 10).shape)


            #If no induced noise is given..
            elif induced_noise == 0:
                #Open the LBP features given the feature and the photonumber
                HOG = np.load('Final/LBP_features/LBP_features_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the ROIs
                    all_HOG.append(HOG[:, :, 0].flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features, :, 0].flatten())
                    print(features)
                    print(HOG[features, :, 0].shape)

            else:
                #Open the LBP features given the feature and the photonumber
                HOG = np.load('Final/LBP_features/LBP_features_%s_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal, str(induced_noise)), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the ROIs
                    all_HOG.append(HOG[:, :, 0].flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features, :, 0].flatten())

        # #If extraction method is given as SIFT..
        elif extraction_method == 'SIFT':
            if single_model == True:
                try:
                    #open the extracted SIFT features
                    HOG = np.load('Final/SIFT_features/single_model/SIFT_features_%s_%s_%s.npy' % (percentage, photonumber, animal), allow_pickle = False)
                    if features == None:
                        #If no features are specified, use all the  ROIs
                        all_HOG.append(HOG.flatten())
                    else:
                        #Otherwise, only save the corresponding extracted features
                        all_HOG.append(HOG[features].flatten())
                        print(HOG[features].flatten().shape)
                except:
                    print('except')
                    all_HOG.append(np.zeros(len(features) * 128))
                    print(np.zeros(len(features) * 128).shape)

            #If no induced noise is given
            elif induced_noise == 0:

                print('opening')
                #open the extracted SIFT features
                HOG = np.load('Final/SIFT_features/SIFT_features_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the  ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())
                    print(features)
                    print(HOG[features].shape)

            else:
                #open the extracted SIFT features
                HOG = np.load('Final/SIFT_features/SIFT_features_%s_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal, str(induced_noise)), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the  ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())

        #If the extraction method is given as CNN..
        elif extraction_method == 'CNN':

            if single_model == True:
                print(str_head_pose, features)
                try:
                    #Load the extracted CNN features
                    HOG = np.load('Final/CNN_features/single_model/CNN_features_%s_%s_%s.npy' % (percentage, photonumber, animal), allow_pickle = False)
                    if features == None:
                        #If no features are specified, use all the  ROIs
                        all_HOG.append(HOG.flatten())
                    else:
                        #Otherwise, only save the corresponding extracted features
                        all_HOG.append(HOG[features].flatten())
                    print(HOG[features].flatten().shape)
                except:
                    print('except')
                    all_HOG.append(np.zeros(len(features) * 4096))
                    print(np.zeros((len(features) * 4096)).shape)

            #if no induced noise is given..
            elif induced_noise == 0:
                #Load the extracted CNN features
                HOG = np.load('Final/CNN_features/CNN_features_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the  ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())
                    print(features)
                    print(HOG[features].shape)

            else:
                #Load the extracted CNN features
                HOG = np.load('Final/CNN_features/CNN_features_%s_%s_%s_%s_%s.npy' % (str_head_pose, percentage, photonumber, animal, str(induced_noise)), allow_pickle = False)
                if features == None:
                    #If no features are specified, use all the  ROIs
                    all_HOG.append(HOG.flatten())
                else:
                    #Otherwise, only save the corresponding extracted features
                    all_HOG.append(HOG[features].flatten())

        else:
            # if extraction_method != 'SIFT':
            all_HOG = list(itertools.chain.from_iterable(all_HOG))
            #Just append all the features to the list
            X_data.append(all_HOG)

        #set old_photonumber to the current photonumber
        old_photonumber = photonumber

    data = np.array(X_data)
    print(data.shape)

    #Return the big array and the list of keypoint numbers.
    return data
