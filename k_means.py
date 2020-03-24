import numpy as np
import pandas as pd
import random
from PA import apply_procrustes_transform
from PA import generalized_procrustes_analysis
from PA import procrustes
import math
import cv2
from PA import open_landmark_txt
import os

def augmentation(head_pose, str_head_pose, split = 'train', induced = None):
    """This function augmentates the landmarks by adding noise (2% and 6% of the normalization distance). It saves the landmarks as a numpy file.
    Input:
    - head_pose: The excel file containing the photonumbers and the missing landmarks
    - str_head_pose: The head pose of the animals in the image, could either be tilted, front or side
    - split: if the train or test set is given. Once the test set  is given, no noise will be added to the landmarks. Choose between train or test (in string format), default is train
    - induced: If induced is true, add noise to the test set. Which is either 4% or 8% of the normalization distance
    Output:
    - splited_standard: an array containing all the augmentated landmarks together
    - index: list containing all the photonumbers which are augmented """
    #Select the photonubers
    photonumbers = head_pose.iloc[:, 0]

    #Create empty lists
    splited_standard = []
    index = []
    #Select augmentation ratios for both train and test set
    if split == 'train':
        percentages = [0, 0.02, 0.06]
    else:
        percentages = [0]
    #If induced is set to true, set ratios to 0, 0.04 and 0.08
    if induced == True:
        percentages = [0, 0.04, 0.08]
    #iterate over the number of Images
    for i in photonumbers:
        print(i)
        #open the image
        image = cv2.imread('all_images_together/%s.jpg' % (str(i)))




        #open the landmark file of a given image and concatenate all the landmarks of a given head pose
        landmarks = open_landmark_txt('all_landmarks_together/landmarks_%s.txt' % (str(i)))
        #If the tilted head pose is given..
        if len(landmarks) == 44:
            #select the eye_center
            eye_center = [(landmarks[13][0] + landmarks[10][0])/2, (landmarks[12][1] + landmarks[15][1])/2]
            #select the nose center
            nose_center = [(landmarks[18][0] + landmarks[16][0])/2, (landmarks[20][1] + landmarks[17][1])/2]
            #calculate the distance
            yard_stick = list(np.array(eye_center) - np.array(nose_center))
            #make all the distance absolute
            yard_stick = [abs(i) for i in yard_stick]
            yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
            for perc in percentages:
                #create list to save the augmentated landmarks to
                augmentated_points = []
                #get the true value for the selected allowable noise for the landmarks
                allowable_noise = perc * yard_stick_distance
                #iterate over all the landmarks in the landmark set
                for point in landmarks:
                    try:
                        #select a random distance within the allowable range for x
                        random_distance_x = random.uniform(-allowable_noise, allowable_noise)
                        #select a random distance within the allowable range for y
                        random_distance_y = random.uniform(-allowable_noise, allowable_noise)
                        #add the random distance to x
                    except:
                        random_distance_x = 0
                        random_distance_y = 0
                    #Add the random noise the the true x coordinate
                    new_point_x = point[0] + random_distance_x
                    if new_point_x > image.shape[1]:
                        #If the new point exceeds the boundary of the image, select the ground truth landmark
                        new_point_x = point[0]
                    #add the random distance to y
                    new_point_y = point[1] + random_distance_y
                    if new_point_y > image.shape[0]:
                        new_point_y = point[1]
                    #save the noisy landmarks to the list
                    augmentated_points.append([new_point_x, new_point_y])
                #save the photonumber to the list
                index.append(i)
                #Make an array out of the landmarks
                augmentated_points = np.array(augmentated_points)
                #Save the files
                if induced == True:
                    #check directory already exists, if not create it
                    if os.path.isdir('Final/noisy_landmarks/%s' % (str(perc))) == False:
                        os.makedirs('Final/noisy_landmarks/%s' % (str(perc)))
                    np.save('Final/noisy_landmarks/%s/noisy_landmarks_%s_%s' % (str(perc), str(i), str(perc)), augmentated_points)

                else:
                    if os.path.isdir('Final/noisy_landmarks') == False:
                        os.makedirs('Final/noisy_landmarks')
                    np.save('Final/noisy_landmarks/noisy_landmarks_%s_%s' % (str(i), str(perc)), augmentated_points)
                #Save the augmentated landmarks to a list
                splited_standard.append(augmentated_points)

        #If the front head pose is given..
        elif len(landmarks) == 54:
            #Calculate the mean normalization distance from the both eye-nostril distances in the face.
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
            for perc in percentages:
                #create list to save the augmentated landmarks to
                augmentated_points = []
                #get the true value for the selected allowable noise for the landmarks
                allowable_noise = perc * yard_stick_distance
                #iterate over all the landmarks in the landmark set
                for point in landmarks:
                    try:
                        #select a random distance within the allowable range for x
                        random_distance_x = random.uniform(-allowable_noise, allowable_noise)
                        #select a random distance within the allowable range for y
                        random_distance_y = random.uniform(-allowable_noise, allowable_noise)
                        #add the random distance to x
                    except:
                        random_distance_x = 0
                        random_distance_y = 0
                    new_point_x = point[0] + random_distance_x
                    if new_point_x > image.shape[1]:
                        new_point_x = point[0]
                    #add the random distance to x
                    new_point_y = point[1] + random_distance_y
                    if new_point_y > image.shape[0]:
                        new_point_y = point[1]
                    #save the noisy landmarks to the list
                    augmentated_points.append([new_point_x, new_point_y])
                index.append(i)
                augmentated_points = np.array(augmentated_points)
                #Save the landmarks to a numpy file
                if induced == True:
                    #check directory already exists, if not create it
                    if os.path.isdir('Final/noisy_landmarks/%s' % (str(perc))) == False:
                        os.makedirs('Final/noisy_landmarks/%s' % (str(perc)))
                    np.save('Final/noisy_landmarks/%s/noisy_landmarks_%s_%s' % (str(perc), str(i), str(perc)), augmentated_points)
                else:
                    if os.path.isdir('Final/noisy_landmarks') == False:
                        os.makedirs('Final/noisy_landmarks')
                    np.save('Final/noisy_landmarks/noisy_landmarks_%s_%s' % (str(i), str(perc)), augmentated_points)
                #Save the landmarks to the list
                splited_standard.append(augmentated_points)

        #If the side head pose is given..
        elif len(landmarks) == 45:
            #Calculate the normalization distance based on the eye and nostril centers
            eye_center = [(landmarks[22][0] + landmarks[25][0])/2, (landmarks[27][1] + landmarks[24][1])/2]
            nose_center = [(landmarks[17][0] + landmarks[20][0])/2, (landmarks[16][1] + landmarks[18][1])/2]
            yard_stick = list(np.array(eye_center) - np.array(nose_center))
            yard_stick = [abs(i) for i in yard_stick]
            yard_stick_distance = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)
            for perc in percentages:
                #create list to save the augmentated landmarks to
                augmentated_points = []
                #get the true value for the selected allowable noise for the landmarks
                allowable_noise = perc * yard_stick_distance
                #iterate over all the landmarks in the landmark set
                for point in landmarks:
                    try:
                        #select a random distance within the allowable range for x
                        random_distance_x = random.uniform(-allowable_noise, allowable_noise)
                        #select a random distance within the allowable range for y
                        random_distance_y = random.uniform(-allowable_noise, allowable_noise)

                    except:
                        random_distance_x = 0
                        random_distance_y = 0
                    #add the random distance to x
                    new_point_x = point[0] + random_distance_x
                    if new_point_x > image.shape[1]:
                        #If the new point exceeds the image boundary, select the ground truth x-coordinate
                        new_point_x = point[0]
                    #add the random distance to y
                    new_point_y = point[1] + random_distance_y
                    if new_point_y > image.shape[0]:
                        #If the new point exceeds the image boundary, select the ground truth y-coordinate
                        new_point_y = point[1]
                    #save the noisy landmarks to the list
                    augmentated_points.append([new_point_x, new_point_y])
                #Save the photonumber to a list
                index.append(i)
                #make an array out of the augmented landmarks
                augmentated_points = np.array(augmentated_points)
                #Save the augmented landmarks as a numpy file
                if induced == True:
                    #check directory already exists, if not create it
                    if os.path.isdir('Final/noisy_landmarks/%s' % (str(perc))) == False:
                        os.makedirs('Final/noisy_landmarks/%s' % (str(perc)))
                    np.save('Final/noisy_landmarks/%s/noisy_landmarks_%s_%s' % (str(perc), str(i), str(perc)), augmentated_points)
                else:
                    if os.path.isdir('Final/noisy_landmarks') == False:
                        os.makedirs('Final/noisy_landmarks')
                    np.save('Final/noisy_landmarks/noisy_landmarks_%s_%s' % (str(i), str(perc)), augmentated_points)
                #Add the augmented landmarks to a list
                splited_standard.append(augmentated_points)

        #Check if path already exists, if not create it
        if os.path.isdir('Final/indices_horse') == False:
            os.makedirs('Final/indices_horse')
        #Save the photonumbers to a text file
        with open('Final/indices_horse/indices_noisy_%s.txt' % (str_head_pose), 'w') as f:
            for item in index:
                f.write("%s\n" % item)

    #return the augmented landmarks and the list of photonumbers
    return splited_standard, index

def scaling(head_pose, str_head_pose, save = False, single_model = False, already_augmentated_percentage = None, animal =None, split = 'train'):
    """This function creates one big 3D numpy array from all the landmarks given a head pose.
    Input:
    - head_pose: Excel file containing the head pose, photonumber and the missing landmarks
    - str_head_pose: The head pose of the images in string format could either be tilted, front or side
    - save: if set to True, the photonumbers of the created dataframe will be saved to a txt file
    - single_model: if set to True, the dataset will not be split by head pose but all the head poses will be used
    - already_augmentated_percentage: If set to a float, it will use this induced noise landmarks as the basis of the pipeline
    - animal: The animal in the picture to predict the pain score, could either be horse or donkey or None
    - split: define if the training or test set is given, default is the training set. Could either be train or test
    Output:
    - splited_standard: Is the 3D numpy array containing all the landmarks
    - indices: a list containing the corresponding photonubers of the created array"""
    #create an empty list to save the photonumbers to
    indices = []
    #create an empty list to save the landmarks to
    splited_standard = []
    #save the photonumbers given the head pose
    photonumbers = head_pose.iloc[:, 0]
    #if the training set is given, also take into account the augmented landmarks
    if split == 'train':
        percentages = [0, 0.02, 0.06]

    else:
        #Select the noise induced set
        if already_augmentated_percentage != None:
            percentages = [already_augmentated_percentage]
        else:
            #else only take the non-augmented landmarks
            percentages = [0]
    #iterate over the photonumbers
    for i in photonumbers:
        # iterate over the percentages of augmentation
        for percentage in percentages:
            #if it is already augmentated, open the augmentated landmarks
            if already_augmentated_percentage != None:
                landmarking = np.load('Final/noisy_landmarks/%s/noisy_landmarks_%s_%s.npy' % (str(already_augmentated_percentage), str(i), str(percentage)))
                # landmarking =  open_landmark_txt('noisy_landmarks/%s/noisy_landmarks_%s_%s.txt' % (str(already_augmentated_percentage), str(i), str(percentage)))
            #else open the noisy landmarks
            else:
                landmarking = np.load('Final/noisy_landmarks/noisy_landmarks_' + str(i) + '_' + str(percentage) + '.npy')

                # landmarking = open_landmark_txt('noisy_landmarks/only_training/noisy_landmarks_' + str(i) + '_' + str(percentage) + '.txt')
            #if the number of landmarks is 44..
            if len(landmarking) == 44:
                #if a single model is given...
                if single_model == True:
                    #append only the eye, nose and ear and save the photonumber
                    splited_standard.append(landmarking[5:22])
                    indices.append(i)
                # if no single model is given..
                else:
                    #append all the landmarks to the list and save the photonumber
                    splited_standard.append(landmarking)
                    indices.append(i)

            #same for if number of landmarks is 54..
            elif len(landmarking) == 54:
                if single_model == True:
                    splited_standard.append(landmarking[5:22])
                    indices.append(i)
                else:
                    splited_standard.append(landmarking)
                    indices.append(i)
            #same for if number of landmarks is 45..
            elif len(landmarking) == 45:
                if single_model == True:
                    landmark_list = np.concatenate((landmarking[0:5],landmarking[22:28], landmarking[16:22]), axis = 0)
                    splited_standard.append(landmark_list)
                    indices.append(i)
                else:
                    splited_standard.append(landmarking)
                    indices.append(i)
        #if save is given..
        if single_model == False and save != False:
            #save the photonumbers to a txt file
            if animal != None:
                if os.path.isdir('indices_horse') == False:
                    os.makedirs('indices_horse')
                #if animal is given save with animal name
                with open('indices_horse/indices_noisy_%s_%s_%s.txt' % (str_head_pose, animal, split), 'w') as f:
                    for item in indices:
                        f.write("%s\n" % item)
            else:
                if os.path.isdir('indices_horse') == False:
                    os.makedirs('indices_horse')
                #else, save without animal name
                with open('indices_horse/indices_noisy_%s.txt' % (str_head_pose), 'w') as f:
                    for item in indices:
                        f.write("%s\n" % item)
        #if single model is given..
        elif single_model == True:
            if os.path.isdir('indices_horse') == False:
                os.makedirs('indices_horse')
            #save the photonumbers to a txt file
            with open('indices_horse/indices_single_model.txt', 'w') as f:
                for item in indices:
                    f.write("%s\n" % item)
    #return the combined landmarks and the photonumbers
    return splited_standard, indices

def find_clusters(X, n_clusters, pose, indices,rseed=2, augmentate = False):
    """https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    This function finds the cluster seperations.
    Input:
    - X: one big array containing all the shapes
    - n_clusters: the number of clusters, in our case always set to 1
    - pose: the excel file containing the information of the head position (photonumber and missing landmarks)
    Output:
    - centers: The consensus shape per cluster
    - labels: A list of the assigned cluster per sample
    - error: The alignment error of the GPA given in RSS
     """

    # 1. Randomly choose cluster centers
    #The clusteer center in this case is a random selected shape
    rng = np.random.RandomState(rseed)
    # i = rng.permutation(X)[:n_clusters]
    i = rng.permutation(X)[:n_clusters]
    centers = i
    old_distance = 100000000
    counter = 0

    while True:
        counter += 1
        # Create an empty list to append the labels (in this case the number of a cluster)
        labels = []
        #create an empty list to append the indexes of files with missing values
        filtered_labels = []
        #add a counter
        n = -1
        #loop over all shapes given an head pose
        if augmentate == True:
            pose = pose.loc[pose.index.repeat(5)]

        for matrix in X:
            #count the index of the shapes
            n += 1
            photonumber = indices[n]
            missing = pose.loc[pose.iloc[:, 0] == int(photonumber)]
            missing = missing.dropna()

            #get the information per matrix from the excel file
            # missing = pose.iloc[n, 2:].dropna()
            #if there are missing values intered:
            if len(missing) > 0:
                missing = missing.values
                flat_list = [item for sublist in missing for item in sublist]
                missing = flat_list[2:]

                #add the index of the shape with missing values to the list
                filtered_labels.append(n)

                #create a list of integers out of the missing landmarks. It is -1 because the landmark numbers start with 1 isntead of 0
                missing_landmarks_columns = list(missing)

                missing_landmarks = [int(i) - 1 for i in missing]
                filtered_matrix = np.delete(matrix, missing_landmarks, axis = 0)
                #create an empty list to append all the generated SS error of a shape to all the centers
                distances = []
                #loop over all the centers
                for center in centers:
                    filtered_center = np.delete(center, missing_landmarks, axis=0)
                    #Perform procrustes analysis to determine the SS error of a shape to a given center
                    d,Z,Tform = procrustes(filtered_center, filtered_matrix)
                    #append the SS error to the list
                    distances.append(d)
                    #return the index of the lowest error and save this to the labels list
                labels.append(distances.index(min(distances)))
            else:
                #create an empty list to append all the generated SS error of a shape to all the centers
                distances = []
                #loop over all the centers
                for center in centers:
                    #Perform procrustes analysis to determine the SS error of a shape to a given center
                    d,Z,Tform = procrustes(center, matrix)
                    #append the SS error to the list
                    distances.append(d)
                #return the index of the lowest error and save this to the labels list
                labels.append(distances.index(min(distances)))
        #create an empty list to append the newly generated centers to
        new_centers = []
        #loop over the number of clusters
        for j in range(0, len(centers)):
            #return the index values of labels given a label (cluster number)
            indexes = [i for i, value in enumerate(labels) if value == j]
            #Create an empty list to append the shapes in a cluster
            shapes_per_cluster = []
            list_missing_values = []
                #loop over the indexes in a cluster
            for index in indexes:
                shapes_per_cluster.append(X[index])
                if index in filtered_labels:
                    #return the missing values into a list
                    missing = pose.loc[pose.iloc[:, 0] == int(photonumber)]
                    missing = missing.dropna()
                    # missing = list(pose.iloc[index, 2:].dropna())
                    missing = missing.values
                    flat_list = [item for sublist in missing for item in sublist]
                    missing = flat_list[2:]
                    # missing_landmarks = [int(i) - 1 for i in missing]
                    list_missing_values.append(missing_landmarks)
                else:
                    #append an empty list to the missing value list to keep the indices of both the shapes_per_cluster cluster list as the missing value list the same.
                    list_missing_values.append([])

            #generate a consensus shape based on all the shapes in an assigned cluster
            print('cluster number: ', j)
            mean_shape, new_shapes, distance = generalized_procrustes_analysis(shapes_per_cluster, list_missing_values)
                        #append the consensus shape to the new_centers list to save these shapes and set it to new centers
            new_centers.append(mean_shape)
        #Check if the new_centers are the same as the previously determined center or if the max iteration number is reached
        if np.all(np.array(new_centers) == centers) or counter == 25:
            #If it is converged, stop the while loop
            break

        #assign the newly generated consensus shapes as the new centers for the kmeans clustering
        centers = new_centers
        old_distance = distance
    #set the overall total error of the clustering to 0
    total_dist = 0
    error = []
    number_of_clusters = []
    #loop over the number of clusters
    for j in range(0, len(centers)):
        #return the index values of labels given a label (cluster number)
        indexes = [i for i, value in enumerate(labels) if value == j]
        #Create an empty list to append the shapes in a cluster
        shapes_per_cluster = []
        #set the error per cluster to 0
        dist_per_clust = 0
        #loop over the indexes in a cluster
        for index in indexes:
            #apply the procrustes analysis on all the matrices to the generated cluster mean shape
            tr_Y_pts, M, mean_shape, d = apply_procrustes_transform(X[index], centers[j], len(X[index]))
            #add the error to the overall error
            total_dist += d
            #add the error to the cluster error
            dist_per_clust += d
        #print the errors
        print('error cluster %i: ' % j, dist_per_clust)
    print('total error over all clusters: ', total_dist)
    error.append(total_dist)
    number_of_clusters.append(n_clusters)

    #return the centers and the labels
    return centers, labels, error

def try_kmeans(head_pose, str_head_pose, animal = None, split = 'train'):
    """This function creates a consensus shape based on a single cluster in KMeans.
    Input:
    - head_pose: The information in an excel file containing the photonumers and the missing landmarks
    - str_head_pose: The head pose of the animals in the images in string format. Could either be tilted, side or front
    - animal: The animal in the images, could either be horse or donkey Default is None
    - split: Define if the test or training set is given, Could either be test or train in string format. Default is train
    Output:
    2 text files. One containing the mean shape and the other one containing the labels assigned by KMeans.
    """
    #Create a big array
    trying, indices = scaling(head_pose, str_head_pose, save = True, animal = animal, single_model = False)
    # initialize empty lists to append the cluster number and a list to append the total error of the cluster choice
    number_of_clusters = []

    #find the mean shape using a single cluster
    centers, labels, error = find_clusters(trying, 1, head_pose, indices)
    if os.path.isdir('Final/labels_horses') == False:
        os.makedirs('Final/labels_horses')
    # write the label to a txt file
    if animal != None:
        with open('Final/labels_horses/labels_noisy_%s_clusters_%s_%s_all.txt' % (str(1), str_head_pose, animal), 'w') as f:
            for item in labels:
                f.write("%s\n" % item)
    else:
        with open('Final/labels_horses/labels_noisy_%s_clusters_%s.txt' % (str(1), str_head_pose), 'w') as f:
            for item in labels:
                f.write("%s\n" % item)
    # add a counter
    k = 0
    # loop over the centers
    for center in centers:

        if animal != None:
            np.savetxt('Final/labels_horses/center_noisy_%s_%s_clusters_%s_%s_all.txt' % (k, str(1) , str_head_pose, animal), center)

        else:
            # save the center landmarks to a txt file
            np.savetxt('Final/labels_horses/center_noisy_%s_%s_clusters_%s.txt' % (k, str(1), str_head_pose), center)
        k += 1

    # # close the label file
    f.close()
    #return the number_of_clusters initiated
    return number_of_clusters

def kmeans_one_headpose(head_pose, str_head_pose = None):
    """This function assigns the consensus shape by a KMeans clustering if a single model is given.
    It saves the consensus shape as a text file and also saves the clusters assigned to each sample as a text file.
    Input:
    - head_pose: Excel file containing all the training samples with the photonumber, the head pose and the missing landmarks
    - str_head_pose: Default is None, since the single model deals with all the head poses
    Output:
    - number_of_clusters: the assigned labels per sample """
    #Create one big array out of the landmarks
    trying, indices = scaling(head_pose, str_head_pose, save = True, single_model = True)
    print('trying done')
    # initialize empty lists to append the cluster number and a list to append the total error of the cluster choice
    number_of_clusters = []
    all_error = []


    # find the clusters based on the initialized number of clusters and head positions
    centers, labels, error = find_clusters(trying, 1, head_pose, indices)
    #Check if path exists, if not create it
    if os.path.isdir('Final/labels_horses') == False:
        os.makedirs('Final/labels_horses')
    # write the label to a txt file
    with open('Final/labels_horses/labels_single_model_1_clusters.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
    # add a counter
    k = 0
    # loop over the centers
    for center in centers:
        #Save the centers as a text file
        np.savetxt('Final/labels_horses/center_single_model_%s_1_clusters.txt' % (k), center)
        k += 1
    # append the number of clusters to the list
    number_of_clusters.append(1)
    # append all error of the number of clusters to a list
    all_error.append(error)
    # close the label file
    f.close()

    return number_of_clusters
