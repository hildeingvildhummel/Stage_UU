import pandas as pd
import numpy as np
import shutil
import os
from sklearn.model_selection import GroupShuffleSplit
import random
import itertools



def make_train_test_split(scores, front, side, tilted, animal = None):
    """This function makes the train and test split, having 80% training set and 20% test set. This function ensures none of the same horses/donkeys are in both the
    train set as in the test set. This function saves the score file splitted in both train and test set per head pose as a csv file in the split directory
    Input:
    - scores: The excel file containing all the images with all the pain scores given by the expert
    - front: The excel file of the front head pose with the missing landmarks
    - side: The excel file of the side head pose with the missing landmarks
    - tilted: The excel file of the tilted head pose with the missing landmarks
    - animal: The animal on the images, could either be horse or donkey
    Output:
    - head_pose_test: The test set of the given head pose after the split """
    #make a range of unique random numbers
    n_unique = random.sample(range(1, 4000), 2200)
    #fill the empty ID labels with a unique random number
    scores['paardnr met meerdere foto\'s'] = scores['paardnr met meerdere foto\'s'].fillna(pd.Series(np.random.choice(n_unique, size=len(scores.index))))
    #Select the number of rows based on the animal
    if animal == 'horse':
        scores = scores.iloc[0:1856, :]
        list = [(front, 'front'), (tilted, 'tilted'), (side, 'side')]
    elif animal == 'donkey':
        scores = scores.iloc[1856:, :]
        list = [(tilted, 'tilted'), (side, 'side')]

    #iterate over all the head poses
    for head_pose, str_head_pose in list:
        #Initate empty lists to save the split
        test_index = []
        train_index = []
        #filter the photonumbers given the head pose
        head_pose_scores = scores.loc[scores.iloc[:, 0].isin(head_pose.iloc[:, 0])]
        head_pose_scores = head_pose_scores.fillna(13)
        head_pose = head_pose_scores.iloc[:, 2:8]
        #Filter the unique pain score combinations
        result = head_pose.drop_duplicates()
        #Iterate over all the unique pain score combinations
        for index, row in result.iterrows():
            #Select the rows with this combination
            scoring = head_pose_scores.loc[(head_pose_scores['ears'] == row['ears']) & (head_pose_scores['orbital tightening'] == row['orbital tightening']) & (head_pose_scores['angulated upper eyelid'] == row['angulated upper eyelid']) & (head_pose_scores['sclera'] == row['sclera']) & (head_pose_scores['corners of the mouth'] == row['corners of the mouth']) & (head_pose_scores['nostrils'] == row['nostrils'])]

            try:
                #Split the data based on the ID labels given to the images. 80% training data and 20% test data
                train, test = next(GroupShuffleSplit(test_size = 0.2).split(scoring, groups = scoring['paardnr met meerdere foto\'s']))
                #save the corresponding photoumbers to a list
                test_index.append(scoring.iloc[test, 0].values)
                train_index.append(scoring.iloc[train, 0].values)

            except:
                #If split cannot be made, save everything as training data
                train = range(0, len(scoring))
                train_index.append(scoring.iloc[train, 0].values)
        #Concatenate the training and test splits
        train_index = np.concatenate(train_index).ravel().tolist()
        test_index = np.concatenate(test_index).ravel().tolist()
        #Get the scores corresponding to the photonumbers
        head_pose_test = head_pose_scores[head_pose_scores['photonumber'].isin(test_index)]
        head_pose_train = head_pose_scores[head_pose_scores['photonumber'].isin(train_index)]
        print(test_index)
        print(head_pose_test)

        #check if path already exits, if not make the path
        if os.path.isdir('Final/split') == False:
            os.makedirs('Final/split')
        #save the train and test split of the scores to a csv file.
        if animal != None:
            head_pose_test.to_csv('Final/split/%s_test_%s.csv' % (str_head_pose, animal), index = False)
            head_pose_train.to_csv('Final/split/%s_train_%s.csv' % (str_head_pose, animal), index = False)
        else:
            head_pose_test.to_csv('Final/split/%s_test.csv' % (str_head_pose), index = False)
            head_pose_train.to_csv('Final/split/%s_train.csv' % (str_head_pose), index = False)
    #Return the test set
    return head_pose_test

def training_val_split(train_scores, val_indices):
    """This function splits the training set into a traning and validation set.
    Input:
    - train_scores: Dataframe containing all the pain scores and photonumbers
    - val_indices: List of photonumbers which have already been used in the validation set
    Output:
    - head_pose_test: The validation set
    - head_pose_train: The  training set
    - head_pose_test['photonumber']: The photonumbers of the validation set
    - head_pose_train['photonumber']: The photonumbers of the training set
    """
    #initiate empty lists to save the split to
    test_index = []
    train_index = []


    #filter the photonumbers given the head pose
    head_pose_scores = train_scores.fillna(13)
    head_pose = head_pose_scores.iloc[:, 2:8]
    #If the val_indices is given, replace the ID value
    if len(val_indices) != 0:
        for i in val_indices:
            head_pose_scores.loc[head_pose_scores.photonumber == i, 'paardnr met meerdere foto\'s'] = 100000000000000000
    #Filter only the unique pain score combinations
    result = head_pose.drop_duplicates()
    #Iterate over all the unique pain score combinations
    for index, row in result.iterrows():
        #Select the scores with the corresponding unique pain scores
        scoring = head_pose_scores.loc[(head_pose_scores['ears'] == row['ears']) & (head_pose_scores['orbital tightening'] == row['orbital tightening']) & (head_pose_scores['angulated upper eyelid'] == row['angulated upper eyelid']) & (head_pose_scores['sclera'] == row['sclera']) & (head_pose_scores['corners of the mouth'] == row['corners of the mouth']) & (head_pose_scores['nostrils'] == row['nostrils'])]

        try:
            #Split the data based on the ID labels given to the images. 80% training data and 20% test data
            train, test = next(GroupShuffleSplit(test_size = 0.20).split(scoring, groups = scoring['paardnr met meerdere foto\'s']))
            #save the corresponding photoumbers to a list
            test_index.append(scoring.iloc[test, 0].values)
            train_index.append(scoring.iloc[train, 0].values)

        except:

            #If split cannot be made, save everything as training data
            train = range(0, len(scoring))
            train_index.append(scoring.iloc[train, 0].values)

    #Concatenate all the splits of the training and validation set
    train_index = np.concatenate(train_index).ravel().tolist()
    test_index = np.concatenate(test_index).ravel().tolist()
    #Get the scores corresponding to the photonumbers
    head_pose_test = head_pose_scores[head_pose_scores['photonumber'].isin(test_index)]
    head_pose_train = head_pose_scores[head_pose_scores['photonumber'].isin(train_index)]

    #Retrun the validation set, the training set and the photonumbers of both the validation and training set. 
    return head_pose_test, head_pose_train, head_pose_test['photonumber'], head_pose_train['photonumber']
