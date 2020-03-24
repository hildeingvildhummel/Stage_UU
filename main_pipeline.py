import pandas as pd
from train_test_val_split import make_train_test_split
from k_means import try_kmeans, augmentation, kmeans_one_headpose
from HOG import get_extraction_features
from horse_classifier import predict_total_score, SVM_classifier, results_combined
from fusion import fuse_results, find_wrong_horses
from annotate_data import data_annotation
from yolo_evaluation import get_test_iou, plot_yolo_results
from donkey_classifier import donkey_svm_classifier
import numpy as np
from noise_vs_performance import get_raw_results_noise_induced, plot_noise_vs_performance


#open the landmark information file over head pose positions and missing landmarks
landmark_info = pd.read_excel('ALL_POSITIONS.xlsx')
right_tilted = landmark_info.loc[landmark_info.iloc[:, 1] == 'right tilted']
#filter only the ones which are labeled as left_tilted
left_tilted = landmark_info.loc[landmark_info.iloc[:, 1] == 'left tilted']
#combine the tilted landmarks
tilted = right_tilted.append(left_tilted)
right_side = landmark_info.loc[landmark_info.iloc[:, 1] == 'right side']
#filter only the ones which are labeled as left_side
left_side = landmark_info.loc[landmark_info.iloc[:, 1] == 'left side']
#combine the side landmarks
side = right_side.append(left_side)
#filter only the ones which are labeled as front
front = landmark_info.loc[landmark_info.iloc[:, 1] == 'front']
#
# open the excel file containing the scores given by an expert
scores = pd.read_excel('Thijs_horse_and_donkey.xlsx')

"""Manual landmarking"""
#Running this function will show every image in the dataset and the coordinates of a mouse click will be saved
data_annotation('all_landmarks_together/')

"""Horse pain Classification"""
#make the train test split per head pose ensuring none of the same horses are in both the test set and train set
make_train_test_split(scores, front, side, tilted, animal = 'horse')

#Open the train scores
train_front = pd.read_csv('Final/split/front_train_horse.csv')
train_side = pd.read_csv('Final/split/side_train_horse.csv')
train_tilted = pd.read_csv('Final/split/tilted_train_horse.csv')


# Loop over all the head poses
for head_pose_score, str_head_pose in [(train_front, 'front'), (train_side, 'side'), (train_tilted, 'tilted')]:
    #filter the missing landmarks per head pose
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    # augmentate the landmarks before alignment by adding noise to the landmarks (max 2% and max 6%)
    augmentation(head_pose, str_head_pose)
    #create a mean shape based on the head pose and animal given
    try_kmeans(head_pose, str_head_pose, animal='horse')
#
# after aligning the image to the mean shape extract all the features (SIFT, LBP, VGG16, HOG)
# iterate over all the head poses of the training set
for head_pose_score, str_head_pose in [(train_front, 'front'), (train_side, 'side'), (train_tilted, 'tilted')]:

    #select only the missing landmarks of a specific head pose
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    #select the features to be extracted given the head pose
    get_extraction_features(head_pose, str_head_pose, 1, 'HOG', single_model = False, animal = 'horse', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'CNN', single_model = False, animal = 'horse', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'LBP', single_model = False, animal = 'horse', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'SIFT', single_model = False, animal = 'horse', split = 'train')

# #Open the test set per head pose
test_front = pd.read_csv('Final/split/front_test_horse.csv')
test_side = pd.read_csv('Final/split/side_test_horse.csv')
test_tilted = pd.read_csv('Final/split/tilted_test_horse.csv')


#Iterate over all the head poses in the test set
for head_pose_score, str_head_pose in [(test_front, 'front'), (test_side, 'side'), (test_tilted, 'tilted')]:

    #Select the photonumbers of a specific head pose in the test set
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]

    #Save the landmarks with no augmentation due to the split='test' is given
    augmentation(head_pose, str_head_pose, split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'HOG', single_model = False, animal = 'horse', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'LBP', single_model = False, animal = 'horse', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'SIFT', single_model = False, animal = 'horse', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'CNN', single_model = False, animal = 'horse', split = 'test')

#Train the hose SVM classifiers per extraction method
SVM_classifier('horse', 'LBP')
SVM_classifier('horse', 'HOG')
SVM_classifier('horse', 'CNN')
SVM_classifier('horse', 'SIFT')

"""Performance with induced noise"""
for head_pose_score, str_head_pose in [(test_front, 'front'), (test_side, 'side'), (test_tilted, 'tilted')]:

    #Select the photonumbers of a specific head pose in the test set
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    #Create the noise induced test sets
    augmentation(head_pose, str_head_pose, split = 'test', induced = True)
    #Extract the noise induced test features
    for percentage in [0.04, 0.08]:
        # Extract SIFT, HOG, LBP and CNN features given the region of interest
        get_extraction_features(head_pose, str_head_pose, 1, 'HOG', single_model = False, animal = 'horse', split = 'test', already_augmentated_percentage = percentage)
        get_extraction_features(head_pose, str_head_pose, 1, 'LBP', single_model = False, animal = 'horse', split = 'test', already_augmentated_percentage = percentage)
        get_extraction_features(head_pose, str_head_pose, 1, 'SIFT', single_model = False, animal = 'horse', split = 'test', already_augmentated_percentage = percentage)
        get_extraction_features(head_pose, str_head_pose, 1, 'CNN', single_model = False, animal = 'horse', split = 'test', already_augmentated_percentage = percentage)
#Generate the noise induced prediction
get_raw_results_noise_induced()
#Plot the results
plot_noise_vs_performance()

"""Single model"""
#Concatenate the scores of all the head poses
train_dataframes = [train_front, train_side, train_tilted]
test_dataframes = [test_front, test_side, test_tilted]
total_train = pd.concat(train_dataframes)
total_test = pd.concat(test_dataframes)
# print(total_test)
test_head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(total_test.iloc[:, 0])]
train_head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(total_train.iloc[:, 0])]

#Create a single mean shape
kmeans_one_headpose(train_head_pose)
#Extract the features of the test set
get_extraction_features(test_head_pose, None, 1, 'HOG', single_model = True, animal = 'horse', split = 'test')
get_extraction_features(test_head_pose, None, 1, 'SIFT', single_model = True, animal = 'horse', split = 'test')
get_extraction_features(test_head_pose, None, 1, 'LBP', single_model = True, animal = 'horse', split = 'test')
get_extraction_features(test_head_pose, None, 1, 'CNN', single_model = True, animal = 'horse', split = 'test')
# Extract the features of the training set
get_extraction_features(train_head_pose, None, 1, 'HOG', single_model = True, animal = 'horse', split = 'train')
get_extraction_features(train_head_pose, None, 1, 'SIFT', single_model = True, animal = 'horse', split = 'train')
get_extraction_features(train_head_pose, None, 1, 'LBP', single_model = True, animal = 'horse', split = 'train')
get_extraction_features(train_head_pose, None, 1, 'CNN', single_model = True, animal = 'horse', split = 'train')

#Train the SVM classifier for the single model per extracted feature
SVM_classifier('horse', 'LBP', single_model = True)
SVM_classifier('horse', 'HOG', single_model = True)
SVM_classifier('horse', 'CNN', single_model = True)
SVM_classifier('horse', 'SIFT', single_model = True)



"""For Donkeys"""
#Split the donkey set into 80% validation adn 20% test
make_train_test_split(scores, None, side, tilted, animal = 'donkey')
#Open the donkey pain scores
train_side_donkey = pd.read_csv('Final/split/side_train_donkey.csv')
train_tilted_donkey = pd.read_csv('Final/split/tilted_train_donkey.csv')

test_side_donkey = pd.read_csv('Final/split/side_test_donkey.csv')
test_tilted_donkey = pd.read_csv('Final/split/tilted_test_donkey.csv')
#Concatenate the scores to create a training and test set
training_side = [train_side, test_side, train_side_donkey]
training_tilted = [train_tilted, test_tilted, train_tilted_donkey]

train_side_total = pd.concat(training_side)
train_tilted_total = pd.concat(training_tilted)

test_side_total = test_side_donkey
test_tilted_total = test_tilted_donkey
#Iterate over all the donkey images and augmentate if training set is given
for head_pose_score, str_head_pose, split in [(train_side_donkey, 'side', 'train'), (train_tilted_donkey, 'tilted', 'train'), (test_side_donkey, 'side', 'test'), (test_tilted_donkey, 'tilted', 'test')]:
    #Select corresponding missing landmarks
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    #Augmentate the donkey dataset if training set is given
    augmentation(head_pose, str_head_pose, split = split)


# Loop over all the head poses except for the front head pose of the training set
for head_pose_score, str_head_pose in [(train_side_total, 'side'), (train_tilted_total, 'tilted')]:
    #filter the missing landmarks per head pose
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    #create a mean shape based on the both the horse and donkey shapes
    try_kmeans(head_pose, str_head_pose, animal='donkey')
    #Extract the features
    get_extraction_features(head_pose, str_head_pose, 1, 'HOG', single_model = False, animal = 'donkey', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'SIFT', single_model = False, animal = 'donkey', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'LBP', single_model = False, animal = 'donkey', split = 'train')
    get_extraction_features(head_pose, str_head_pose, 1, 'CNN', single_model = False, animal = 'donkey', split = 'train')
# Loop over all the head poses except for the front head pose of the test set
for head_pose_score, str_head_pose in [(test_side_total, 'side'), (test_tilted_total, 'tilted')]:
    #filter the missing landmarks per head pose
    head_pose = landmark_info.loc[landmark_info.iloc[:, 0].isin(head_pose_score.iloc[:, 0])]
    #Extract the features of the test set
    get_extraction_features(head_pose, str_head_pose, 1, 'HOG', single_model = False, animal = 'donkey', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'SIFT', single_model = False, animal = 'donkey', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'LBP', single_model = False, animal = 'donkey', split = 'test')
    get_extraction_features(head_pose, str_head_pose, 1, 'CNN', single_model = False, animal = 'donkey', split = 'test')
#Iterate over all head poses
for y_training, y_testing, str_head_pose in [(train_side_total, test_side_donkey, 'side'), (train_tilted_total, test_tilted_donkey, 'tilted')]:
    #Create a list of all the feature extractio methods
    extraction_list = ['HOG', 'LBP', 'SIFT', 'CNN']
    #Iterate over all feature extraction methods
    for extraction_method in extraction_list:
        #Train the donkey classifier
        donkey_svm_classifier(extraction_method, y_training, y_testing, str_head_pose, animal = 'donkey')



"""YOLO output"""
#Extract the results of the YOLO
get_test_iou()
#Plot the results in a bar plot
plot_yolo_results()

"""Fuse Results Horses"""
#Fuse the horse model predictions by a simple fusion and a simple weighted fusion
fuse_results()

"""Compute Total Score"""
# Train a linear model to predict the total score
predict_total_score()

"""Combine the Results"""
#After running pipeline.py this function creates an overview of the results
results_combined()

"""Analysis of horse pain prediction"""
#This function finds the wrongly classified horses and compares it to the scores given by all 3 experts
find_wrong_horses()
