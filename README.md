Automatic Pain Assesment in Equines 

** General Description *** 

This GitHub only contains the code used to for the master thesis to perform the described analysis. The dependent files are not given. 


ALL_POSTIONS, an excel file containing all the photonumbers together with their head pose and missing landmarks.

annotate_data.py, used for manual landmarking

back_ground_subtraction.py, used for background removal 

CNN_feature_extraction.py, extracts the VGG16 features 

create_pts_file.py, not written by myself. A code which is needed to complete the pipeline

donkey_classifier.py, trains the donkey pain detection 

fusion.py, trains a weighted fusion model together with applying a simple fusion 

get_data.py, not written by myself. A code which is needed to complete the pipeline

HOG.py, extracts the HOG features and contains the functions to extract all feature types from the images

horse_classifier.py, trains the horse pain detection models 

intra_rel.py, visualizes the dataset in multiple ways

k_means.py, augmentates the training images and creates the consensus shape using Generalized Procrustes Analysis

LBP.py, extracts the LBP features after reducing the resolution

main_pipeline.py, an overview of the functions given to train the models

noise_vs_performance.py, applies the analysis of the sensitivity of the proposed pipeline to noise 

PA.py, contains all the functions needed to apply the alignment based on Procrustes

pipeline.py, combines the complete pipeline with automatic head pose estimation and landmarking 

pose_classifier.py, not written by myself. A code which is needed to complete the pipeline

SIFT.py, extracts the SIFT features based on the landmarks 

SVM.py, performs the actual alignment and creates the training and test set based on the feature extraction method type 

train_test_val_split.py, creates the training-test split and the split of the training-validation set during cross-validation

yolo_evaluation.py, performs the evaluation of the YOLOV3 model. 
