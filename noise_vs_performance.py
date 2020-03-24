import numpy as np
import pandas as pd
from SVM import create_HOG_dataframe_with_induced_noise
import matplotlib.pyplot as plt
import statistics
import pickle
import os
from horse_classifier import mse_classifictaion

def get_raw_results_noise_induced():
    """This function Combines all the predictions made by the models trained on all the feature types extracted into a single dataframe.
    It opens the predictions of both the test and training set. So, the models need to be trained and saved already and tested. If not already done,
    do this before running this function.
    Output:
    Saving the complete dataframe as a csv file per head pose. The column names are given by the feature type extracted, the pain feature predicted and the level of induced noise by the test set."""

    #A list of all the pain features
    expression_list =  ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera', 'Corners of the mouth', 'Nostrils']
    #A list of all the head poses
    head_pose = ['front', 'tilted', 'side']
    #Create an empty dataframe
    score_f1 = pd.DataFrame(columns = ['extraction method'])
    score_f1['extraction method'] = ['MSE']
    #Iterate over all the head poses
    for str_head_pose in head_pose:
        #Iterate over all the noise induced levels
        for noise in [0, 0.04, 0.08]:
            if str_head_pose == 'front':
                #exclude the mouth pain feature
                features = [2, 3, 4, 5, 7]

            #otherwise, predict all pain features
            else:
                features = [2, 3, 4, 5, 6, 7]
            #Iterate over all the feature extraction methods
            for extraction_method in ['CNN', 'HOG', 'LBP', 'SIFT']:
                #Open the true test pain scores
                y_testing = pd.read_csv('Final/split/%s_test_horse.csv' % (str_head_pose))
                #Iterate over the pain features
                for predict_feature_index in features:
                    #Select the corresponding ROI index per pain feature, if SIFT is given, select the corresponding number of landmarks in the given ROI
                    if predict_feature_index == 2:
                        if str_head_pose == 'side':
                            if extraction_method == 'SIFT':
                                roi_features = list(range(6))
                            else:
                                roi_features = [0]
                        else:
                            if extraction_method == 'SIFT':
                                roi_features = list(range(10))
                            else:
                                roi_features = [0,1]
                    elif predict_feature_index == 3 or predict_feature_index == 4 or predict_feature_index == 5:
                        if str_head_pose == 'front' or str_head_pose == None:
                            if extraction_method == 'SIFT':
                                roi_features = [10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27]
                            else:
                                roi_features = [2, 4]
                        elif str_head_pose == 'side':
                            if extraction_method == 'SIFT':
                                roi_features = list(range(6, 12))
                            else:
                                roi_features = [1]
                        else:
                            if extraction_method == 'SIFT':
                                roi_features = list(range(10, 16))
                            else:
                                roi_features = [2]

                    elif predict_feature_index == 6:
                        if str_head_pose == 'side':
                            if extraction_method == 'SIFT':
                                roi_features = list(range(18, 22))
                            else:
                                roi_features = [3]
                        else:
                            if extraction_method == 'SIFT':
                                roi_features = list(range(22, 25))
                            else:
                                roi_features = [4]

                    elif predict_feature_index == 7:
                        if str_head_pose == 'front' or str_head_pose == None:
                            if extraction_method == 'SIFT':
                                roi_features = [16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33]
                            else:
                                roi_features = [3, 5]
                        elif str_head_pose == 'side':
                            if extraction_method == 'SIFT':
                                roi_features = list(range(12, 18))
                            else:
                                roi_features = [2]
                        else:
                            if extraction_method == 'SIFT':
                                roi_features = list(range(16, 22))
                            else:
                                roi_features = [3]
                    #Create the test set given the extraction method, the head pose and the level of induced noise
                    X_non_augmentated = create_HOG_dataframe_with_induced_noise(y_testing['photonumber'], str_head_pose, roi_features, extraction_method = extraction_method, induced_noise = noise, single_model = False, animal = 'horse', split = 'test')
                    #Save the missing values in the test file as a variable
                    index_test = y_testing.iloc[:, predict_feature_index].index[y_testing.iloc[:, predict_feature_index].apply(np.isnan)]
                    print(len(index_test))
                    print('pre: ', len(y_testing))
                    #Select the pain feature we would like to predict in the test set
                    y_test = y_testing.iloc[:, predict_feature_index]
                    #Drop the missing values
                    y_test = y_test.dropna(axis=0)
                    print('post y : ', len(y_test), 'pre: ', X_non_augmentated.shape)
                    #Remove the extracted features of the missing values in the scores
                    X_test = np.delete(X_non_augmentated, index_test, 0)
                    print(X_test.shape)
                    #Convert test set to numpy
                    X_test = np.float64(X_test)
                    #Load the already trained model
                    svregressor = pickle.load(open('Final/models/SVC_%s_%s_%s_horse.sav' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method), 'rb'))
                    #Predict the test set
                    prediction = svregressor.predict(X_test)
                    #Create column name based on the extraction method, the level of induced noise and the pain feature
                    column_name = extraction_method + '_' + str(noise) + '_' + expression_list[predict_feature_index - 2]
                    print(column_name)
                    #Initiate an empty list
                    final_y_pred_regres = []
                    #loop over all the predictions
                    for i in prediction:
                        #create an empty list to save the distance of the predicted value with all possible values
                        distance_list = []
                        #calculate each possible distance of the predicted value to a possible outcome
                        distance_list.append(abs(0 - i))
                        distance_list.append(abs(1 - i))
                        distance_list.append(abs(2 - i))
                        #save the index of the lowest distance to the list of the final predictions
                        final_y_pred_regres.append(distance_list.index(min(distance_list)))
                    #Calculate the binarized MSE
                    f1 = mse_classifictaion(y_test, final_y_pred_regres)
                    print(f1)
                    #Add this error to its corresponding column
                    score_f1[column_name] = [f1]
                    print(score_f1)
            #check directory already exists, if not create it
            if os.path.isdir('Final/noise_performance') == False:
                os.makedirs('Final/noise_performance')
            # outcomes.to_csv('workshop_results/noise_vs_performance/%s_raw_prediction.csv' % (str_head_pose), index=False)
            score_f1.to_csv('Final/noise_performance/%s_horse.csv' % (str_head_pose), index=False)

def plot_noise_vs_performance():
    """This function makes the error level of the prediction of the test set with different levels of induced noise. This function creates
    a bar plot per given head pose showing the influence of the noisy test set on the performance of the model. The model is only shown, not
    automatically saved. Before this plot can be made, run the function above to create the right files. """
    #Open the total dataframe of the performance per head pose
    tilted = pd.read_csv('Final/noise_performance/tilted_horse.csv')
    front = pd.read_csv('Final/noise_performance/front_horse.csv')
    side = pd.read_csv('Final/noise_performance/side_horse.csv')

    #Select the extraction methods to plot
    extraction_methods = ['HOG', 'SIFT', 'CNN']
    #Select the head poses
    head_poses = [tilted, side, front]
    #Iterate over the head poses
    for head_pose in head_poses:
        print(head_pose.columns)
        #Create empty lists per induced noise level
        no_noise = []
        noise_4 = []
        noise_8 = []
        #Iterate over the extraction methods
        for extraction_method in extraction_methods:
            #Iterate over the columns of the big dataframe
            for column in head_pose.columns:

                #Define the name of the column by looking at the extraction method and the level of induced noise
                #Add the error to the corresponding list
                if column.startswith(extraction_method):
                    print(column)
                    if column.startswith(extraction_method + '_0_'):
                        no_noise.append(head_pose[column].values)

                    if column.startswith(extraction_method + '_0.04'):

                        noise_4.append(head_pose[column].values)
                    if column.startswith(extraction_method + '_0.08'):

                        noise_8.append(head_pose[column].values)
            #Calculate the mean per induced noise lists
            mean_0 = statistics.mean(np.concatenate(no_noise, axis = 0))
            mean_4 = statistics.mean(np.concatenate(noise_4, axis =0))
            mean_8 = statistics.mean(np.concatenate(noise_8, axis = 0))
            #Calculate the standard deviation per induced noise lists
            std_0 = statistics.stdev(np.concatenate(no_noise, axis=0))
            std_4 = statistics.stdev(np.concatenate(noise_4, axis = 0))
            std_8 = statistics.stdev(np.concatenate(noise_8, axis = 0))
            #Rename the means and standard deviations by the extraction method
            if extraction_method == 'HOG':
                HOG = [mean_0, mean_4, mean_8]
                HOG_std = [std_0, std_4, std_8]

            elif extraction_method == 'SIFT':
                SIFT = [mean_0, mean_4, mean_8]
                SIFT_std = [std_0, std_4, std_8]
            elif extraction_method == 'CNN':
                CNN = [mean_0, mean_4, mean_8]
                CNN_std = [std_0, std_4, std_8]
        #Select the noise levels
        noise = [0, 0.04, 0.08]
        #Determine the number of unique extraction methods
        n_groups = 3

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.8
        #Plot the HOG performance
        rects1 = plt.bar(index, HOG, bar_width,
        alpha=opacity,
        color='b',
        label='HOG')
        #Plot the SIFT performance
        rects2 = plt.bar(index + bar_width, SIFT, bar_width,
        alpha=opacity,
        color='r',
        label='SIFT')
        #Plot the CNN performance
        rects2 = plt.bar(index + bar_width + bar_width, CNN, bar_width,
        alpha=opacity,
        color='y',
        label='CNN')
        #Plot the plot
        plt.xlabel('Noise')
        plt.ylabel('mean MSE')
        # plt.title('Score')
        plt.xticks(index + bar_width, ('0%', '4%','8%'))
        # plt.ylim((0, 1))
        #
        plt.legend()

        plt.tight_layout()

        plt.show()
        #Print the standard deviations per extraction method
        print('HOG: ', HOG_std)
        print('SIFT: ', SIFT_std)
        print('CNN: ', CNN_std)
