import numpy as np
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
import pandas as pd
from SVM import create_HOG_dataframe_with_induced_noise
from pykernels.pykernels.regular import Min, GeneralizedHistogramIntersection
from pykernels.pykernels.basic import Linear
import os
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid, GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import mean_squared_error, f1_score, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from numpy.linalg import norm
import pickle
from sklearn.cluster import KMeans
import json
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn import preprocessing
from train_test_val_split import training_val_split
import time
import json
# from scipy.lingalg import lstsq

def get_rounded(prediction):
    """This function rounds the predicted values to the nearest pain score.
    Input:
    - prediction: array of the predicted pain scores
    Output:
    - cv_y_pred_regres: A list with the rounded predictions, which are either 0, 1 or 2"""
    # create an empty list to assign the predictions to
    cv_y_pred_regres = []
    #iterate over all the predictions

    for i in prediction:
        #make an empty list to add the distances of the possible pain scores and the prediction
        distance_list = []
        #calculate all the possible distances of the pain score with the prediction and save it to the list
        distance_list.append(abs(0 - i))
        distance_list.append(abs(1 - i))
        distance_list.append(abs(2 - i))
        #append the index with the lowest distance to the prediction list
        cv_y_pred_regres.append(distance_list.index(min(distance_list)))
    #Return the list of rounded predictions
    return cv_y_pred_regres

def f1_scorer(true_values, prediction):
    """This function calculates the micro F1 score given a set of predictions and the true pain scores.
    Input:
    - true_values: The true pain scores given as an array
    - prediction: The predicted pain scores given as an array
    Output:
    - f1_micro: The micro F1 score of the prediction"""
    #Round the predictions to the nearest pain score
    cv_y_pred_regres = get_rounded(prediction)
    #Calculate the micro f1 score
    f1_micro = f1_score(true_values, cv_y_pred_regres, average = 'micro')
    #Return the micro F1 score
    return f1_micro

def f1_classification(true_values, prediction):
    """This function calculates the micro F1 score given a set of predictions of a classification model and the true pain scores.
    Input:
    - true_values: The true pain scores given as an array
    - prediction: The predicted pain scores given as an array
    Output:
    - f1_micro: The micro F1 score of the prediction"""
    f1_micro = f1_score(true_values, prediction, average = 'micro')
    return f1_micro

def mse_classifictaion(true_values, prediction):
    """This function calculates the Mean Squared Error (MSE) of the classified pain scores. The predictions are first binarized
    using [0,0] for class 0, [0,1] for class 1 and [1,1] for class 2. Then, from these binarized predictions, the MSE is calculated.
    Input:
    - true_values: an array containing the true pain scores
    - prediction: an array containing the predicted values by the classifier
    Output:
    - score: the MSE score of the binarized predictions """
    #Initiate empty lists to save the binarized results
    true_list = []
    pred_list = []
    #iterate over all the true pain scores
    for i in true_values:
        #Save the correct binarized array based on the pain scores and append it to the list
        if int(i) == 0:
            true_list.append([0,0])
        elif int(i) == 1:
            true_list.append([1,0])
        elif int(i) == 2:
            true_list.append([1,1])
        else:
            print('except true')
    #Iterate over all the predictions
    for i in prediction:
        #Save the correct binarized array based on the pain scores and append it to the list
        if int(i) == 0:
            pred_list.append([0,0])
        elif int(i) == 1:
            pred_list.append([1,0])
        elif int(i) == 2:
            pred_list.append([1,1])
        else:
            print('except predict')
    #Calculate the MSE of the binarized results
    score = mean_squared_error(np.array(true_list), np.array(pred_list))
    #Return the MSE
    return score

def SVM_classifier(animal, extraction_method, single_model = False):
        """This Function uses a SVM regressor to predict the pain scores. The SVM is trained on the training set, given a feature type. The hyperparameters are chosen by a 5-fold random search cross validation.
        It iterates over all the head poses and combines all the pain features together to predict the pain score.
        The input :
        - animal: could either be horse or None, when horse is given a model for specifically horses is craeted. Once None is given, it will train a model on both horses and donkeys
        - extraction_method: This is the feature type used to train the SVR. This could either be HOG, CNN, SIFT or LBP.
        - single_model: This is by default False. If set to True, it will train a SVR on all the head poses at once.

        The output:
        The mean squared error of the model per pain feature is saved in a txt file in the directory Confusion matrices/MSE/"""

        #A list containing all the pain features
        expression_list =  ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera', 'Corners of the mouth', 'Nostrils']


        #A list containing all the head poses
        head_pose = ['front', 'tilted', 'side']

        #If single model is set to True, no difference is given between the head poses
        if single_model == True:
            head_pose = [None]
        #iterate over all the head poses
        for str_head_pose in head_pose:
            #open the train file containing the expert scores
            y_train_full = pd.read_csv('Final/split/%s_train_horse.csv' % (str_head_pose))


            #repeat every row 3 times, due to the augmentation and save only the photoumbers
            y_train_photonumbers = np.repeat(y_train_full['photonumber'], 3)
            #open the test file containing the expert scores
            y_test_full = pd.read_csv('Final/split/%s_test_horse.csv' % (str_head_pose))

            #if the head pose is given as front...
            if str_head_pose == 'front':
                #exclude the mouth pain feature
                features = [2, 3, 4, 5, 7]

            #otherwise, predict all pain features
            else:
                features = [2, 3, 4, 5, 6, 7]

            #Iterate over all the pain features
            for predict_feature_index in features:
                #Select the correct number of ROIs per pain feature. If SIFT is given, select the correct number of landmarks given in the ROI.
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
                #Create the X_train data from the extracted features
                X_augment = create_HOG_dataframe_with_induced_noise(y_train_photonumbers, str_head_pose, roi_features, extraction_method = extraction_method, induced_noise = 0, single_model = single_model, animal = animal, split = 'train')
                data = np.array(X_augment)
                print(data.shape)

                #Create the X_test data from the extracted features
                X_nonaugment = create_HOG_dataframe_with_induced_noise(y_test_full['photonumber'], str_head_pose, roi_features, extraction_method = extraction_method, induced_noise = 0, single_model = single_model, animal = animal, split = 'test')

                #Convert the train and test data to an numpy array
                X_augmentated = np.array(X_augment)
                X_non_augmentated = np.array(X_nonaugment)

                #print the shapes of both the training and test data
                print('X augmentation: ', X_augmentated.shape)
                print('X non augmentation: ', X_non_augmentated.shape)
                # print('X augmentation: ', X_augmentated_landmarks.shape)
                #Save the missing values in the test file as a variable
                index_test = y_test_full.iloc[:, predict_feature_index].index[y_test_full.iloc[:, predict_feature_index].apply(np.isnan)]
                #Select the pain feature we would like to predict in the test set
                y_test = y_test_full.iloc[:, predict_feature_index]
                #Drop the missing values
                y_test = y_test.dropna(axis=0)
                #Remove the extracted features of the missing values in the scores
                X_test = np.delete(X_non_augmentated, index_test, 0)



                #Select the pain feature we would like to predict in the train set
                y_train = y_train_full.iloc[:, predict_feature_index]
                print('y_train pre: ', len(y_train))
                #Repeat every row 3 times, due to the augmentation
                y_train = np.repeat(y_train, 3)
                print('y_train post:', len(y_train))
                #Save the missing values to a variable
                index_train = y_train.index[y_train.apply(np.isnan)]
                #Create an empty list
                converted_index = []
                #initiate a counter
                counter = 0
                #Iterate over all the missing value indices
                for i in index_train:
                    #Multiply the index by 3 and add the counter to match the indices of the X_train data
                    index = i * 3 + counter
                    #Save the matching indices to the empty list
                    converted_index.append(index)
                    #Add 1 to the counter
                    counter += 1
                    #If counter == 3...
                    if counter == 3:
                        #set the counter to 0
                        counter = 0
                print(len(index_train))
                print(converted_index)
                #Drop the missing values from the y_train
                y_train = y_train.dropna(axis=0)
                #Remove the corresponding missing values from the X_train by the use of the newly created index
                X_train = np.delete(X_augmentated, converted_index, 0)
                # X_train_landmarks = np.delete(X_augmentated_landmarks, converted_index, 0)
                print(len(X_train), len(y_train))

                #If single model is given..
                if single_model == True:
                    params = {'kernel': [Linear()], 'C': [0.1, 1, 10, 100, 1000]}
                #Choose between these hyperparameters for the SVC
                if extraction_method == 'CNN':
                    params = ParameterGrid({'kernel': [Linear()], 'C': [0.1, 1, 10, 100, 1000]})

                else:
                    params = ParameterGrid({'kernel': [Linear(), GeneralizedHistogramIntersection(), Min()], 'C': [0.1, 1, 10, 100, 1000]})

                #convert both the X_train as the X_test to float64, to be able to be used in the SVM
                X_train = np.float64(X_train)
                X_test = np.float64(X_test)


                #Remove the index of both the training scores and the test scores
                y_train = y_train.reset_index(drop = True)
                y_test = y_test.reset_index(drop=True)


                # #Initiate a formula to calculate the sample weights.
                square = lambda vector: (vector ** 2 + 0.2)
                print(y_train)
                #Try to open a saved model
                try:
                    if single_model == True:
                        svregressor = pickle.load(open('Final/models/single_model/SVC_%s_%s_%s.sav' % (expression_list[predict_feature_index - 2], extraction_method, animal), 'rb'))
                    else:
                        svregressor = pickle.load(open('Final/models/SVC_%s_%s_%s_%s.sav' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal), 'rb'))
                    #Extract the hyperparameters of the model
                    best_reg_params = svregressor.get_params()

                    print('opened')
                #If no model exists yet, create a new one
                except:
                    #Select the binarized MSE as evaluation function for the cross-validation
                    cv_scorer = make_scorer(mse_classifictaion)
                    #Initiate a high number as best_error
                    best_score = 100000000
                    #Set the number of iterations to 0
                    k = 0
                    #Create an empty list
                    indices = []
                    #Iterate 5 times
                    while k < 5:
                        print('iteration: ', k)

                        #Create empty lists to save the training-validation split to
                        indices_X_train = []
                        indices_X_val = []
                        #Remove the missing values of the full dataframe
                        y_train_full_validation = y_train_full.drop(index_train)
                        #repeat the full training dataframe 3 times, due to augmentation
                        y_train_full_val = pd.DataFrame(np.repeat(y_train_full_validation.values, 3, axis = 0))
                        y_train_full_val.columns = y_train_full_validation.columns
                        #Fill the NANs to make the split
                        y_train_full_val = y_train_full_val.fillna('13')
                        #Make the training-validation split
                        y_val, y_train_val, val_indices, train_indices = training_val_split(y_train_full_val, indices)
                        #Save the indices of both the training and validation set
                        for i in val_indices:
                            indices_X_val.append(y_train_full_val.index[y_train_full_val['photonumber'] == i])
                        for i in train_indices:
                            indices_X_train.append(y_train_full_val.index[y_train_full_val['photonumber'] == i])
                        #Convert the indices to a 1D array
                        indices_X_train = np.unique(np.concatenate(indices_X_train, axis = 0))
                        indices_X_val = np.unique(np.concatenate(indices_X_val, axis = 0))
                        #Select the corresponding extracted features for the training set during validation
                        X_train_val = X_train[indices_X_train]
                        print(X_train_val.shape)
                        #Select the corresponding extracted features for the validation set
                        X_val = X_train[indices_X_val]
                        #Iterate over all possible hyperparameters
                        for param in params:
                            print(param)
                            #Initiate start time
                            tic = time.perf_counter()
                            #Initiate SVM classsifier with the selected hyperparameters
                            ecoc = SVC(**param)
                            #Fit the SVM to the training set without the validation set
                            ecoc.fit(X_train_val, y_train_val.iloc[:, predict_feature_index], sample_weight = y_train_val.iloc[:, predict_feature_index].apply(square))
                            #Precdict the validation set
                            prediction_val = ecoc.predict(X_val)
                            #calculate the binarized MSE of the validation prediction
                            prediction_error = mse_classifictaion(y_val.iloc[:, predict_feature_index], prediction_val)

                            #If the predicted error is lower than the best score...
                            if prediction_error < best_score:
                                #The prediction error is the new best score
                                best_score = prediction_error
                                #Save parameters
                                best_reg_params = param
                            #Select end time and print the time differencee between starting time and ending time
                            toc = time.perf_counter()
                            print(f"Cross validation fit in {toc - tic:0.4f} seconds")
                        #Select the selected validation indices as the new indices
                        indices = val_indices
                        print(best_score)
                        #Add 1 to the number of iterations
                        k += 1

                    print(expression_list[predict_feature_index - 2])


                    #Select SVM with the best performing hyperparameters

                    svregressor = SVC(**best_reg_params, probability = True)
                    #fit the model with the best selected hyperparameters to the training data with the sample weight function
                    svregressor.fit(X_train, y_train, sample_weight = y_train.apply(square))
                    if single_model == True:
                        #check directory already exists, if not create it
                        if os.path.isdir('Final/models/single_model') == False:
                            os.makedirs('Final/models/single_model')
                        filename = 'Final/models/single_model/SVC_%s_%s_%s.sav' % (expression_list[predict_feature_index - 2], extraction_method, animal)
                        #Save the trained model
                        pickle.dump(svregressor, open(filename, 'wb'))
                    else:

                        #check directory already exists, if not create it
                        if os.path.isdir('Final/models') == False:
                            os.makedirs('Final/models')
                        filename = 'Final/models/SVC_%s_%s_%s_%s.sav' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal)
                        #Save the trained model
                        pickle.dump(svregressor, open(filename, 'wb'))

                #check directory already exists, if not create it
                if os.path.isdir('Final/Evaluation') == False:
                    os.makedirs('Final/Evaluation')
                #open a txt file
                if animal != None:
                    if single_model == True:
                        try:
                            file = open('Final/Evaluation/confusion_single_model_%s_%s.txt' %(expression_list[predict_feature_index - 2], extraction_method), 'w')

                        except:
                            file = open('Final/Evaluation/confusion_single_model_%s_%s.txt' %(expression_list[predict_feature_index - 2], extraction_method, animal), 'w')


                    else:
                        file = open('Final/Evaluation/confusion_%s_%s_%s_%s.txt' %(str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal), 'w')

                else:
                    try:
                        file = open('Final/Evaluation/confusion_%s_%s_%s.txt' %(str_head_pose, expression_list[predict_feature_index - 2], extraction_method), 'w')

                    except:
                        file = open('Final/Evaluation/confusion_%s_%s_%s_horse.txt' %(str_head_pose, expression_list[predict_feature_index - 2], extraction_method), 'w')

                #predict the test set
                pred_test = svregressor.predict(X_test)

                #predict the training set
                pred_train = svregressor.predict(X_train)


                print('X_test: ', len(X_test), 'pred list: ', len(pred_test))
                #Calculate the micro F1 score of both the training and the test set
                error_test = f1_classification(y_test, pred_test)
                error_train = f1_classification(y_train, pred_train)
                #Calculate the binarized MSE of both the training and the test set
                mse_test = mse_classifictaion(y_test, pred_test)
                mse_train = mse_classifictaion(y_train, pred_train)
                #Create a confusion matrix of the test set prediction
                confusion = confusion_matrix(y_test, pred_test)
                #convert confusion matrix to string
                cm = np.array2string(confusion)
                #Save the confusion matrix to the text file
                file.write('Confusion Matrix\n\n{}\n'.format(cm))

                print('knn sanity check')
                #fit a 5NN classifier to the training data
                best_knn = KNeighborsClassifier()
                best_knn.fit(X_train, y_train)
                #predict the test set
                predict_knn = best_knn.predict(X_test)
                #Calculate the MSE of the 5NN
                knn_error = f1_classification(y_test, predict_knn)
                mse_knn = mse_classifictaion(y_test, predict_knn)

                #Save all results to a text file
                file.write('micro F1 Test\n {}\n'.format(error_test))
                file.write('micro F1 Training\n {}\n\n'.format(error_train))
                file.write('MSE Test\n {}\n'.format(mse_test))
                file.write('MSE Training\n {}\n\n'.format(mse_train))
                file.write('Selected hyperparameters \n {}\n\n'.format(best_reg_params))
                file.write('micro F1 5NN\n {}\n\n'.format(knn_error))
                file.write('MSE 5NN\n {}\n\n'.format(mse_knn))

                #Initiate a dummy classifier which predicts the majority vote
                dummy = DummyClassifier(strategy = 'most_frequent')
                #Fit it to the training set
                dummy.fit(X_train, y_train)
                #Predict the test set
                dummy_prediction = dummy.predict(X_test)
                #Calculate both the micro F1 score as the binarized MSE
                dummy_error = f1_classification(y_test, dummy_prediction)
                mse_dummy = mse_classifictaion(y_test, dummy_prediction)
                #Save the results to the text file
                file.write('F1 micro Dummy\n{}\n\n'.format(dummy_error))
                file.write('MSE Dummy\n{}\n\n'.format(mse_dummy))
                file.close()

                #create a dataframe for saving the raw test output
                test_results = pd.DataFrame(columns = ['photonumber'])
                #Remove the photonumbers containing missing values and save it as a variable
                save_y_test = y_test_full.drop(axis = 0, index = index_test)
                save_y_train = y_train_full.drop(axis = 0, index = index_train)
                #Repeat every training photonumber 3 times
                save_y_train = np.repeat(save_y_train['photonumber'], 3)
                #Save the test photonumbers to the dataframe
                test_results['photonumber'] = save_y_test.iloc[:, 0]
                if single_model == True:
                    #Create a name for the column
                    column_name = expression_list[predict_feature_index - 2] + '_' + extraction_method
                else:
                    #Create a name for the column
                    column_name = str_head_pose + '_' + expression_list[predict_feature_index - 2] + '_' + extraction_method
                #Save the test output to the dataframe
                test_results[column_name] = pred_test
                #Do the same for the training output
                train_results = pd.DataFrame(columns =['photonumber'])
                train_results['photonumber'] = save_y_train
                print(len(train_results['photonumber']), len(pred_train))
                train_results[column_name] = pred_train
                if single_model == True:
                    #check directory already exists, if not create it
                    if os.path.isdir('Final/raw_results/single_model') == False:
                        os.makedirs('Final/raw_results/single_model')

                    #Save both the training and test output to a csv file
                    test_results.to_csv('Final/raw_results/single_model/test_scores_%s_%s_%s_%s.csv' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal))
                    train_results.to_csv('Final/raw_results/single_model/train_scores_%s_%s_%s_%s.csv' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal))
                else:
                    #check directory already exists, if not create it
                    if os.path.isdir('Final/raw_results') == False:
                        os.makedirs('Final/raw_results')

                    #Save both the training and test output to a csv file
                    test_results.to_csv('Final/raw_results/test_scores_%s_%s_%s_%s.csv' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal))
                    train_results.to_csv('Final/raw_results/train_scores_%s_%s_%s_%s.csv' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal))
        return 5

def predict_total_score(input_head_pose = None, test_df = None):
    """This function predicts the total score based on the generated Regression output of all the Pain features and all the extracted feature models.
    It trains a new model no input is given and no model exist yet and the errors (MSE and R2) generated will be saved into a text file.
    Input:
    - input_head_pose: the head pose of the given image in string format, could either be 'tilted', 'side' or 'front'. When no head pose is given, a model to predict the
    total score will be trained
    - test_df: The regression output of the SVR as a dataframe
    Output:
    - prediction: The prediction of the total pain score made by the model.
    - pain: If the predicted total score exceeds the pain cutoff value (for now a dummy threshold set to 4) return if the horse is in pain or not
    """
    #If no head pose is specified, iterate over all head poses
    if input_head_pose == None:
        head_pose_list = ['side', 'front', 'tilted']
    #Otherwise, use the initiated head pose
    else:
        head_pose_list = [input_head_pose]
    #iterate over the head pose(s)
    for str_head_pose in head_pose_list:
        #A list containing all the pain features based on the head pose
        if str_head_pose == 'front':
            expression_list =  ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera', 'Nostrils']
        else:
            expression_list =  ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera', 'Corners of the mouth', 'Nostrils']
        print(str_head_pose)
        # if no head pose is not initially specified.
        if input_head_pose == None:
            #open the true values of the training and test set
            training_scores = pd.read_csv('Final/split/%s_train_horse.csv' % (str_head_pose))
            test_scores = pd.read_csv('Final/split/%s_test_horse.csv' % (str_head_pose))
            #Select the true total score of the training set
            y_train = training_scores.iloc[:, 8]
            #Repeat this score 3 times due to augmentation
            y_train = np.repeat(y_train, 3)
            #Select the true total score of the test set
            y_test = test_scores.iloc[:, 8]
            #Initate an empty dataframe
            train_df = pd.DataFrame()
            #Assign the training photonumber to the empty dataframe
            train_df['photonumber'] = np.repeat(training_scores.iloc[:, 0].tolist(), 3)
            #Initiate another empty dataframe
            test_df = pd.DataFrame()
            #Assign the test photonumbers to the empty dataframe
            test_df['photonumber'] = test_scores.iloc[:, 0]
            #Iterate over the extracted features
            for extraction_method in ['HOG', 'CNN', 'SIFT']:
                #Iterate over the pain features
                for feature in expression_list:
                    #Open the predicted result given the pain feature and the feature type extracted
                    regression_train = pd.read_csv('Final/raw_results/train_scores_%s_%s_%s_horse.csv' % (str_head_pose, feature, extraction_method))
                    #Initiate an empty list
                    index_list = []
                    #Initiate a counter
                    old_index = -1
                    #Iterate over all predicted pain scores
                    for index in regression_train.index:
                        #Set photonumber to new index
                        new_index = regression_train.iloc[index, 0]
                        #check if previous photonumber is the same as the current one
                        if new_index == old_index:
                            #If so, add one to the index value and save it to the list
                            index_value += 1
                            index_list.append(index_value)
                        else:
                            #Otherwise, multiply the index value by 3 and save it to the list
                            index_value = new_index * 3
                            index_list.append(index_value)
                        #Initiate the current photonumber as the old photonumber
                        old_index = new_index
                    #Set the index to the training set
                    regression_train['index_list'] = index_list
                    regression_train = regression_train.set_index('index_list', drop=True)
                    #Add the dataframe to the big dataframe
                    train_df = train_df.join(regression_train.iloc[:, 2])
                    #Fill the missing values with the column means
                    train_df = train_df.fillna(train_df.mean())

                    #open the predicted test set values
                    regression_test = pd.read_csv('Final/raw_results/test_scores_%s_%s_%s_horse.csv' % (str_head_pose, feature, extraction_method))
                    #Append these values to the big dataframe
                    test_df = test_df.join(regression_test.iloc[:, 2])
                    #Fill missing values by the column mean
                    test_df = test_df.fillna(test_df.mean())

            #drop the photonumber column
            test_df.drop('photonumber', axis =1, inplace = True)
            #Try to open an already saved model
            try:
                model = pickle.load(open('Final/models/total_score_linear_regression_%s.sav' % (str_head_pose), 'rb'))
                #Drop the photonumber column from the training set
                train_df.drop('photonumber', axis =1, inplace = True)
            #If no saved model exists yet..
            except:
                print('Gridsearch')
                #Save the missing values in the test file as a variable
                index_train = y_train.index[y_train.apply(np.isnan)]
                #initiate a counter
                old_index = -1
                #initiate an empty list
                index_train_remove_list = []
                #Iterate over all the training indices
                for index in index_train:
                    #If the previous photonumber is the same as the current photonumber..
                    if index == old_index:
                        #Add 1 to the index value and append it to the list
                        index_value += 1
                        index_train_remove_list.append(index_value)
                    #Else..
                    else:
                        #multiply the index value by 3 and append it to the list
                        index_value = index * 3
                        index_train_remove_list.append(index_value)
                    #save the current photonumber as the old photonumber
                    old_index = index
                #Drop the missing values from the test set
                index_test = y_test.index[y_test.apply(np.isnan)]
                #Drop the same indices from the test dataframe
                test_df.drop(index_test, axis =0, inplace = True)


                #Drop the missing values and reset the index of the training scores
                y_train = y_train.dropna(axis=0)
                y_train = y_train.reset_index()
                #Drop the first colum
                y_train = y_train.drop(y_train.columns[0], axis = 1)
                #Remove the extracted features of the missing values in the scores
                train_df.drop(index_train_remove_list, axis = 0, inplace = True)
                training_scores.drop(index_train_remove_list, axis = 0, inplace = True)
                #create a new dataframe containing all pain scores and multiply each row 3 times due to augmentation
                new_df = pd.DataFrame(np.repeat(training_scores.values, 3, axis =0))
                new_df.columns = training_scores.columns
                #reset the index of the new dataframe
                new_df = new_df.reset_index(drop = True)
                #Set the binarized MSE as cross-validation scoring metric
                cv_scorer = make_scorer(mean_squared_error)
                #Select hyperparameters
                params = ParameterGrid({'normalize': [True, False], 'alpha': [0.01, 0.1, 1, 10]})
                #Set number of iterations to 0
                k = 0
                #Set a high number as best score
                best_score = 1000000000
                #Create an empty list
                indices = []
                #Iterate 5 times
                while k < 5:
                    print('iteration: ', k)
                    #Create empty lists to assign the training-validation split to
                    indices_X_val = []
                    indices_X_train = []
                    #Make the training-validation split
                    y_val, y_train_val, indices_val, indices_train = training_val_split(new_df, indices)
                    print(len(y_train_val), len(indices_train))
                    #select the corresponding indices per training and validation set
                    for i in indices_val:
                        indices_X_val.append(new_df.index[new_df['photonumber'] == i])

                    for i in indices_train:

                        indices_X_train.append(new_df.index[new_df['photonumber'] == i])

                    #Create an array out of the training and validation indices
                    indices_X_train = np.unique(np.concatenate(indices_X_train, axis = 0))
                    indices_X_val = np.unique(np.concatenate(indices_X_val, axis = 0))
                    #Select the training set
                    X_train_val = train_df.iloc[train_df.index.isin(indices_X_train), :]
                    #Select the validation set
                    X_val = train_df.iloc[train_df.index.isin(indices_X_val), :]
                    #Drop the photonumber columns of both the training and the test set
                    X_train_val.drop('photonumber', axis = 1, inplace = True)
                    X_val.drop('photonumber', axis = 1, inplace = True)
                    #Iterate over all possible hyperparameter combinations
                    for param in params:
                        #Select LinearRegression model with penelization
                        model = Ridge(**param)
                        #Fit the model to the training set
                        try:
                            model.fit(X_train_val, y_train_val.iloc[:, 8])
                        except:
                            model.fit(X_train_val, y_train_val.iloc[:-3, 8])
                        #predict the validation set
                        prediction_val = model.predict(X_val)
                        #Calculate the binarized MSE
                        error = mean_squared_error(y_val.iloc[:, 8], prediction_val)
                        print(error)
                        #If the error is lower than the best score...
                        if error < best_score:
                            #Save the hyperparameters and set the error to the best_score
                            best_params = param
                            best_score = error
                    #Save the validation indices to the indices variable
                    indices = indices_val
                    #Add 1 to the number of iterations
                    k += 1


                print('X: ', len(train_df), 'y: ', len(y_train))

                #Check if path already exists, if not create it
                if os.path.isdir('Final/Evaluation/total_score') == False:
                    os.makedirs('Final/Evaluation/total_score')
                #open text file and save the selected hyperparameters
                with open('Final/Evaluation/total_score/%s_hyperparameters.txt' % (str_head_pose), 'w') as f:
                    f.write('Hyperparameters: {}\n'.format(best_params))
                #drop the photonumber column of the training set
                train_df.drop('photonumber', axis = 1, inplace = True)
                # Create model with the best hyperparameters and fit it to the trainingset
                model = Ridge(**best_params).fit(train_df, y_train)
                #Check if path already exists, if not create it
                if os.path.isdir('Final/models') == False:
                    os.makedirs('Final/models')
                #Save the model
                pickle.dump(model, open('Final/models/total_score_linear_regression_%s.sav' % (str_head_pose), 'wb'))

        else:
            #Load the saved model
            model = pickle.load(open('Final/models/total_score_linear_regression_%s.sav' % (str_head_pose), 'rb'))
        #If no head pose is specified
        if input_head_pose == None:

            #Save the missing values in the test file as a variable
            index_test = y_test.index[y_test.apply(np.isnan)]
            #Drop the missing values
            y_test = y_test.dropna(axis=0)
            #Remove the extracted features of the missing values in the scores
            test_df.drop(index_test, axis = 0, inplace = True)
        #predict the test set and the training set
        prediction = model.predict(test_df)
        prediction_train = model.predict(train_df)
        #If no head pose is specified
        if input_head_pose == None:
            #calculate the MSE and the R2 score
            error = mean_squared_error(y_test, prediction)
            r2 = r2_score(y_test, prediction)
            print('MSE: ', error, 'R2: ', r2)
            #Check if path already exists, if not create it
            if os.path.isdir('Final/Evaluation/total_score') == False:
                os.makedirs('Final/Evaluation/total_score')
            #open text file
            with open('Final/Evaluation/total_score/%s_total_error.txt' % (str_head_pose), 'w') as f:
                #Save the results
                f.write('Mean squared error: {}\n'.format(str(error)))
                f.write('R2 score: {}'.format(str(r2)))
            #Save the error based on the head pose
            if str_head_pose == 'front':
                front_error = error
            elif str_head_pose == 'side':
                side_error = error
            elif str_head_pose == 'tilted':
                tilted_error = error
        #Initiate an empty list
        pain = []
        #Iterate over all predictions
        for i in prediction:
            #If total score is bigger than 4, the save in pain, else save not in pain
            if i > 4:
                pain.append('in pain')
            else:
                pain.append('not in pain')
    #If no head pose is specified
    if input_head_pose == None:
        #Plot a bar plot of the MSE error
        objects = ('Front', 'Tilted', 'Profile')
        y_pos = np.arange(len(objects))
        performance = [front_error, tilted_error, side_error]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('MSE')
        #Save the plot
        plt.savefig('figures/total/total_score.png', bbox='tight')
        #Show the plot
        plt.show()
    return prediction, pain

def results_combined():
    """This function opens the saved json files of the combined pipeline and calculates the error.
    No input
    Output:
    Text file per head pose showing the MSE per pain feature + the total score"""
    #Iterate over all head poses
    for str_head_pose in ['tilted', 'front', 'side']:
        #Initiate an empty list per pain feature and the total score
        ears = []
        orbital = []
        eyelid = []
        sclera = []
        mouth = []
        nose = []
        total = []
        #Create a list for the missing predictions
        missing = []
        #Open the true scores
        scores = pd.read_csv('Final/split/%s_test_horse.csv' % (str_head_pose))
        #Select the photonumbers of the head pose
        photonumbers = scores.iloc[:, 0]
        #Open text file to save the results to
        with open('Final/Evaluation/combined_results_%s.txt' % (str_head_pose), 'w') as f:
            #Iterate over all the photoumbers of the head pose
            for photonumber in photonumbers:
                #Try to open the combined prediction
                try:
                    with open('Final/dataset/images/test/%s.txt'  % (photonumber)) as json_file:
                        data = json.load(json_file)

                        #Select the pain scores
                        pain = data['pain']
                        #Append the pain score per pain feature to the corresponding list
                        ears.append(pain['Ears']['score'])
                        orbital.append(pain['Orbital tightning']['score'])
                        eyelid.append(pain['Angulated upper eyelid']['score'])
                        sclera.append(pain['Sclera']['score'])
                        #Skip the mouth pain feature if the front head pose is given
                        if str_head_pose != 'front':
                            try:
                                mouth.append(pain['Corners of the mouth']['score'])
                            except:
                                print('except')
                        nose.append(pain['Nostrils']['score'])
                        total.append(pain['Total pain']['pain level'])
                #If no prediction exists..
                except:
                    #Save the photonumber to the missing list and print the photonumber
                    missing.append(photonumber)
                    print(photonumber)
                    #Continue the loop
                    continue
            #Remove the scores with no prediction
            scores = scores[~scores['photonumber'].isin(missing)]
            #Create a list of pain features with the corresponding predictions based on the head pose
            if str_head_pose == 'front':
                iterator = [('Ears', 2, ears), ('Orbital', 3, orbital), ('Eyelid', 4, eyelid), ('Sclera', 5, sclera), ('Nostrils', 7, nose), ('Total score', 8, total)]
            else:
                iterator = [('Ears', 2, ears), ('Orbital', 3, orbital), ('Eyelid', 4, eyelid), ('Sclera', 5, sclera), ('Mouth', 6, mouth), ('Nostrils', 7, nose), ('Total score', 8 , total)]
            #Iterate over the scores and predictions
            for pain_feature, predict_feature_index, prediction in iterator:

                print(str_head_pose, pain_feature)
                #Save the indices of the missing values in the true pain scores
                index_test = scores.iloc[:, predict_feature_index].index[scores.iloc[:, predict_feature_index].apply(np.isnan).values].tolist()
                #Select the pain feature we would like to predict in the test set
                y_test = scores.iloc[:, predict_feature_index]
                #Drop the missing values
                y_test = y_test.dropna(axis=0)
                print(index_test, print(len(prediction)))
                #if there are missing values...
                if len(index_test) != 0:
                    #Initiate counter
                    counter = 0
                    #iterate over the missing value indices
                    for i in index_test:
                        print(i)
                        #Remove these indices from the prediction
                        prediction.pop(i - counter)
                        counter += 1
                #Calculate the MSE of the predictions
                error = mean_squared_error(y_test, prediction)
                #Save the error to the text file
                f.write('MSE {}:\t{}\n'.format(pain_feature, str(error)))
